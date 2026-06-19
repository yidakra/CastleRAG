#!/bin/bash
# resume2_day1.slurm — finish day-1 build with ISOLATED-DIR parallel captioning (safe).
# caption/events are NOT camera-scoped (they rglob+rewrite ALL clips.jsonl under chunks_dir),
# so each of the 5 cameras needing captions gets its OWN chunks_dir (chunks_iso/<cam>) →
# its caption process only ever touches its own files. Frames are read by absolute path
# (shared). After caption, copy captions back to the shared tree, then run --events ONCE
# (single process), embed, index, smoke-test.
#   sbatch --account=gpuuva082 scripts/slurm/resume2_day1.slurm

#SBATCH --job-name=castle-resume2
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=360G
#SBATCH --gres=gpu:3
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
SC="/scratch-shared/${USER}"
ISO="${SC}/castle_derived/chunks_iso"
SHARED="${SC}/castle_derived/chunks"
QFILE="${SC}/questions.json"
FULLCONF="configs/snellius_me.yaml"
CAMS=(Onanong Stevan Tien Werner Klaus)
export HF_HOME="${SC}/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_LOGGING_LEVEL=WARNING
J="${SLURM_JOB_ID}"
echo "[r2] host=$(hostname) start=$(date -Iseconds)"; nvidia-smi -L 2>/dev/null|head -3
module purge; module load 2024; module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 FFmpeg/7.0.2-GCCcore-13.3.0
source "${HOME}/castlerag_venv/bin/activate"; cd /gpfs/home2/scur0225/CastleRAG; mkdir -p logs

wait_ready(){ for i in $(seq 1 150); do curl -sf "http://localhost:$1/v1/models">/dev/null 2>&1 && { echo "[r2] $2 ready ($((i*10))s)";return 0;}; sleep 10; done; echo "[r2] $2 NOT ready"; tail -30 "$3"; return 1; }

CUDA_VISIBLE_DEVICES=2 python scripts/omniembed_server.py --port 8200 > "logs/r2_omni_${J}.log" 2>&1 & EMB=$!
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8201 --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl --gpu-memory-utilization 0.85 --max-model-len 4096 --trust-remote-code > "logs/r2_gen0_${J}.log" 2>&1 & G0=$!
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8202 --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl --gpu-memory-utilization 0.85 --max-model-len 4096 --trust-remote-code > "logs/r2_gen1_${J}.log" 2>&1 & G1=$!
export QDRANT__STORAGE__STORAGE_PATH="${SC}/qdrant_storage/storage"
"${HOME}/qdrant/qdrant" > "logs/r2_qdrant_${J}.log" 2>&1 & QP=$!
trap 'kill $EMB $G0 $G1 $QP 2>/dev/null' EXIT
wait_ready 8201 gen0 "logs/r2_gen0_${J}.log" || exit 1
wait_ready 8202 gen1 "logs/r2_gen1_${J}.log" || exit 1
sleep 6; curl -sf http://localhost:6333/readyz>/dev/null && echo "[r2] qdrant ready"

# --- Phase 1: parallel caption, each camera in its ISOLATED chunks_dir ---
echo "[r2] === Phase 1: isolated caption $(date -Iseconds) ==="
declare -a P; idx=0
for cam in "${CAMS[@]}"; do
  port=$((8201 + idx % 2)); idx=$((idx+1))
  ( T0=$(date +%s)
    VLLM_BASE_URL="http://localhost:${port}/v1" castlerag preprocess --config "configs/_full/iso_${cam}.yaml" --day 1 --skip-base --caption > "logs/r2_cam_${cam}_${J}.log" 2>&1
    echo "[r2] $cam rc=$? in $(( $(date +%s)-T0 ))s" ) & P+=($!)
done
FAIL=0; for p in "${P[@]}"; do wait "$p" || FAIL=1; done
echo "[r2] === Phase 1 done $(date -Iseconds) FAIL=$FAIL ==="

# --- Validate isolated dirs fully captioned + clean ---
python3 - "${ISO}" "${CAMS[@]}" <<'PY' || { echo "[r2] ISO VALIDATION FAILED"; exit 1; }
import glob,json,sys
iso=sys.argv[1]; cams=sys.argv[2:]
ok=True
for cam in cams:
    tot=cap=bad=0
    for f in glob.glob(f"{iso}/{cam}/day1/{cam}/*/clips.jsonl"):
        for l in open(f):
            l=l.strip()
            if not l: continue
            try:
                d=json.loads(l); tot+=1
                if d.get("clip_caption"): cap+=1
            except: bad+=1
    pct=100*cap/max(tot,1)
    print(f"  {cam}: {cap}/{tot} captioned ({pct:.0f}%) bad={bad}")
    if bad or pct<98: ok=False
sys.exit(0 if ok else 1)
PY

# --- Copy captioned clips.jsonl back into the shared tree ---
echo "[r2] === copy captions back to shared ==="
for cam in "${CAMS[@]}"; do
  cp -rf "${ISO}/${cam}/day1/${cam}/." "${SHARED}/day1/${cam}/" && echo "  $cam copied back"
done

# --- Validate shared tree: all clean, ~100% captioned ---
python3 - "${SHARED}/day1" <<'PY' || { echo "[r2] SHARED VALIDATION FAILED"; exit 1; }
import glob,json,sys
base=sys.argv[1]; tot=cap=bad=0
for f in glob.glob(f"{base}/*/*/clips.jsonl"):
    for l in open(f):
        l=l.strip()
        if not l: continue
        try:
            d=json.loads(l); tot+=1
            if d.get("clip_caption"): cap+=1
        except: bad+=1
print(f"  shared: clips={tot} captioned={cap} ({100*cap/max(tot,1):.1f}%) bad={bad}")
sys.exit(0 if bad==0 and cap>=0.98*tot else 1)
PY

# --- Phase 2: events ONCE, single process over all cameras ---
echo "[r2] === Phase 2: events (single process) $(date -Iseconds) ==="
VLLM_BASE_URL="http://localhost:8201/v1" castlerag preprocess --config "$FULLCONF" --day 1 --skip-base --events || { echo "[r2] events FAILED"; exit 1; }

# --- Phase 3: embed + index ---
wait_ready 8200 embed "logs/r2_omni_${J}.log" || exit 1
export VLLM_BASE_URL="http://localhost:8200/v1"
echo "[r2] === embed $(date -Iseconds) ==="; castlerag embed --config "$FULLCONF" --day 1 || { echo "[r2] embed FAILED"; exit 1; }
echo "[r2] === index $(date -Iseconds) ==="; castlerag index --config "$FULLCONF" --day 1 --create-collection || { echo "[r2] index FAILED"; exit 1; }

# --- Phase 4: eval ---
echo "[r2] === precompute + smoke-test $(date -Iseconds) ==="
OMNIEMBED_URL="http://localhost:8200/v1" python scripts/precompute_queries.py "$QFILE" --model Tevatron/OmniEmbed-v0.1-multivent --out "${SC}/castle_derived/embeddings/query_cache.npz"
export VLLM_BASE_URL="http://localhost:8201/v1"
export OMNIEMBED_QUERY_CACHE="${SC}/castle_derived/embeddings/query_cache.npz"
castlerag smoke-test "$QFILE" --n 5 --config "$FULLCONF"
echo "[r2] DONE $(date -Iseconds)  outputs: ${SC}/castle_outputs/smoke_test/"
