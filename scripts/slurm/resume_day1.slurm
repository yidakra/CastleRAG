#!/bin/bash
# resume_day1.slurm — finish the timed-out full day-1 build.
# Reuses base (all 10 cams) + captions for the 6 done cams. Re-captions the 4 incomplete
# cams (Onanong/Stevan/Tien/Werner), regenerates the corrupt Klaus/11, then runs --events
# ONCE single-process (the cross-camera race that corrupted the first run), embed, index, eval.
#   sbatch --account=gpuuva082 scripts/slurm/resume_day1.slurm

#SBATCH --job-name=castle-resume-day1
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
QFILE="${SC}/questions.json"
FULLCONF="configs/snellius_me.yaml"
export HF_HOME="${SC}/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_LOGGING_LEVEL=WARNING
J="${SLURM_JOB_ID}"
echo "[resume] host=$(hostname) start=$(date -Iseconds)"; nvidia-smi -L 2>/dev/null|head -3

module purge; module load 2024; module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 FFmpeg/7.0.2-GCCcore-13.3.0
source "${HOME}/castlerag_venv/bin/activate"; cd /gpfs/home2/scur0225/CastleRAG; mkdir -p logs

wait_ready(){ for i in $(seq 1 150); do curl -sf "http://localhost:$1/v1/models">/dev/null 2>&1 && { echo "[resume] $2 ready ($((i*10))s)";return 0;}; sleep 10; done; echo "[resume] $2 NOT ready"; tail -30 "$3"; return 1; }

CUDA_VISIBLE_DEVICES=2 python scripts/omniembed_server.py --port 8200 > "logs/r_omni_${J}.log" 2>&1 & EMB=$!
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8201 --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl --gpu-memory-utilization 0.85 --max-model-len 4096 --trust-remote-code > "logs/r_gen0_${J}.log" 2>&1 & G0=$!
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8202 --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl --gpu-memory-utilization 0.85 --max-model-len 4096 --trust-remote-code > "logs/r_gen1_${J}.log" 2>&1 & G1=$!
export QDRANT__STORAGE__STORAGE_PATH="${SC}/qdrant_storage/storage"
"${HOME}/qdrant/qdrant" > "logs/r_qdrant_${J}.log" 2>&1 & QP=$!
trap 'kill $EMB $G0 $G1 $QP 2>/dev/null' EXIT
wait_ready 8201 gen0 "logs/r_gen0_${J}.log" || exit 1
wait_ready 8202 gen1 "logs/r_gen1_${J}.log" || exit 1
sleep 6; curl -sf http://localhost:6333/readyz>/dev/null && echo "[resume] qdrant ready"

# --- Phase 1: re-caption the 4 incomplete cams (skip-base) + regenerate corrupt Klaus/11 (base+caption). NO --events. ---
echo "[resume] === Phase 1: caption $(date -Iseconds) ==="
declare -a P
# task: "<label>|<config>|<extra-flags>|<port>"
TASKS=(
  "Onanong|configs/_full/day1_Onanong.yaml|--skip-base --caption|8201"
  "Stevan|configs/_full/day1_Stevan.yaml|--skip-base --caption|8202"
  "Tien|configs/_full/day1_Tien.yaml|--skip-base --caption|8201"
  "Werner|configs/_full/day1_Werner.yaml|--skip-base --caption|8202"
  "Klaus_h11|configs/_full/day1_Klaus_h11.yaml|--caption|8201"
)
for t in "${TASKS[@]}"; do
  IFS='|' read -r lbl cfg flags port <<< "$t"
  ( T0=$(date +%s)
    VLLM_BASE_URL="http://localhost:${port}/v1" castlerag preprocess --config "$cfg" --day 1 $flags > "logs/r_cam_${lbl}_${J}.log" 2>&1
    echo "[resume] $lbl rc=$? in $(( $(date +%s)-T0 ))s" ) & P+=($!)
done
FAIL=0; for p in "${P[@]}"; do wait "$p" || FAIL=1; done
echo "[resume] === Phase 1 done $(date -Iseconds) FAIL=$FAIL ==="

# --- Phase 1b: validate ALL clips.jsonl parse + are captioned (abort events if any corrupt) ---
echo "[resume] === validate chunks ==="
python3 - <<'PY' || { echo "[resume] VALIDATION FAILED — not proceeding to events/embed"; exit 1; }
import glob,json,os,sys
files=glob.glob(f"/scratch-shared/{os.environ['USER']}/castle_derived/chunks/day1/*/*/clips.jsonl")
bad=[]; tot=cap=0
for f in files:
    try:
        for line in open(f):
            line=line.strip()
            if not line: continue
            d=json.loads(line); tot+=1
            if d.get("clip_caption"): cap+=1
    except Exception as e:
        bad.append(f)
print(f"files={len(files)} clips={tot} captioned={cap} ({100*cap/max(tot,1):.0f}%) corrupt={len(bad)}")
if bad: print("CORRUPT:", bad[:10]); sys.exit(1)
if cap < 0.98*tot: print(f"WARN: only {100*cap/tot:.0f}% captioned");
sys.exit(0)
PY

# --- Phase 2: events ONCE, single process (reads all cams; no concurrent caption writes now) ---
echo "[resume] === Phase 2: events (single process) $(date -Iseconds) ==="
VLLM_BASE_URL="http://localhost:8201/v1" castlerag preprocess --config "$FULLCONF" --day 1 --skip-base --events || { echo "[resume] events FAILED"; exit 1; }

# --- Phase 3: embed + index over all 10 cams ---
wait_ready 8200 embed "logs/r_omni_${J}.log" || exit 1
export VLLM_BASE_URL="http://localhost:8200/v1"
echo "[resume] === embed $(date -Iseconds) ==="
castlerag embed --config "$FULLCONF" --day 1 || { echo "[resume] embed FAILED"; exit 1; }
echo "[resume] === index $(date -Iseconds) ==="
castlerag index --config "$FULLCONF" --day 1 --create-collection || { echo "[resume] index FAILED"; exit 1; }

# --- Phase 4: eval ---
echo "[resume] === precompute + smoke-test $(date -Iseconds) ==="
OMNIEMBED_URL="http://localhost:8200/v1" python scripts/precompute_queries.py "$QFILE" --model Tevatron/OmniEmbed-v0.1-multivent --out "${SC}/castle_derived/embeddings/query_cache.npz"
export VLLM_BASE_URL="http://localhost:8201/v1"
export OMNIEMBED_QUERY_CACHE="${SC}/castle_derived/embeddings/query_cache.npz"
castlerag smoke-test "$QFILE" --n 5 --config "$FULLCONF"
echo "[resume] DONE $(date -Iseconds)  outputs: ${SC}/castle_outputs/smoke_test/"
