#!/bin/bash
# full_day1.slurm — full day-1 ego build (all 10 ego cams, all hours) in ONE job.
# Captioning is sequential per client, so we run all 10 cameras as PARALLEL preprocess
# clients against 2 Qwen3-VL servers (vLLM batches the concurrent requests) + 1 custom
# OmniEmbed server, then embed -> index -> smoke-test.  3 GPUs on one node.
#
#   sbatch --account=gpuuva082 scripts/slurm/full_day1.slurm

#SBATCH --job-name=castle-full-day1
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=360G
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail   # NOTE: no -e; we manage per-camera failures explicitly

SC="/scratch-shared/${USER}"
QFILE="${SC}/questions.json"
FULLCONF="configs/snellius_me.yaml"        # all 10 cams — used for embed/index/eval
export HF_HOME="${SC}/hf_cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_LOGGING_LEVEL=WARNING
J="${SLURM_JOB_ID}"
CAMS=(Allie Bjorn Cathal Florian Klaus Luca Onanong Stevan Tien Werner)

echo "[full] host=$(hostname) start=$(date -Iseconds)"
nvidia-smi -L 2>/dev/null | head -3

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load FFmpeg/7.0.2-GCCcore-13.3.0
source "${HOME}/castlerag_venv/bin/activate"
cd /gpfs/home2/scur0225/CastleRAG
mkdir -p logs

wait_ready() { # port label logfile
  for i in $(seq 1 150); do
    curl -sf "http://localhost:$1/v1/models" >/dev/null 2>&1 && { echo "[full] $2 ready ($((i*10))s)"; return 0; }
    sleep 10
  done
  echo "[full] ERROR: $2 not ready"; tail -30 "$3" 2>/dev/null; return 1
}

# --- GPU2: OmniEmbed server (needed for the embed stage) ---
CUDA_VISIBLE_DEVICES=2 python scripts/omniembed_server.py --port 8200 > "logs/omni_${J}.log" 2>&1 &
EMB_PID=$!
# --- GPU0 + GPU1: two Qwen3-VL servers for batched captioning (short prompts -> small KV) ---
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8201 \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl \
    --gpu-memory-utilization 0.85 --max-model-len 4096 --trust-remote-code \
    > "logs/gen0_${J}.log" 2>&1 &
G0=$!
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8202 \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl \
    --gpu-memory-utilization 0.85 --max-model-len 4096 --trust-remote-code \
    > "logs/gen1_${J}.log" 2>&1 &
G1=$!
# --- Qdrant (storage on shared scratch) ---
export QDRANT__STORAGE__STORAGE_PATH="${SC}/qdrant_storage/storage"
"${HOME}/qdrant/qdrant" > "logs/qdrant_${J}.log" 2>&1 &
QP=$!
trap 'kill $EMB_PID $G0 $G1 $QP 2>/dev/null' EXIT

wait_ready 8201 gen0 "logs/gen0_${J}.log" || exit 1
wait_ready 8202 gen1 "logs/gen1_${J}.log" || exit 1
sleep 6; curl -sf http://localhost:6333/readyz >/dev/null && echo "[full] qdrant ready"

# --- FAN OUT: all 10 cameras in parallel, split across the 2 gen servers ---
echo "[full] === preprocess fan-out: 10 cameras (caption+events) $(date -Iseconds) ==="
declare -a PIDS CAMPID
for idx in "${!CAMS[@]}"; do
  cam="${CAMS[$idx]}"
  port=$((8201 + idx % 2))                 # even idx -> 8201, odd -> 8202 (5 each)
  ( T0=$(date +%s)
    VLLM_BASE_URL="http://localhost:${port}/v1" \
      castlerag preprocess --config "configs/_full/day1_${cam}.yaml" --day 1 --caption --events \
      > "logs/cam_${cam}_${J}.log" 2>&1
    rc=$?
    echo "[full] camera ${cam} (port ${port}) rc=${rc} in $(( $(date +%s)-T0 ))s"
  ) &
  PIDS+=($!); CAMPID+=("$cam")
done
FAIL=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "[full] CAMERA ${CAMPID[$i]} FAILED"; FAIL=1; }
done
echo "[full] === fan-out done $(date -Iseconds) (FAIL=${FAIL}) ==="
[ "$FAIL" -ne 0 ] && echo "[full] WARNING: >=1 camera failed; continuing with what succeeded"

# --- embed (OmniEmbed, GPU2) over ALL cameras' chunks, then index ---
wait_ready 8200 embed "logs/omni_${J}.log" || exit 1
export VLLM_BASE_URL="http://localhost:8200/v1"
echo "[full] === embed $(date -Iseconds) ==="
castlerag embed --config "${FULLCONF}" --day 1 || { echo "[full] embed FAILED"; exit 1; }
echo "[full] === index $(date -Iseconds) ==="
castlerag index --config "${FULLCONF}" --day 1 --create-collection || { echo "[full] index FAILED"; exit 1; }

# --- eval: query cache from OmniEmbed, generation via Qwen3-VL ---
echo "[full] === precompute query cache $(date -Iseconds) ==="
OMNIEMBED_URL="http://localhost:8200/v1" \
  python scripts/precompute_queries.py "${QFILE}" \
    --model Tevatron/OmniEmbed-v0.1-multivent \
    --out "${SC}/castle_derived/embeddings/query_cache.npz"
echo "[full] === smoke-test $(date -Iseconds) ==="
export VLLM_BASE_URL="http://localhost:8201/v1"
export OMNIEMBED_QUERY_CACHE="${SC}/castle_derived/embeddings/query_cache.npz"
castlerag smoke-test "${QFILE}" --n 5 --config "${FULLCONF}"
echo "[full] DONE $(date -Iseconds)  outputs: ${SC}/castle_outputs/smoke_test/"
