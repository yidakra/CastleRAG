#!/bin/bash
# smoke_min_allinone.slurm  (Stage-1 validation — NOT part of the production chain)
# Runs the ENTIRE real-data pipeline for the minimal scope (configs/smoke_min.yaml:
# 2 ego cams, day 1, hour 08) inside ONE gpu_a100 quarter-node, hosting both vLLM
# models on the single A100.  Validates pipeline + models end-to-end at ~1 A100-h.
#
#   sbatch --account=gpuuva082 scripts/slurm/smoke_min_allinone.slurm

#SBATCH --job-name=castle-smoke-min
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

CONF="configs/smoke_min.yaml"
SC="/scratch-shared/${USER}"
QFILE="${SC}/questions.json"
export HF_HOME="${SC}/hf_cache"
export HF_HUB_OFFLINE=1            # models pre-cached; no compute-node internet needed
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_LOGGING_LEVEL=WARNING

echo "[smoke-min] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1) start=$(date -Iseconds)"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load FFmpeg/7.0.2-GCCcore-13.3.0
source "${HOME}/castlerag_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p logs

wait_ready() { # url label logfile [pid]
  local pid="${4:-}"
  for i in $(seq 1 120); do
    curl -sf "$1" >/dev/null 2>&1 && { echo "[smoke-min] $2 ready ($((i*10))s)"; return 0; }
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[smoke-min] ERROR: $2 server process died"; tail -30 "$3" 2>/dev/null; return 1
    fi
    sleep 10
  done
  echo "[smoke-min] ERROR: $2 not ready after 1200s"; tail -30 "$3" 2>/dev/null; return 1
}

# Nodes here are A100-40GB, so give each model its own GPU (GPU0=OmniEmbed, GPU1=Qwen3-VL).
# --- OmniEmbed embeddings server (custom transformers server; GPU0, port 8200) ---
EMB_LOG="logs/omniembed_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0 python scripts/omniembed_server.py --port 8200 > "${EMB_LOG}" 2>&1 &
EMB_PID=$!
trap 'kill $EMB_PID 2>/dev/null' EXIT

# --- Qwen3-VL generation server (vLLM; GPU1, port 8201) ---
GEN_LOG="logs/vllm_gen_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8201 \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --trust-remote-code \
    > "${GEN_LOG}" 2>&1 &
GEN_PID=$!
trap 'kill $EMB_PID $GEN_PID 2>/dev/null' EXIT
wait_ready "http://localhost:8200/v1/models" embed "${EMB_LOG}" $EMB_PID
trap 'kill $EMB_PID $GEN_PID 2>/dev/null' EXIT

# --- Qdrant (storage on shared scratch) ---
export QDRANT__STORAGE__STORAGE_PATH="${SC}/qdrant_storage/storage"
"${HOME}/qdrant/qdrant" > "logs/qdrant_${SLURM_JOB_ID}.log" 2>&1 &
QPID=$!
trap 'kill $EMB_PID $GEN_PID $QPID 2>/dev/null' EXIT

wait_ready "http://localhost:6333/readyz" qdrant "logs/qdrant_${SLURM_JOB_ID}.log" $QPID
wait_ready "http://localhost:8201/v1/models" gen "${GEN_LOG}" $GEN_PID

# --- offline build ---
echo "[smoke-min] === preprocess (caption+events) ==="
export VLLM_BASE_URL="http://localhost:8201/v1"          # vision model for caption/events
castlerag preprocess --config "${CONF}" --day 1 --caption --events
echo "[smoke-min] === embed ==="
export VLLM_BASE_URL="http://localhost:8200/v1"          # OmniEmbed for passages
castlerag embed --config "${CONF}" --day 1
echo "[smoke-min] === index ==="
castlerag index --config "${CONF}" --day 1 --create-collection

# --- eval: query vectors from cache (OmniEmbed), generation via Qwen3-VL ---
echo "[smoke-min] === precompute query cache ==="
OMNIEMBED_URL="http://localhost:8200/v1" \
  python scripts/precompute_queries.py "${QFILE}" \
    --model Tevatron/OmniEmbed-v0.1-multivent \
    --out "${SC}/castle_derived/embeddings/query_cache.npz"
echo "[smoke-min] === smoke-test ==="
export VLLM_BASE_URL="http://localhost:8201/v1"
export OMNIEMBED_QUERY_CACHE="${SC}/castle_derived/embeddings/query_cache.npz"
castlerag smoke-test "${QFILE}" --n 5 --config "${CONF}"

echo "[smoke-min] DONE $(date -Iseconds)"
echo "[smoke-min] outputs under ${SC}/castle_outputs/smoke_test/"
