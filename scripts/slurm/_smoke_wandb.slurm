#!/bin/bash
# _smoke_wandb.slurm — NON-DESTRUCTIVE 5-question end-to-end smoke test that
# logs the run to Weights & Biases (foundationmodels/castlerag).
#
# Boots OmniEmbed (:8200) + Qwen3-VL (:8201) + Qdrant against the EXISTING
# persisted castle_multimodal_v1 index and runs ONLY `castlerag smoke-test
# --wandb` (no preprocess / embed / index / create-collection).  Aborts if the
# collection looks emptied.  Queries embed LIVE via OmniEmbed (no query cache),
# so fresh day-1 questions from the bank work.
#
#   sbatch --account=gpuuva082 scripts/slurm/_smoke_wandb.slurm

#SBATCH --job-name=castle-smoke-wandb
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --time=01:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

CONF="configs/snellius_me.yaml"
SC="/scratch-shared/${USER}"
QFILE="data/smoke_day1.csv"          # 40 day-1 questions w/ ground truth
N="${N:-5}"                          # how many to run; override via `sbatch --export=ALL,N=40`
COLL="castle_multimodal_v1"
export HF_HOME="${SC}/hf_cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_LOGGING_LEVEL=WARNING

# --- W&B: api key lives in shared ~/.netrc; keep service from hanging on init ---
export WANDB__SERVICE_WAIT=300

echo "[smoke] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1) start=$(date -Iseconds)"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load FFmpeg/7.0.2-GCCcore-13.3.0
source "${HOME}/castlerag_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p logs

wait_ready() { # url label logfile pid
  local pid="${4:-}"
  for i in $(seq 1 120); do
    curl -sf "$1" >/dev/null 2>&1 && { echo "[smoke] $2 ready ($((i*10))s)"; return 0; }
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[smoke] ERROR: $2 server died"; tail -30 "$3" 2>/dev/null; return 1
    fi
    sleep 10
  done
  echo "[smoke] ERROR: $2 not ready after 1200s"; tail -30 "$3" 2>/dev/null; return 1
}

# --- OmniEmbed (GPU0:8200) — embeds the fresh smoke queries live ---
EMB_LOG="logs/smoke_omniembed_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0 python scripts/omniembed_server.py --port 8200 > "${EMB_LOG}" 2>&1 &
EMB_PID=$!
# --- Qwen3-VL generation/rerank (GPU1:8201) ---
GEN_LOG="logs/smoke_vllm_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8201 --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --trust-remote-code \
    > "${GEN_LOG}" 2>&1 &
GEN_PID=$!
# --- Qdrant against EXISTING persisted storage (NO create-collection) ---
export QDRANT__STORAGE__STORAGE_PATH="${SC}/qdrant_storage/storage"
QLOG="logs/smoke_qdrant_${SLURM_JOB_ID}.log"
"${HOME}/qdrant/qdrant" > "${QLOG}" 2>&1 &
QPID=$!
trap 'kill $EMB_PID $GEN_PID $QPID 2>/dev/null' EXIT

wait_ready "http://localhost:6333/readyz"     qdrant "${QLOG}"   $QPID
wait_ready "http://localhost:8200/v1/models"  embed  "${EMB_LOG}" $EMB_PID
wait_ready "http://localhost:8201/v1/models"  gen    "${GEN_LOG}" $GEN_PID

# --- SAFETY: confirm the existing index is intact before generating ---
PTS=$(curl -s "http://localhost:6333/collections/${COLL}" | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["points_count"])')
echo "[smoke] collection ${COLL} points_count=${PTS}"
if [ "${PTS}" -lt 25000 ]; then
  echo "[smoke] ABORT: collection has ${PTS} points (<25000); index looks altered. Not running."
  exit 2
fi

# --- live engine wiring: embeddings -> :8200, rerank/generation -> :8201 ---
export CASTLERAG_CONFIG="${CONF}"
export OMNIEMBED_BASE_URL="http://localhost:8200/v1"   # live query embeddings
export VLLM_BASE_URL="http://localhost:8201/v1"        # rerank + generation
# NOTE: OMNIEMBED_QUERY_CACHE intentionally UNSET so fresh questions embed live.

echo "[smoke] === smoke-test --wandb (n=${N}) ==="
castlerag smoke-test "${QFILE}" --n "${N}" --wandb --config "${CONF}"

echo "[smoke] === predictions ==="
find "${SC}/castle_outputs/smoke_test" -name 'predictions*.json' -exec sh -c 'echo "file: $1"; cat "$1"' _ {} \; 2>/dev/null
echo
echo "[smoke] DONE $(date -Iseconds)"
