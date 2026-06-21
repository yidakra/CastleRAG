#!/bin/bash
# ui_live.slurm — serve the CastleRAG dashboard wired to the REAL RAG pipeline.
#
# Boots OmniEmbed (embeddings, :8200), Qwen3-VL (rerank+generation, :8201) and
# Qdrant against the EXISTING persisted index, then launches `castlerag ui` with
# the live RagEngine.  Non-destructive: it never preprocesses / embeds / indexes
# / creates collections, and aborts if the collection looks emptied.
#
# Reach it from your laptop with an SSH tunnel to the compute node (the script
# prints the exact command once the node is known):
#
#   sbatch --account=gpuuva082 scripts/slurm/ui_live.slurm
#   # then, from your laptop (NODE printed in logs/castle-ui-live_<jobid>.out):
#   ssh -L 8050:<NODE>:8050 scur0225@snellius.surf.nl
#   # open http://localhost:8050  (top-bar chip should read "live RAG")

#SBATCH --job-name=castle-ui-live
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

CONF="configs/snellius_me.yaml"
SC="/scratch-shared/${USER}"
COLL="castle_multimodal_v1"
UI_PORT="${UI_PORT:-8050}"
export HF_HOME="${SC}/hf_cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_LOGGING_LEVEL=WARNING

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load FFmpeg/7.0.2-GCCcore-13.3.0
source "${HOME}/castlerag_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p logs

echo "[ui] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1) start=$(date -Iseconds)"

wait_ready() { # url label logfile pid
  local pid="${4:-}"
  for i in $(seq 1 120); do
    curl -sf "$1" >/dev/null 2>&1 && { echo "[ui] $2 ready ($((i*10))s)"; return 0; }
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[ui] ERROR: $2 server died"; tail -30 "$3" 2>/dev/null; return 1
    fi
    sleep 10
  done
  echo "[ui] ERROR: $2 not ready after 1200s"; tail -30 "$3" 2>/dev/null; return 1
}

# --- OmniEmbed embeddings server (GPU0, :8200) — embeds the live UI queries ---
EMB_LOG="logs/ui_omniembed_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0 python scripts/omniembed_server.py --port 8200 > "${EMB_LOG}" 2>&1 &
EMB_PID=$!
# --- Qwen3-VL rerank + generation server (GPU1, :8201) ---
GEN_LOG="logs/ui_vllm_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8201 --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --trust-remote-code \
    > "${GEN_LOG}" 2>&1 &
GEN_PID=$!
# --- Qdrant against the EXISTING persisted storage (NO create-collection) ---
export QDRANT__STORAGE__STORAGE_PATH="${SC}/qdrant_storage/storage"
QLOG="logs/ui_qdrant_${SLURM_JOB_ID}.log"
"${HOME}/qdrant/qdrant" > "${QLOG}" 2>&1 &
QPID=$!
trap 'kill $EMB_PID $GEN_PID $QPID 2>/dev/null' EXIT

wait_ready "http://localhost:6333/readyz"        qdrant "${QLOG}"   $QPID
wait_ready "http://localhost:8200/v1/models"      embed  "${EMB_LOG}" $EMB_PID
wait_ready "http://localhost:8201/v1/models"      gen    "${GEN_LOG}" $GEN_PID

# --- SAFETY: confirm the existing index is intact before serving ---
PTS=$(curl -s "http://localhost:6333/collections/${COLL}" | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["points_count"])' 2>/dev/null)
echo "[ui] collection ${COLL} points_count=${PTS:-<none>}"
if ! [[ "${PTS}" =~ ^[0-9]+$ ]]; then
  echo "[ui] ABORT: could not read points_count for ${COLL} (got '${PTS}'); is Qdrant up and the collection present? Not serving."
  exit 2
fi
if [ "${PTS}" -lt 25000 ]; then
  echo "[ui] ABORT: collection has ${PTS} points (<25000); index looks altered. Not serving."
  exit 2
fi

# --- live engine wiring: embeddings -> :8200, rerank/generation -> :8201 ---
export CASTLERAG_CONFIG="${CONF}"                      # host paths (scratch cache_dir, collection)
export OMNIEMBED_BASE_URL="http://localhost:8200/v1"   # OmniEmbed query embeddings
export VLLM_BASE_URL="http://localhost:8201/v1"        # Qwen3-VL rerank + generation
# NOTE: OMNIEMBED_QUERY_CACHE intentionally UNSET so arbitrary UI queries embed live.

NODE="$(hostname)"
cat <<EOF

[ui] ============================================================
[ui]  CastleRAG live UI is starting on ${NODE}:${UI_PORT}
[ui]  From your LAPTOP, open a tunnel, then browse http://localhost:${UI_PORT}
[ui]
[ui]    ssh -L ${UI_PORT}:${NODE}:${UI_PORT} ${USER}@snellius.surf.nl
[ui]
[ui]  (alt, direct to node):  ssh -J ${USER}@snellius.surf.nl ${USER}@${NODE} -L ${UI_PORT}:localhost:${UI_PORT}
[ui]  The top-bar chip should read "live RAG". scancel ${SLURM_JOB_ID} when done.
[ui] ============================================================

EOF

# Bind 0.0.0.0 so the login node can forward the tunnel to this compute node.
# --require-live: fail loudly with the exact missing dependency instead of
# silently serving the offline demo (the whole point of this job).
exec castlerag ui --host 0.0.0.0 --port "${UI_PORT}" \
    --require-live --config "${CONF}"
