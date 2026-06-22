#!/bin/bash
# smoke_day1_roomfix.slurm — NON-DESTRUCTIVE 40-question day-1 smoke eval against
# the EXISTING castle_multimodal_v1 index, to validate the room-hard-filter fix
# (issue #50 / PR #53).  Boots OmniEmbed + Qwen3-VL + Qdrant on the persisted
# scratch storage and runs ONLY `castlerag smoke-test` (no preprocess / embed /
# index / create-collection).  Aborts if the collection has been emptied.
#
# The fix is query-time only (retrieval no longer hard-filters dense search by
# the room hint), so the existing index needs NO rebuild — only the working-tree
# code on the fix/room-hard-filter branch (editable install) must be checked out.
#
#   sbatch --account=gpuuva082 scripts/slurm/smoke_day1_roomfix.slurm

#SBATCH --job-name=castle-smoke-roomfix
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

CONF="configs/snellius_me.yaml"
SC="/scratch-shared/${USER}"
QFILE="data/smoke_day1_questions.json"
NQ=40
COLL="castle_multimodal_v1"
export HF_HOME="${SC}/hf_cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_LOGGING_LEVEL=WARNING

echo "[smoke] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1) start=$(date -Iseconds)"
echo "[smoke] branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) commit=$(git rev-parse --short HEAD 2>/dev/null)"

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

# --- OmniEmbed (GPU0:8200) — for query-cache misses on the 40-question set ---
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

wait_ready "http://localhost:6333/readyz" qdrant "${QLOG}" $QPID
wait_ready "http://localhost:8200/v1/models" embed "${EMB_LOG}" $EMB_PID
wait_ready "http://localhost:8201/v1/models" gen "${GEN_LOG}" $GEN_PID

# --- SAFETY: confirm the existing index is intact before generating ---
PTS=$(curl -s "http://localhost:6333/collections/${COLL}" | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["points_count"])' 2>/dev/null || echo "")
echo "[smoke] collection ${COLL} points_count=${PTS:-<none>}"
if ! [[ "${PTS}" =~ ^[0-9]+$ ]]; then
  echo "[smoke] ABORT: could not read points_count for ${COLL}; not running."; exit 2
fi
if [ "${PTS}" -lt 25000 ]; then
  echo "[smoke] ABORT: collection has ${PTS} points (<25000); index looks altered. Not running."; exit 2
fi

# --- pre-warm the query cache for THIS question set (OmniEmbed @ 8200) ---
# The eval embed client is pointed at VLLM_BASE_URL (the Qwen3-VL gen server on
# 8201), so a live query-embed on a cache MISS would POST /v1/embeddings to the
# gen server and 404. Precomputing every query variant against OmniEmbed (8200)
# gives 100% cache hits, matching the designed full_day1 flow. Written to a
# smoke-specific file so the existing query_cache.npz is untouched.
SMOKE_CACHE="${SC}/smoke_day1_query_cache.npz"
echo "[smoke] === precompute query cache (${NQ} questions) ==="
OMNIEMBED_URL="http://localhost:8200/v1" \
  python scripts/precompute_queries.py "${QFILE}" \
    --model Tevatron/OmniEmbed-v0.1-multivent \
    --out "${SMOKE_CACHE}"

# --- 40-question day-1 smoke eval (retrieve -> rerank -> generate) ---
export VLLM_BASE_URL="http://localhost:8201/v1"
export OMNIEMBED_QUERY_CACHE="${SMOKE_CACHE}"
echo "[smoke] === smoke-test (${NQ} questions, ${QFILE}) ==="
castlerag smoke-test "${QFILE}" --n "${NQ}" --config "${CONF}"

OUT="${SC}/castle_outputs/smoke_test"
echo "[smoke] === metrics.json ==="
cat "${OUT}/metrics.json" 2>/dev/null || find "${OUT}" -name 'metrics*.json' -exec cat {} \;
echo

# --- issue #50 summary: zero-evidence (is_supported=False) breakdown ---
echo "[smoke] === issue #50 summary (zero-evidence questions) ==="
PRED=$(find "${OUT}" -name 'predictions*.json' | head -1)
python3 - "$PRED" "${QFILE}" <<'PY'
import json, sys
try:
    preds = json.load(open(sys.argv[1]))
except Exception as e:
    print("could not read predictions:", e); sys.exit(0)
qtext = {}
try:
    qtext = {k: v.get("query", "") for k, v in json.load(open(sys.argv[2])).items()}
except Exception:
    pass
items = list(preds.values()) if isinstance(preds, dict) else preds
unsupported = [p for p in items if isinstance(p, dict) and p.get("is_supported") is False]
print(f"total predictions      : {len(items)}")
print(f"zero-evidence (is_supported=False): {len(unsupported)}  "
      f"(baseline #50 was 8/40)")
for p in unsupported:
    qid = p.get("question_id")
    print(f"  - {qid}  predicted={p.get('predicted_answer')}  "
          f"priors={p.get('support_priors')}")
    if qtext.get(qid):
        print(f"      Q: {qtext[qid]}")
PY

echo "[smoke] DONE $(date -Iseconds)"
