#!/usr/bin/env bash
# vps_ui.sh — single-host launcher for the CastleRAG live RAG demo on a VPS.
#
# Mirrors scripts/slurm/ui_live.slurm but for a 2x NVIDIA A2 (16 GB) box:
#   - Qdrant via docker (v1.17.1, reads /home/ubuntu/storage)
#   - OmniEmbed 8-bit on GPU 0 :8200  (scripts/omniembed_server.py)
#   - Qwen3-VL AWQ 4-bit on GPU 1 :8201 (vllm serve)
#   - castlerag ui on :8050 behind HTTP basic auth (CASTLERAG_UI_BASIC_AUTH)
#   - cloudflared quick tunnel -> :8050
#
# Required env (DO NOT commit values):
#   HF_TOKEN                          gated Qwen2.5-Omni-7B base
#   CASTLERAG_UI_BASIC_AUTH=user:pw   gates the dashboard (mandatory for public)
#
# Non-destructive: never recreates the Qdrant collection. Aborts if points<25000.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

QDRANT_STORAGE="${QDRANT_STORAGE:-/home/ubuntu/storage}"
QDRANT_SNAPSHOTS="${QDRANT_SNAPSHOTS:-/home/ubuntu/qdrant_snapshots}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.17.1}"
COLL="${COLL:-castle_multimodal_v1}"
MIN_POINTS="${MIN_POINTS:-25000}"

EMB_PORT="${EMB_PORT:-8200}"
GEN_PORT="${GEN_PORT:-8201}"
UI_PORT="${UI_PORT:-8050}"

CONF="${CONF:-configs/vps.yaml}"
GEN_MODEL_REPO="${GEN_MODEL_REPO:-cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit}"
GEN_SERVED_NAME="${GEN_SERVED_NAME:-Qwen/Qwen3-VL-8B-Instruct}"

VENV="${REPO_DIR}/.venv"
PY="${VENV}/bin/python"

mkdir -p "${QDRANT_SNAPSHOTS}"

wait_ready() { # url label logfile [pid]
    local url="$1" label="$2" logfile="$3" pid="${4:-}"
    for _ in $(seq 1 240); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            echo "[vps] ${label} ready"
            return 0
        fi
        if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "[vps] ERROR: ${label} process died"
            tail -40 "${logfile}" 2>/dev/null || true
            return 1
        fi
        sleep 5
    done
    echo "[vps] ERROR: ${label} not ready after 1200s"
    tail -40 "${logfile}" 2>/dev/null || true
    return 1
}

PIDS=()
on_exit() {
    set +e
    for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null; done
    sudo docker rm -f castle_qdrant 2>/dev/null
}
trap on_exit EXIT

# 1. Qdrant container ---------------------------------------------------------
echo "[vps] starting Qdrant (${QDRANT_IMAGE}) bind-mount ${QDRANT_STORAGE}"
sudo docker rm -f castle_qdrant 2>/dev/null
sudo docker run -d --name castle_qdrant --restart unless-stopped \
    -p 6333:6333 -p 6334:6334 \
    -v "${QDRANT_STORAGE}:/qdrant/storage" \
    -v "${QDRANT_SNAPSHOTS}:/qdrant/snapshots" \
    --user "$(id -u):$(id -g)" \
    "${QDRANT_IMAGE}" >/dev/null

wait_ready "http://localhost:6333/readyz" qdrant "/dev/null"

PTS=$(curl -s "http://localhost:6333/collections/${COLL}" \
    | "${PY}" -c 'import sys,json; print(json.load(sys.stdin)["result"]["points_count"])' 2>/dev/null)
echo "[vps] collection ${COLL} points_count=${PTS:-<none>}"
if ! [[ "${PTS}" =~ ^[0-9]+$ ]] || [ "${PTS}" -lt "${MIN_POINTS}" ]; then
    echo "[vps] ABORT: collection missing or has fewer than ${MIN_POINTS} points; not serving."
    exit 2
fi

# 2. OmniEmbed 8-bit on GPU 0 -------------------------------------------------
EMB_LOG="${LOG_DIR}/omniembed.log"
echo "[vps] starting OmniEmbed (8-bit) on GPU 0 :${EMB_PORT} (log: ${EMB_LOG})"
CUDA_VISIBLE_DEVICES=0 OMNIEMBED_LOAD_IN_8BIT=1 \
    nohup "${PY}" scripts/omniembed_server.py --port "${EMB_PORT}" \
    > "${EMB_LOG}" 2>&1 &
EMB_PID=$!
PIDS+=("${EMB_PID}")

# 3. Qwen3-VL AWQ on GPU 1 ----------------------------------------------------
GEN_LOG="${LOG_DIR}/vllm_gen.log"
echo "[vps] starting Qwen3-VL AWQ on GPU 1 :${GEN_PORT} (log: ${GEN_LOG})"
# VLLM_USE_FLASHINFER_SAMPLER=0: bypass the JIT-compiled flashinfer sampler kernel
# that requires a CUDA toolkit (nvcc) at runtime. The native PyTorch sampler is
# functionally equivalent for this demo and adds no measurable latency at our scale.
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    nohup "${VENV}/bin/vllm" serve "${GEN_MODEL_REPO}" \
    --port "${GEN_PORT}" \
    --served-model-name "${GEN_SERVED_NAME}" qwen3vl \
    --quantization compressed-tensors \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enforce-eager \
    --trust-remote-code \
    > "${GEN_LOG}" 2>&1 &
GEN_PID=$!
PIDS+=("${GEN_PID}")

wait_ready "http://localhost:${EMB_PORT}/v1/models" omniembed "${EMB_LOG}" "${EMB_PID}"
wait_ready "http://localhost:${GEN_PORT}/v1/models" vllm-gen "${GEN_LOG}" "${GEN_PID}"

# 4. Dashboard ----------------------------------------------------------------
if [ -z "${CASTLERAG_UI_BASIC_AUTH:-}" ]; then
    echo "[vps] ABORT: CASTLERAG_UI_BASIC_AUTH must be set (user:pw) before public exposure."
    exit 2
fi

UI_LOG="${LOG_DIR}/ui.log"
echo "[vps] launching castlerag ui on :${UI_PORT} (log: ${UI_LOG})"
# CASTLERAG_UI_YOUTUBE_EMBED_HOST: swap the YouTube embed host from the privacy
# `youtube-nocookie.com` default to plain `youtube.com`.  Tunneled demos
# (Cloudflare quick-tunnel, ngrok, etc.) often hit YouTube's "Sign in to
# confirm you're not a bot" gate on nocookie embeds because that host is
# cookie-blind by design — even a signed-in visitor looks anonymous to it.
# youtube.com gives the iframe a chance to carry the YouTube cookie when the
# browser allows third-party cookies for youtube.com.
CASTLERAG_CONFIG="${CONF}" \
OMNIEMBED_BASE_URL="http://localhost:${EMB_PORT}/v1" \
VLLM_BASE_URL="http://localhost:${GEN_PORT}/v1" \
CASTLERAG_UI_BASIC_AUTH="${CASTLERAG_UI_BASIC_AUTH}" \
CASTLERAG_UI_YOUTUBE_EMBED_HOST="${CASTLERAG_UI_YOUTUBE_EMBED_HOST:-https://www.youtube.com}" \
    nohup "${VENV}/bin/castlerag" ui \
    --host 0.0.0.0 --port "${UI_PORT}" --require-live --config "${CONF}" \
    > "${UI_LOG}" 2>&1 &
UI_PID=$!
PIDS+=("${UI_PID}")

# Readiness check sends the basic-auth credentials (every route is gated).
for _ in $(seq 1 90); do
    if curl -sf -u "${CASTLERAG_UI_BASIC_AUTH}" \
        "http://localhost:${UI_PORT}/" >/dev/null 2>&1; then
        echo "[vps] dashboard ready"
        break
    fi
    kill -0 "${UI_PID}" 2>/dev/null || { echo "[vps] ERROR: UI process died"; tail -40 "${UI_LOG}"; exit 1; }
    sleep 2
done

# 5. cloudflared quick tunnel -------------------------------------------------
CF_LOG="${LOG_DIR}/cloudflared.log"
CFBIN="${CLOUDFLARED:-${HOME}/.local/bin/cloudflared}"
echo "[vps] starting cloudflared quick tunnel -> :${UI_PORT} (log: ${CF_LOG})"
nohup "${CFBIN}" tunnel --no-autoupdate --url "http://localhost:${UI_PORT}" \
    > "${CF_LOG}" 2>&1 &
CF_PID=$!
PIDS+=("${CF_PID}")

# Scrape the public URL once it appears (cloudflared logs it).
PUBLIC_URL=""
for _ in $(seq 1 60); do
    PUBLIC_URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "${CF_LOG}" 2>/dev/null | head -1)
    [ -n "${PUBLIC_URL}" ] && break
    kill -0 "${CF_PID}" 2>/dev/null || { echo "[vps] ERROR: cloudflared died"; tail -40 "${CF_LOG}"; exit 1; }
    sleep 2
done

cat <<EOF

[vps] ============================================================
[vps]  CastleRAG live demo is up.
[vps]    Public URL:  ${PUBLIC_URL:-<not yet captured — see ${CF_LOG}>}
[vps]    Basic auth:  ${CASTLERAG_UI_BASIC_AUTH%%:*} : (env CASTLERAG_UI_BASIC_AUTH)
[vps]    Status chip: should read "live RAG"
[vps]
[vps]  Foreground logs:  tail -F ${LOG_DIR}/{omniembed,vllm_gen,ui,cloudflared}.log
[vps]  Stop:             pkill -P \$\$  (or Ctrl-C if interactive)
[vps] ============================================================

EOF

# Wait on the UI; if it exits, the trap kills the rest.
wait "${UI_PID}"
