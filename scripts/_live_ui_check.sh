#!/bin/bash
# Bounded live-backend validation: boot Qwen3-VL (gen/rerank/suggestions) + Qdrant
# on the persisted day-1 index, then exercise the real RagEngine end to end
# (require-live build -> answer a cached question -> live per-camera suggestion +
# refined query). Embeddings come from the query cache, so OmniEmbed isn't needed.
set -euo pipefail
SC="/scratch-shared/${USER}"
export HF_HOME="${SC}/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_LOGGING_LEVEL=WARNING
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
source "${HOME}/castlerag_venv/bin/activate"
cd "${HOME}/CastleRAG"
mkdir -p logs
GEN_LOG="logs/live_vllm_$$.log"; QLOG="logs/live_qdrant_$$.log"

echo "[live] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1)"

vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8201 \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --trust-remote-code \
    > "${GEN_LOG}" 2>&1 &
GEN_PID=$!
export QDRANT__STORAGE__STORAGE_PATH="${SC}/qdrant_storage/storage"
"${HOME}/qdrant/qdrant" > "${QLOG}" 2>&1 &
QPID=$!
trap 'kill ${GEN_PID} ${QPID} 2>/dev/null' EXIT

wait_ready(){ for i in $(seq 1 "$2"); do curl -sf "$1" >/dev/null 2>&1 && return 0; kill -0 "$3" 2>/dev/null || return 1; sleep 5; done; return 1; }
wait_ready http://localhost:6333/readyz 24 "${QPID}" || { echo "[live] QDRANT FAILED"; tail -20 "${QLOG}"; exit 1; }
echo "[live] qdrant ready"
wait_ready http://localhost:8201/v1/models 120 "${GEN_PID}" || { echo "[live] VLLM FAILED"; tail -40 "${GEN_LOG}"; exit 1; }
echo "[live] vllm ready"

export VLLM_BASE_URL=http://localhost:8201/v1
export OMNIEMBED_QUERY_CACHE="${SC}/castle_derived/embeddings/query_cache.npz"
export CASTLERAG_CONFIG=configs/snellius_me.yaml

PYTHONPATH=src python - <<'PY'
import json, sys, traceback
out = {}
try:
    from castlerag.config import load_config
    from castlerag.ui.engine_factory import build_engine, engine_mode
    from castlerag.ui.youtube import YouTubeMirror
    cfg = load_config(override_path="configs/snellius_me.yaml")
    eng = build_engine(YouTubeMirror.from_csv(), cfg=cfg, require_live=True)
    out["engine"] = type(eng).__name__
    out["mode"] = engine_mode(eng)

    # (A) Review-suggestion path — chat model only, no retrieval needed.
    try:
        sc = "The footage confirms a participant discussing breakfast in the kitchen."
        out["live_suggestion"] = eng.suggest_justification(
            sc, "Bjorn", "confirmed", "transcript: let's make some porridge for breakfast",
            {"clock_label": "08:30", "place_label": "Kitchen", "match_score": 0.74},
        )[:240]
        review = {"Bjorn": {"state": "confirmed", "justification": "clear view"},
                  "Luca": {"state": "flagged", "justification": "occluded angle"},
                  "Klaus": {"state": "rejected", "justification": "no kitchen view"}}
        out["live_refined_query"] = eng.suggest_refined_query(sc, review)[:240]
    except Exception:
        out["suggestion_error"] = traceback.format_exc()[-700:]

    # (B) Full pipeline — real query + choices so the dense query hits the cache.
    try:
        q = "What breakfast food is discussed in the kitchen on day 1?"
        ans = {"a": "Pancakes", "b": "Sausages", "c": "Cereal", "d": "Porridge"}
        res = eng.answer(q, choices=ans)
        claim = res.claim.text if res.claim else ""
        out["answer_text"] = (res.answer_text or "")[:200]
        out["predicted_choice"] = res.predicted_choice
        out["support"] = res.claim.support.value if res.claim else None
        out["n_moments"] = len(res.moments)
        if res.moments:
            m = res.moments[0]
            out["moment0"] = {
                "clock": m.clock_label, "place": m.place_label,
                "cams": [(c.camera_id, round(c.match_score, 3), bool(c.evidence_text)) for c in m.cameras],
            }
    except Exception:
        out["answer_error"] = traceback.format_exc()[-900:]
except Exception:
    out["error"] = traceback.format_exc()[-1500:]
print("RESULT_JSON " + json.dumps(out, ensure_ascii=False))
# Exit non-zero if any stage captured an error, so the run can't go false-green.
if any(out.get(k) for k in ("error", "answer_error", "suggestion_error")):
    sys.exit(1)
PY
RC=$?
echo "[live] DONE (rc=${RC})"
exit "${RC}"
