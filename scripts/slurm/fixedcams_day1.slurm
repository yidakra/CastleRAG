#!/bin/bash
# fixedcams_day1.slurm — ADDITIVE day-1 ingest of the fixed room cameras
# (issue #50 Bug B) into the EXISTING castle_multimodal_v1 collection.
#
# Never recreates the collection and never re-captions ego clips:
#   * preprocess runs with `--camera <cam>` so base/caption/events only touch
#     the fixed cameras' chunks (ego clips.jsonl checksums are asserted
#     unchanged afterwards);
#   * embed caches are incremental per record id, so existing ego rows in
#     *_day1.npz are kept verbatim and only new fixed rows are embedded;
#   * `index --day 1` WITHOUT --create-collection upserts by deterministic
#     UUIDv5 id — ego points are overwritten with identical data, fixed points
#     are new.
#
# CAMS selects the cameras. Default "auto" = only fixed cameras with ZERO
# main_clip points in the collection today (see scripts/qdrant_camera_counts.py);
# if none are missing the job exits 0 without touching anything.
#
#   sbatch --account=gpuuva082 scripts/slurm/fixedcams_day1.slurm
#   sbatch --account=gpuuva082 --export=ALL,CAMS="Reading" scripts/slurm/fixedcams_day1.slurm
#   sbatch --account=gpuuva082 --export=ALL,CAMS="Kitchen Living1 Living2 Meeting Reading" \
#          scripts/slurm/fixedcams_day1.slurm
#
# Full runbook: docs/fixedcams_reingest.md

#SBATCH --job-name=castle-fixedcams-day1
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=360G
#SBATCH --gres=gpu:3
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail   # no -e: per-camera failures are handled explicitly

SC="/scratch-shared/${USER}"
CONF="${CONF:-configs/snellius_fixedcams.yaml}"
COLL="castle_multimodal_v1"
CAMS="${CAMS:-auto}"
SKIP_BASE="${SKIP_BASE:-0}"       # 1 = chunks/frames already exist for CAMS
SNAPSHOT="${SNAPSHOT:-1}"         # 1 = Qdrant snapshot before upserting
MIN_POINTS="${MIN_POINTS:-25000}" # the ego-only day-1 index had 25,920
CHUNKS="${SC}/castle_derived/chunks/day1"
EMB="${SC}/castle_derived/embeddings"
export HF_HOME="${SC}/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_LOGGING_LEVEL=WARNING
export QDRANT_CLIENT_TIMEOUT="${QDRANT_CLIENT_TIMEOUT:-300}"
J="${SLURM_JOB_ID}"

echo "[fixed] host=$(hostname) start=$(date -Iseconds) conf=${CONF} CAMS=${CAMS}"
echo "[fixed] branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) commit=$(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi -L 2>/dev/null | head -3

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load FFmpeg/7.0.2-GCCcore-13.3.0
source "${HOME}/castlerag_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
mkdir -p logs

wait_ready() { # url label logfile pid
  local pid="${4:-}"
  for i in $(seq 1 150); do
    curl -sf "$1" >/dev/null 2>&1 && { echo "[fixed] $2 ready ($((i*10))s)"; return 0; }
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[fixed] ERROR: $2 died"; tail -30 "$3" 2>/dev/null; return 1
    fi
    sleep 10
  done
  echo "[fixed] ERROR: $2 not ready"; tail -30 "$3" 2>/dev/null; return 1
}

# --- Qdrant on the EXISTING persisted storage (no create-collection, ever) ---
export QDRANT__STORAGE__STORAGE_PATH="${SC}/qdrant_storage/storage"
export QDRANT__STORAGE__SNAPSHOTS_PATH="${SC}/qdrant_snapshots"
mkdir -p "${QDRANT__STORAGE__SNAPSHOTS_PATH}"
QLOG="logs/fixed_qdrant_${J}.log"
"${HOME}/qdrant/qdrant" > "${QLOG}" 2>&1 &
QP=$!
PIDS_TO_KILL="${QP}"
trap 'kill ${PIDS_TO_KILL} 2>/dev/null' EXIT
wait_ready "http://localhost:6333/readyz" qdrant "${QLOG}" "${QP}" || exit 1

# --- Phase 0: state check -------------------------------------------------
PTS=$(curl -s "http://localhost:6333/collections/${COLL}" \
      | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["points_count"])' 2>/dev/null || echo "")
echo "[fixed] collection ${COLL} points_count=${PTS:-<none>}"
if ! [[ "${PTS}" =~ ^[0-9]+$ ]] || [ "${PTS}" -lt "${MIN_POINTS}" ]; then
  echo "[fixed] ABORT: ${COLL} missing or < ${MIN_POINTS} points. Restore the archive"
  echo "[fixed]        (~/castle_archives/bugb_day1_*/RESTORE.md) or rebuild ego first."
  exit 2
fi
python scripts/qdrant_camera_counts.py --collection "${COLL}" --day day1 | tee "logs/fixed_counts_before_${J}.txt"
if [ "${CAMS}" = "auto" ]; then
  CAMS="$(python scripts/qdrant_camera_counts.py --collection "${COLL}" --day day1 --missing-only)"
  echo "[fixed] auto-selected missing fixed cameras: '${CAMS}'"
  if [ -z "${CAMS// /}" ]; then
    echo "[fixed] all 5 fixed cameras already have main_clip points — nothing to ingest."
    exit 0
  fi
fi
read -r -a CAM_ARR <<< "${CAMS}"
CAM_FLAGS=(); for c in "${CAM_ARR[@]}"; do CAM_FLAGS+=(--camera "$c"); done
# Fail fast on a typo / wrong scope before any GPU time is spent.
castlerag preprocess --config "${CONF}" --day 1 "${CAM_FLAGS[@]}" --dry-run || exit 1

# Raw video present for every requested camera?
for c in "${CAM_ARR[@]}"; do
  n=$(ls "${SC}/castle2024/main/day1/${c}/video/"*.mp4 2>/dev/null | wc -l)
  echo "[fixed] raw video ${c}: ${n} hour files"
  [ "${n}" -gt 0 ] || { echo "[fixed] ABORT: no raw day-1 video for ${c}"; exit 2; }
done

# --- Phase 0b: rollback material -----------------------------------------
BK="${SC}/castle_derived/embeddings_backup_pre_fixedcams_${J}"
mkdir -p "${BK}"
cp -p "${EMB}"/*_day1.npz "${EMB}"/manifest_day1.json "${EMB}"/transcripts.pkl "${BK}/" 2>/dev/null
echo "[fixed] embedding caches backed up -> ${BK}"
if [ "${SNAPSHOT}" = "1" ]; then
  curl -sf -X POST "http://localhost:6333/collections/${COLL}/snapshots?wait=true" \
    | python3 -c 'import sys,json;print("[fixed] qdrant snapshot:",json.load(sys.stdin)["result"]["name"])' \
    || echo "[fixed] WARNING: snapshot failed (continuing; caches are backed up)"
fi
EGO_SUM="$(pwd)/logs/fixed_ego_sha_${J}.txt"
is_target() { local x; for x in "${CAM_ARR[@]}"; do [ "$1" = "$x" ] && return 0; done; return 1; }
( cd "${CHUNKS}" && for d in */; do
    d="${d%/}"; is_target "$d" || find "$d" -name '*.jsonl' -print0
  done | sort -z | xargs -0 -r sha1sum ) > "${EGO_SUM}"
echo "[fixed] checksummed $(wc -l < "${EGO_SUM}") non-target chunk files"

# --- GPU servers: 2x Qwen3-VL (caption/events) + OmniEmbed (embed) --------
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8201 \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl \
    --gpu-memory-utilization 0.85 --max-model-len 4096 --trust-remote-code \
    > "logs/fixed_gen0_${J}.log" 2>&1 &
G0=$!
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8202 \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl \
    --gpu-memory-utilization 0.85 --max-model-len 4096 --trust-remote-code \
    > "logs/fixed_gen1_${J}.log" 2>&1 &
G1=$!
CUDA_VISIBLE_DEVICES=2 python scripts/omniembed_server.py --port 8200 \
    > "logs/fixed_omni_${J}.log" 2>&1 &
EMBP=$!
PIDS_TO_KILL="${QP} ${G0} ${G1} ${EMBP}"
wait_ready "http://localhost:8201/v1/models" gen0 "logs/fixed_gen0_${J}.log" "${G0}" || exit 1
wait_ready "http://localhost:8202/v1/models" gen1 "logs/fixed_gen1_${J}.log" "${G1}" || exit 1

# --- Phase 1: base + caption + events, (camera x hour-group) workers --------
# Safe in parallel now that caption/events are camera- AND hour-scoped: each
# worker only rewrites its own <cam>/<HH>/clips.jsonl + events.jsonl. Captioning
# is sequential per worker, so each camera's hours are split into SPLIT groups
# to keep ~TARGET_WORKERS clients batching on the 2 gen servers (full_day1 ran
# 10). One camera alone (e.g. Reading) -> 10 workers, not 1.
HOURS="${HOURS:-8 9 10 11 12 13 14 15 16 17 18 19 20}"
read -r -a HOUR_ARR <<< "${HOURS}"
TARGET_WORKERS="${TARGET_WORKERS:-10}"
if [ "${SPLIT:-auto}" = "auto" ]; then
  SPLIT=$(( (TARGET_WORKERS + ${#CAM_ARR[@]} - 1) / ${#CAM_ARR[@]} ))
fi
[ "${SPLIT}" -gt "${#HOUR_ARR[@]}" ] && SPLIT="${#HOUR_ARR[@]}"
echo "[fixed] === Phase 1: preprocess ${CAM_ARR[*]} split=${SPLIT} $(date -Iseconds) ==="
BASE_FLAG=(); [ "${SKIP_BASE}" = "1" ] && BASE_FLAG=(--skip-base)
declare -a P PC
w=0
for cam in "${CAM_ARR[@]}"; do
  for g in $(seq 0 $((SPLIT - 1))); do
    HFLAGS=(); for i in "${!HOUR_ARR[@]}"; do
      [ $(( i % SPLIT )) -eq "$g" ] && HFLAGS+=(--hour "${HOUR_ARR[$i]}")
    done
    [ "${#HFLAGS[@]}" -eq 0 ] && continue
    port=$((8201 + w % 2)); tag="${cam}_g${g}"; w=$((w + 1))
    ( T0=$(date +%s)
      VLLM_BASE_URL="http://localhost:${port}/v1" \
        castlerag preprocess --config "${CONF}" --day 1 --camera "${cam}" "${HFLAGS[@]}" \
          "${BASE_FLAG[@]}" --caption --events > "logs/fixed_cam_${tag}_${J}.log" 2>&1
      rc=$?; echo "[fixed] ${tag} (port ${port}) rc=${rc} in $(( $(date +%s)-T0 ))s"; exit $rc
    ) & P+=($!); PC+=("${tag}")
  done
done
echo "[fixed] launched ${w} preprocess workers"
FAIL=0
for i in "${!P[@]}"; do wait "${P[$i]}" || { echo "[fixed] CAMERA ${PC[$i]} FAILED"; FAIL=1; }; done
echo "[fixed] === Phase 1 done $(date -Iseconds) FAIL=${FAIL} ==="
[ "${FAIL}" -ne 0 ] && { echo "[fixed] ABORT before embed/index (see logs/fixed_cam_*_${J}.log)"; exit 1; }

# --- Phase 1b: validate ----------------------------------------------------
python3 - "${CHUNKS}" "${CAM_ARR[@]}" <<'PY' || { echo "[fixed] VALIDATION FAILED"; exit 1; }
import glob, json, sys
root, cams = sys.argv[1], sys.argv[2:]
ok = True
for cam in cams:
    tot = cap = bad = ev = 0; rooms = set(); types = set()
    for f in glob.glob(f"{root}/{cam}/*/clips.jsonl"):
        for line in open(f):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                bad += 1; continue
            tot += 1; cap += bool(d.get("clip_caption"))
            rooms.add(d.get("room")); types.add(d.get("camera_type"))
    for f in glob.glob(f"{root}/{cam}/*/events.jsonl"):
        ev += sum(1 for line in open(f) if line.strip())
    pct = 100 * cap / max(tot, 1)
    print(f"  {cam}: clips={tot} captioned={cap} ({pct:.0f}%) events={ev} bad={bad} "
          f"camera_type={types} room={rooms}")
    if bad or tot == 0 or pct < 98 or types != {"fixed"} or rooms != {cam}:
        ok = False
sys.exit(0 if ok else 1)
PY
( cd "${CHUNKS}" && sha1sum --quiet -c "${EGO_SUM}" ) \
  || { echo "[fixed] ABORT: non-target (ego) chunk files changed!"; exit 1; }
echo "[fixed] ego chunk files unchanged"

# Free the caption GPUs before embedding (OmniEmbed stays on GPU2).
kill "${G0}" "${G1}" 2>/dev/null; PIDS_TO_KILL="${QP} ${EMBP}"

# --- Phase 2: embed per modality (incremental: only new record ids) --------
wait_ready "http://localhost:8200/v1/models" embed "logs/fixed_omni_${J}.log" "${EMBP}" || exit 1
export VLLM_BASE_URL="http://localhost:8200/v1"
for m in transcript event_summary video; do
  echo "[fixed] === embed --modality ${m} $(date -Iseconds) ==="
  castlerag embed --config "${CONF}" --day 1 --modality "${m}" || { echo "[fixed] embed ${m} FAILED"; exit 1; }
done

# --- Phase 3: additive index (NO --create-collection) ----------------------
echo "[fixed] === index --day 1 (additive) $(date -Iseconds) ==="
castlerag index --config "${CONF}" --day 1 || { echo "[fixed] index FAILED"; exit 1; }

# --- Phase 4: verify -------------------------------------------------------
python scripts/qdrant_camera_counts.py --collection "${COLL}" --day day1 \
  | tee "logs/fixed_counts_after_${J}.txt"
python scripts/qdrant_camera_counts.py --collection "${COLL}" --day day1 \
  --cameras "${CAM_ARR[@]}" --missing-only --fail-if-missing \
  || { echo "[fixed] VERIFY FAILED: some requested cameras still have 0 main_clip points"; exit 1; }
echo "[fixed] DONE $(date -Iseconds). Next: bash scripts/archive_bugb.sh, then the 40-q eval"
echo "[fixed]   sbatch --account=gpuuva082 --export=ALL,CONF=${CONF} scripts/slurm/smoke_day1_roomfix.slurm"
