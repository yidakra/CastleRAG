#!/bin/bash
# ingest_day.slurm — ADDITIVE ingest of one CASTLE day (DAY=1..4) into the
# EXISTING castle_multimodal_v1 collection. Generalises the day-1 fixed-camera
# ingest of issue #50 Bug B / PR #61 (scripts/slurm/fixedcams_day1.slurm is now
# a thin DAY=1 wrapper around this file).
#
# Never recreates the collection and never touches cameras it was not asked for:
#   * preprocess runs with `--camera <cam>` so base/caption/events only touch
#     the selected cameras' chunks (every other camera's clips.jsonl checksums
#     for that day are asserted unchanged afterwards);
#   * embed caches are incremental per record id, so existing rows in
#     *_day${DAY}.npz are kept verbatim and only new rows are embedded;
#   * `index --day ${DAY}` WITHOUT --create-collection upserts by deterministic
#     UUIDv5 id — existing points are overwritten with identical data, new
#     cameras' points are added. Other days' points are not touched.
#
# DAY picks the day (default 1). Paths, log names, every `--day` flag, the
# Qdrant count checks (`--day day${DAY}`) and the HOURS default follow it.
# Pass it with --export, and name the job after the day so the log is findable:
#
#   sbatch --account=gisr109364 --job-name=castle-ingest-day2 \
#          --export=ALL,DAY=2 scripts/slurm/ingest_day.slurm
#   sbatch --account=gisr109364 --job-name=castle-ingest-day2 \
#          --export=ALL,DAY=2,CAMS="Kitchen Living1 Living2 Meeting Reading" \
#          scripts/slurm/ingest_day.slurm
#
# The account is per user (`accinfo` shows yours).
#
# CAMS selects the cameras. Default "auto" = every in-scope camera (from CONF)
# that has raw day-${DAY} video AND has ZERO main_clip points for that day in
# the collection (scripts/qdrant_camera_counts.py). On day 1 that is exactly
# the missing fixed cameras; on a new day it is every camera. If none are
# missing the job exits 0 without touching anything.
#
# The collection must already hold >= MIN_POINTS points (default 25000, i.e. a
# complete day 1). On an empty /scratch-shared the FIRST job passes BOOTSTRAP=1
# (refused if the collection has any points) and the index step creates the
# collection; the next jobs of that day pass MIN_POINTS=1.
#
# A whole day (~140-180 camera-hours) does not fit in one 20 h job. Split it
# into camera groups chained with --dependency=afterok (docs/ingest_all_days.md).
#
# Runbooks: docs/ingest_all_days.md (days 1-4), docs/fixedcams_reingest.md (day-1 fixed cams)

#SBATCH --job-name=castle-ingest-day
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

DAY="${DAY:-1}"
case "${DAY}" in
  1|2|3|4) ;;
  *) echo "[ingest] ABORT: DAY='${DAY}' must be 1, 2, 3 or 4"; exit 1;;
esac
T="[ingest d${DAY}]"
SC="/scratch-shared/${USER}"
CONF="${CONF:-configs/snellius_fixedcams.yaml}"
COLL="castle_multimodal_v1"
CAMS="${CAMS:-auto}"
SKIP_BASE="${SKIP_BASE:-0}"       # 1 = chunks/frames already exist for CAMS
CAPTION="${CAPTION:-1}"           # 0 = with SKIP_BASE=1: rebuild events only (captions kept),
                                  #     e.g. after scripts/reflag_placeholders.py
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}" # 1 = Phase 1 (base+caption+events) finished for CAMS in an
                                  #     earlier job: validate, embed and index only
SNAPSHOT="${SNAPSHOT:-1}"         # 1 = Qdrant snapshot before upserting
MIN_POINTS="${MIN_POINTS:-25000}" # whole collection; the ego-only day-1 index had 25,920
BOOTSTRAP="${BOOTSTRAP:-0}"       # 1 = empty scratch: no collection yet (first job only)
export HF_HOME="${SC}/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_LOGGING_LEVEL=WARNING
export QDRANT_CLIENT_TIMEOUT="${QDRANT_CLIENT_TIMEOUT:-300}"
J="${SLURM_JOB_ID}"
L="logs/ingest_day${DAY}"         # per-day log prefix: ${L}_<what>_${J}.log

echo "${T} host=$(hostname) start=$(date -Iseconds) conf=${CONF} CAMS=${CAMS}"
echo "${T} branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) commit=$(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi -L 2>/dev/null | head -3

module purge; module load 2024; module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 FFmpeg/7.0.2-GCCcore-13.3.0
source "${HOME}/castlerag_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
mkdir -p logs
# Every derived path comes from CONF (scratch for snellius_fixedcams.yaml, the
# project space for snellius_project.yaml), so a job can't mix the two.
PATHS="$(castlerag paths --config "${CONF}")" || { echo "${T} ABORT: castlerag paths failed for ${CONF}"; exit 1; }
eval "${PATHS}"
RAW="${DATA_ROOT}/main/day${DAY}"
CHUNKS="${CHUNKS_DIR}/day${DAY}"
EMB="${EMB_DIR}"
QSTORE="${QDRANT_STORAGE:-${SC}/qdrant_storage}"
echo "${T} paths: raw=${RAW} chunks=${CHUNKS} emb=${EMB} qdrant=${QSTORE}"

wait_ready() { # url label logfile pid
  local pid="${4:-}"
  for i in $(seq 1 150); do
    curl -sf "$1" >/dev/null 2>&1 && { echo "${T} $2 ready ($((i*10))s)"; return 0; }
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "${T} ERROR: $2 died"; tail -30 "$3" 2>/dev/null; return 1
    fi
    sleep 10
  done
  echo "${T} ERROR: $2 not ready"; tail -30 "$3" 2>/dev/null; return 1
}
has_video() { compgen -G "${RAW}/$1/video/*.mp4" >/dev/null; }

# Camera roster from CONF (scope-aware) + which of those are fixed room cams.
if ! ROSTER="$(python - "${CONF}" <<'PY'
import sys
from castlerag.config import load_config
from castlerag.dataset.layout import scoped_cameras
cfg = load_config(override_path=sys.argv[1])
d = cfg.dataset
print(" ".join(scoped_cameras(d.ego_cameras, d.exo_cameras, d.camera_scope)))
print(" ".join(d.exo_cameras))
PY
)"; then
  echo "${T} ABORT: could not read the camera roster from ${CONF}"; exit 1
fi
SCOPE_CAMS="$(sed -n 1p <<< "${ROSTER}")"
FIXED_CAMS="$(sed -n 2p <<< "${ROSTER}")"
echo "${T} cameras in scope: ${SCOPE_CAMS}"

# Hours per day, from src/castlerag/ui/youtube_mirror.csv: days 1-3 run 08-20,
# day 4 ends at 18. Hours without a video file are skipped by preprocess anyway.
if [ "${DAY}" = "4" ]; then
  DEFAULT_HOURS="8 9 10 11 12 13 14 15 16 17 18"
else
  DEFAULT_HOURS="8 9 10 11 12 13 14 15 16 17 18 19 20"
fi
HOURS="${HOURS:-${DEFAULT_HOURS}}"

# --- Qdrant on the EXISTING persisted storage (no create-collection, ever) ---
export QDRANT__STORAGE__STORAGE_PATH="${QSTORE}/storage"
QSNAP="$(dirname "${QSTORE}")/qdrant_snapshots"
export QDRANT__STORAGE__SNAPSHOTS_PATH="${QSNAP}"
mkdir -p "${QDRANT__STORAGE__SNAPSHOTS_PATH}"
QLOG="${L}_qdrant_${J}.log"
"${HOME}/qdrant/qdrant" > "${QLOG}" 2>&1 &
QP=$!
PIDS_TO_KILL="${QP}"
trap 'kill ${PIDS_TO_KILL} 2>/dev/null' EXIT
wait_ready "http://localhost:6333/readyz" qdrant "${QLOG}" "${QP}" || exit 1

# --- Phase 0: state check -------------------------------------------------
HTTP=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:6333/collections/${COLL}")
COLL_EXISTS=0; PTS=""
if [ "${HTTP}" = "200" ]; then
  COLL_EXISTS=1
  PTS=$(curl -s "http://localhost:6333/collections/${COLL}" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["points_count"])' 2>/dev/null || echo "")
elif [ "${HTTP}" != "404" ]; then
  echo "${T} ABORT: Qdrant answered HTTP ${HTTP:-<none>} for ${COLL}"; exit 2
fi
echo "${T} collection ${COLL} exists=${COLL_EXISTS} points_count=${PTS:-<none>}"
if [ "${BOOTSTRAP}" = "1" ]; then
  # First ingest into an empty scratch: `index --day N` creates the collection
  # (it is only ever created when missing, never recreated). Refuse if there
  # is already data, so BOOTSTRAP=1 cannot hide a wrong storage path later on.
  if [ "${COLL_EXISTS}" = "1" ] && { ! [[ "${PTS}" =~ ^[0-9]+$ ]] || [ "${PTS}" -gt 0 ]; }; then
    echo "${T} ABORT: BOOTSTRAP=1 but ${COLL} already has ${PTS:-?} points; drop BOOTSTRAP"; exit 2
  fi
  echo "${T} BOOTSTRAP: empty start, the index step will create ${COLL}"
  PTS=0
elif ! [[ "${PTS}" =~ ^[0-9]+$ ]] || [ "${PTS}" -lt "${MIN_POINTS}" ]; then
  echo "${T} ABORT: ${COLL} missing or < ${MIN_POINTS} points. Restore the archive"
  echo "${T}        (~/castle_archives/*/RESTORE.md), or on an empty scratch start"
  echo "${T}        with BOOTSTRAP=1 (and MIN_POINTS=1 for the next jobs of that day)."
  exit 2
fi
DAY_PTS=0
if [ "${COLL_EXISTS}" = "1" ]; then
  DAY_PTS=$(curl -s -X POST "http://localhost:6333/collections/${COLL}/points/count" \
              -H 'Content-Type: application/json' \
              -d "{\"exact\":true,\"filter\":{\"must\":[{\"key\":\"day\",\"match\":{\"value\":\"day${DAY}\"}}]}}" \
            | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["count"])' 2>/dev/null || echo "")
  if ! [[ "${DAY_PTS}" =~ ^[0-9]+$ ]]; then
    echo "${T} ABORT: could not count day${DAY} points in ${COLL}"; exit 2
  fi
  python scripts/qdrant_camera_counts.py --collection "${COLL}" --day "day${DAY}" \
    | tee "${L}_counts_before_${J}.txt"
fi
echo "${T} day${DAY} points already in ${COLL}: ${DAY_PTS}"
if [ "${CAMS}" = "auto" ]; then
  CAND=(); NOVID=()
  for c in ${SCOPE_CAMS}; do
    if has_video "$c"; then CAND+=("$c"); else NOVID+=("$c"); fi
  done
  [ "${#NOVID[@]}" -gt 0 ] && echo "${T} no raw day-${DAY} video, not auto candidates: ${NOVID[*]}"
  if [ "${#CAND[@]}" -eq 0 ]; then
    echo "${T} ABORT: no raw day-${DAY} video for any in-scope camera under ${RAW}"; exit 2
  fi
  if [ "${COLL_EXISTS}" = "0" ]; then
    CAMS="${CAND[*]}"   # nothing indexed yet: every camera with video is missing
  # Check the count's exit status explicitly: with no `set -e`, a crashed
  # count would print nothing and be misread as "nothing to ingest" (exit 0).
  elif ! CAMS="$(python scripts/qdrant_camera_counts.py --collection "${COLL}" --day "day${DAY}" \
                   --cameras "${CAND[@]}" --missing-only)"; then
    echo "${T} ABORT: camera count failed; cannot auto-select missing cameras."
    exit 2
  fi
  if [ "${COLL_EXISTS}" = "1" ] && [ "${#NOVID[@]}" -gt 0 ]; then
    if GONE="$(python scripts/qdrant_camera_counts.py --collection "${COLL}" --day "day${DAY}" \
                 --cameras "${NOVID[@]}" --missing-only)" && [ -n "${GONE// /}" ]; then
      echo "${T} WARNING: ${GONE} have no day${DAY} points AND no raw video (purged? not in this day?)"
    fi
  fi
  echo "${T} auto-selected cameras with 0 main_clip points: '${CAMS}'"
  if [ -z "${CAMS// /}" ]; then
    echo "${T} every camera with raw day-${DAY} video already has main_clip points — nothing to ingest."
    exit 0
  fi
fi
read -r -a CAM_ARR <<< "${CAMS}"
# A manual CAMS="" would otherwise launch zero workers, pass the (then vacuous)
# validation, and still run embed + the additive index.
if [ "${#CAM_ARR[@]}" -eq 0 ]; then
  echo "${T} ABORT: CAMS='${CAMS}' selected no cameras"; exit 1
fi
# Duplicate selectors would launch workers that rewrite the same output files.
if [ "$(printf '%s\n' "${CAM_ARR[@]}" | sort | uniq -d)" ]; then
  echo "${T} ABORT: duplicate camera in CAMS='${CAMS}'"; exit 1
fi
CAM_FLAGS=(); for c in "${CAM_ARR[@]}"; do CAM_FLAGS+=(--camera "$c"); done
# Fail fast on a typo / wrong scope before any GPU time is spent.
castlerag preprocess --config "${CONF}" --day "${DAY}" "${CAM_FLAGS[@]}" --dry-run || exit 1

# Raw video present for every requested camera? Only the base pass cuts frames
# from it; SKIP_BASE / SKIP_PREPROCESS runs (e.g. the re-flag events rebuild,
# after the raw video was deleted) work from the existing chunks.
if [ "${SKIP_BASE}" != "1" ] && [ "${SKIP_PREPROCESS}" != "1" ]; then
  for c in "${CAM_ARR[@]}"; do
    n=$(ls "${RAW}/${c}/video/"*.mp4 2>/dev/null | wc -l)
    echo "${T} raw video ${c}: ${n} hour files"
    [ "${n}" -gt 0 ] || { echo "${T} ABORT: no raw day-${DAY} video for ${c}"; exit 2; }
  done
fi

# --- Phase 0b: rollback material -----------------------------------------
BK="$(dirname "${EMB}")/embeddings_backup_pre_ingest_day${DAY}_${J}"
mkdir -p "${BK}"
shopt -s nullglob
DAY_NPZ=("${EMB}"/*_day"${DAY}".npz)
shopt -u nullglob
# transcripts.pkl (BM25) is rebuilt from every day by the index step, so it is
# always backed up; visual_text.json too when present (#64).
BK_FILES=()
if [ "${BOOTSTRAP}" != "1" ] || [ -e "${EMB}/transcripts.pkl" ]; then
  BK_FILES+=("${EMB}/transcripts.pkl")
fi
[ -e "${EMB}/visual_text.json" ] && BK_FILES+=("${EMB}/visual_text.json")
if [ "${#DAY_NPZ[@]}" -gt 0 ]; then
  FRESH=0
  BK_FILES+=("${DAY_NPZ[@]}" "${EMB}/manifest_day${DAY}.json")
elif [ "${DAY_PTS}" -gt 0 ]; then
  echo "${T} ABORT: ${COLL} has ${DAY_PTS} day${DAY} points but ${EMB} has no *_day${DAY}.npz."
  echo "${T}        Wrong EMB path or purged caches; restore them before an additive ingest."
  exit 2
else
  FRESH=1
  echo "${T} fresh day: no day${DAY} points and no *_day${DAY}.npz caches yet"
fi
# The later additive index upsert + transcripts.pkl rebuild are only safe with
# real rollback material, so verify the copy instead of trusting it.
if [ "${#BK_FILES[@]}" -eq 0 ]; then
  echo "${T} BOOTSTRAP: nothing in ${EMB} to back up yet"
elif ! cp -p "${BK_FILES[@]}" "${BK}/"; then
  echo "${T} ABORT: backing up embedding caches from ${EMB} failed"; exit 2
fi
if { [ "${BOOTSTRAP}" != "1" ] && [ ! -s "${BK}/transcripts.pkl" ]; } || { [ "${FRESH}" = "0" ] && \
     { ! ls "${BK}"/*_day"${DAY}".npz >/dev/null 2>&1 || [ ! -s "${BK}/manifest_day${DAY}.json" ]; }; }; then
  echo "${T} ABORT: backup in ${BK} is incomplete:"; ls -la "${BK}"; exit 2
fi
echo "${T} embedding caches backed up -> ${BK}:"; ls "${BK}"
if [ "${SNAPSHOT}" = "1" ] && [ "${COLL_EXISTS}" = "1" ]; then
  curl -sf -X POST "http://localhost:6333/collections/${COLL}/snapshots?wait=true" \
    | python3 -c 'import sys,json;print("[ingest] qdrant snapshot:",json.load(sys.stdin)["result"]["name"])' \
    || echo "${T} WARNING: snapshot failed (continuing; caches are backed up)"
fi
OTHER_SUM="$(pwd)/${L}_other_sha_${J}.txt"
is_target() { local x; for x in "${CAM_ARR[@]}"; do [ "$1" = "$x" ] && return 0; done; return 1; }
[ "${FRESH}" = "1" ] && mkdir -p "${CHUNKS}"
[ -d "${CHUNKS}" ] || { echo "${T} ABORT: ${CHUNKS} missing although day${DAY} is already indexed"; exit 2; }
shopt -s nullglob
NT_DIRS=()
for d in "${CHUNKS}"/*/; do
  d="$(basename "$d")"; is_target "$d" || NT_DIRS+=("$d")
done
shopt -u nullglob
SUM_CHECK=1
if [ "${#NT_DIRS[@]}" -eq 0 ]; then
  if [ "${FRESH}" = "1" ]; then
    # Nothing else exists for this day yet, so there is nothing to protect.
    SUM_CHECK=0; : > "${OTHER_SUM}"
    echo "${T} no other cameras' chunks in ${CHUNKS} yet; checksum baseline not needed"
  else
    echo "${T} ABORT: day${DAY} is indexed but ${CHUNKS} has no non-target camera dirs"; exit 2
  fi
else
  # An empty baseline would make the later `sha1sum -c` pass vacuously, so the
  # build must succeed and cover every non-target camera dir that exists.
  if ! ( cd "${CHUNKS}" && for d in "${NT_DIRS[@]}"; do find "$d" -name '*.jsonl' -print0; done \
         | sort -z | xargs -0 -r sha1sum ) > "${OTHER_SUM}"; then
    echo "${T} ABORT: building the checksum baseline in ${CHUNKS} failed"; exit 2
  fi
  if [ ! -s "${OTHER_SUM}" ]; then
    echo "${T} ABORT: checksum baseline is empty (${#NT_DIRS[@]} non-target dirs in ${CHUNKS})"
    exit 2
  fi
  echo "${T} checksummed $(wc -l < "${OTHER_SUM}") chunk files of ${#NT_DIRS[@]} non-target cameras"
fi

# --- GPU servers: 2x Qwen3-VL (caption/events) + OmniEmbed (embed) --------
G0=""; G1=""
if [ "${SKIP_PREPROCESS}" != "1" ]; then
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8201 \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl \
    --gpu-memory-utilization 0.85 --max-model-len 4096 --trust-remote-code \
    > "${L}_gen0_${J}.log" 2>&1 &
G0=$!
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8202 \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl \
    --gpu-memory-utilization 0.85 --max-model-len 4096 --trust-remote-code \
    > "${L}_gen1_${J}.log" 2>&1 &
G1=$!
fi
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/omniembed_server.py --port 8200 \
    > "${L}_omni_${J}.log" 2>&1 &
EMBP=$!
PIDS_TO_KILL="${QP} ${G0} ${G1} ${EMBP}"
if [ "${SKIP_PREPROCESS}" != "1" ]; then
  wait_ready "http://localhost:8201/v1/models" gen0 "${L}_gen0_${J}.log" "${G0}" || exit 1
  wait_ready "http://localhost:8202/v1/models" gen1 "${L}_gen1_${J}.log" "${G1}" || exit 1
fi
# Check OmniEmbed now, not after the multi-hour caption phase (re-checked before embed).
wait_ready "http://localhost:8200/v1/models" embed "${L}_omni_${J}.log" "${EMBP}" || exit 1

# --- Phase 1: base + caption + events, (camera x hour-group) workers --------
# Safe in parallel now that caption/events are camera- AND hour-scoped: each
# worker only rewrites its own <cam>/<HH>/clips.jsonl + events.jsonl. Captioning
# is sequential per worker, so each camera's hours are split into SPLIT groups
# to keep ~TARGET_WORKERS clients batching on the 2 gen servers (full_day1 ran
# 10). One camera alone (e.g. Reading) -> 10 workers, not 1.
read -r -a HOUR_ARR <<< "${HOURS}"
if [ "${#HOUR_ARR[@]}" -eq 0 ]; then
  echo "${T} ABORT: HOURS='${HOURS}' selected no hours"; exit 1
fi
if [ "$(printf '%s\n' "${HOUR_ARR[@]}" | sort | uniq -d)" ]; then
  echo "${T} ABORT: duplicate hour in HOURS='${HOURS}'"; exit 1
fi
TARGET_WORKERS="${TARGET_WORKERS:-10}"
case "${TARGET_WORKERS}" in
  ''|*[!0-9]*|0) echo "${T} ABORT: TARGET_WORKERS='${TARGET_WORKERS}' is not a positive integer"; exit 1;;
esac
if [ "${SPLIT:-auto}" = "auto" ]; then
  SPLIT=$(( (TARGET_WORKERS + ${#CAM_ARR[@]} - 1) / ${#CAM_ARR[@]} ))
fi
case "${SPLIT}" in
  ''|*[!0-9]*|0) echo "${T} ABORT: SPLIT='${SPLIT}' is not a positive integer"; exit 1;;
esac
[ "${SPLIT}" -gt "${#HOUR_ARR[@]}" ] && SPLIT="${#HOUR_ARR[@]}"
if [ "${SKIP_PREPROCESS}" = "1" ]; then
  echo "${T} === Phase 1 skipped (SKIP_PREPROCESS=1): reusing chunks of ${CAM_ARR[*]} ==="
else
echo "${T} === Phase 1: preprocess ${CAM_ARR[*]} hours=${HOURS} split=${SPLIT} $(date -Iseconds) ==="
BASE_FLAG=(); [ "${SKIP_BASE}" = "1" ] && BASE_FLAG=(--skip-base)
CAPTION_FLAG=(--caption); [ "${CAPTION}" = "0" ] && CAPTION_FLAG=()
if [ "${CAPTION}" = "0" ] && [ "${SKIP_BASE}" != "1" ]; then
  echo "${T} ABORT: CAPTION=0 needs SKIP_BASE=1 (a new base pass needs captions)"; exit 1
fi
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
        castlerag preprocess --config "${CONF}" --day "${DAY}" --camera "${cam}" "${HFLAGS[@]}" \
          "${BASE_FLAG[@]}" "${CAPTION_FLAG[@]}" --events > "${L}_cam_${tag}_${J}.log" 2>&1
      rc=$?; echo "${T} ${tag} (port ${port}) rc=${rc} in $(( $(date +%s)-T0 ))s"; exit $rc
    ) & P+=($!); PC+=("${tag}")
  done
done
echo "${T} launched ${w} preprocess workers"
FAIL=0
for i in "${!P[@]}"; do wait "${P[$i]}" || { echo "${T} CAMERA ${PC[$i]} FAILED"; FAIL=1; }; done
echo "${T} === Phase 1 done $(date -Iseconds) FAIL=${FAIL} ==="
[ "${FAIL}" -ne 0 ] && { echo "${T} ABORT before embed/index (see ${L}_cam_*_${J}.log)"; exit 1; }
fi

# --- Phase 1b: validate ----------------------------------------------------
# Fixed cams: camera_type == {"fixed"}, room == {<cam>}. Ego cams:
# camera_type == {"ego"}, room == {None}. Both: >=98 % of real (non-placeholder)
# clips captioned, no corrupt
# lines, and events wherever they can exist: event compression needs 4 adjacent
# non-placeholder clips, which a mostly idle camera may never have (day 1
# Reading: 1147 of 1201 clips blank or frozen, 0 events).
FIXED_CAMS="${FIXED_CAMS}" python3 - "${CHUNKS}" "${CAM_ARR[@]}" <<'PY' || { echo "${T} VALIDATION FAILED"; exit 1; }
import glob, json, os, sys
root, cams = sys.argv[1], sys.argv[2:]
fixed = set(os.environ.get("FIXED_CAMS", "").split())
ok = True
for cam in cams:
    tot = real = cap = bad = ev = groups = 0; rooms = set(); types = set()
    for f in glob.glob(f"{root}/{cam}/*/clips.jsonl"):
        usable = []
        for line in open(f):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                bad += 1; continue
            tot += 1
            rooms.add(d.get("room")); types.add(d.get("camera_type"))
            if not d.get("is_placeholder"):  # test card / blank: never captioned
                real += 1; cap += bool(d.get("clip_caption"))
                usable.append((d.get("absolute_start"), d.get("absolute_end")))
        # Same grouping as `preprocess --events`: consecutive runs of 4 usable
        # clips, kept only when each clip ends where the next one starts.
        usable.sort(key=lambda c: c[0])
        for i in range(0, len(usable) - 3, 4):
            g = usable[i:i + 4]
            groups += all(g[k][1] == g[k + 1][0] for k in range(3))
    for f in glob.glob(f"{root}/{cam}/*/events.jsonl"):
        ev += sum(1 for line in open(f) if line.strip())
    pct = 100 * cap / real if real else 100.0
    want_type, want_room = ({"fixed"}, {cam}) if cam in fixed else ({"ego"}, {None})
    print(f"  {cam}: clips={tot} real={real} captioned={cap} ({pct:.0f}%) events={ev} "
          f"(possible {groups}) bad={bad} camera_type={types} room={rooms}")
    if bad or tot == 0 or (groups and ev == 0) or pct < 98 or types != want_type or rooms != want_room:
        ok = False
sys.exit(0 if ok else 1)
PY
if [ "${SUM_CHECK}" = "1" ]; then
  ( cd "${CHUNKS}" && sha1sum --quiet -c "${OTHER_SUM}" ) \
    || { echo "${T} ABORT: non-target chunk files changed!"; exit 1; }
  echo "${T} non-target chunk files unchanged"
fi

# Free the caption GPUs before embedding (OmniEmbed stays on GPU2).
kill "${G0}" "${G1}" 2>/dev/null; PIDS_TO_KILL="${QP} ${EMBP}"

# --- Phase 2: embed per modality (incremental: only new record ids) --------
wait_ready "http://localhost:8200/v1/models" embed "${L}_omni_${J}.log" "${EMBP}" || exit 1
export VLLM_BASE_URL="http://localhost:8200/v1"
for m in transcript event_summary video; do
  echo "${T} === embed --day ${DAY} --modality ${m} $(date -Iseconds) ==="
  castlerag embed --config "${CONF}" --day "${DAY}" --modality "${m}" \
    || { echo "${T} embed ${m} FAILED"; exit 1; }
done

# --- Phase 3: additive index (NO --create-collection) ----------------------
echo "${T} === index --day ${DAY} (additive) $(date -Iseconds) ==="
castlerag index --config "${CONF}" --day "${DAY}" || { echo "${T} index FAILED"; exit 1; }

# --- Phase 4: verify -------------------------------------------------------
python scripts/qdrant_camera_counts.py --collection "${COLL}" --day "day${DAY}" \
  | tee "${L}_counts_after_${J}.txt"
python scripts/qdrant_camera_counts.py --collection "${COLL}" --day "day${DAY}" \
  --cameras "${CAM_ARR[@]}" --missing-only --fail-if-missing \
  || { echo "${T} VERIFY FAILED: some requested cameras still have 0 main_clip points"; exit 1; }
echo "${T} DONE $(date -Iseconds). Next, on the login node: DAYS=\"<every day indexed so far>\" bash scripts/archive_bugb.sh"
if [ "${DAY}" = "1" ]; then
  echo "${T}   then the 40-q day-1 eval:"
  echo "${T}   sbatch --account=gisr109364 --export=ALL,CONF=${CONF} scripts/slurm/smoke_day1_roomfix.slurm"
fi
