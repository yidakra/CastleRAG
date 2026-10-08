#!/bin/bash
#SBATCH --job-name=castle-move-project
#SBATCH --account=gisr109364
#SBATCH --partition=staging
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
# Copy every derived artifact from scratch (14-day purge) to the project space,
# so later jobs can run with CONF=configs/snellius_project.yaml. Raw video
# stays on scratch. Usage, from the repo root, once nothing writes to the
# derived data:
#   sbatch scripts/slurm/move_to_project.slurm
# Optional: DAYS (frames/chunks to copy, default "1 2 3 4"), SRC_CONF and
# DST_CONF (defaults: configs/snellius_fixedcams.yaml -> snellius_project.yaml).
# rsync makes it resumable; it copies, never deletes. See
# docs/ingest_all_days.md, "Move to the project space".
set -uo pipefail
DAYS="${DAYS:-1 2 3 4}"
SRC_CONF="${SRC_CONF:-configs/snellius_fixedcams.yaml}"
DST_CONF="${DST_CONF:-configs/snellius_project.yaml}"
mkdir -p logs
module purge >/dev/null 2>&1; module load 2024 >/dev/null 2>&1
module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1
source "${HOME}/castlerag_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1

# A job still writing chunks, caches or Qdrant would leave a torn copy. Check
# squeue's own status first: an empty answer from a failed squeue is not "idle".
JOBS="$(squeue -u "$USER" -h -o "%i %j")" || { echo "ABORT: squeue failed"; exit 2; }
BUSY="$(printf '%s\n' "$JOBS" | grep -E "castle-(ingest|thin|upload|reflag|regen)" \
        | grep -v "^${SLURM_JOB_ID:-x} ")"
if [ -n "$BUSY" ]; then
  echo "ABORT: jobs that write derived data are queued or running:"; echo "$BUSY"; exit 2
fi

# `castlerag paths` prints shell-quoted assignments; eval them (prefixed) so
# quoting is decoded the way the shell would.
S="$(castlerag paths --config "$SRC_CONF")" || exit 2
D="$(castlerag paths --config "$DST_CONF")" || exit 2
eval "$(printf '%s\n' "$S" | sed 's/^/SRC_/')"
eval "$(printf '%s\n' "$D" | sed 's/^/DST_/')"
SRC_CHUNKS="$SRC_CHUNKS_DIR"; DST_CHUNKS="$DST_CHUNKS_DIR"
SRC_EMB="$SRC_EMB_DIR";       DST_EMB="$DST_EMB_DIR"
SRC_FR="$SRC_FRAMES_DIR";     DST_FR="$DST_FRAMES_DIR"
SRC_Q="$SRC_QDRANT_STORAGE";  DST_Q="$DST_QDRANT_STORAGE"
for v in SRC_CHUNKS DST_CHUNKS SRC_EMB DST_EMB SRC_FR DST_FR SRC_Q DST_Q; do
  [ -n "${!v}" ] || { echo "ABORT: $v is empty"; exit 2; }
done
[ "$SRC_CHUNKS" != "$DST_CHUNKS" ] || { echo "ABORT: source and destination are the same"; exit 2; }
echo "start $(date -Iseconds): $SRC_CONF -> $DST_CONF, days: $DAYS"

count() { find "$1" -type f 2>/dev/null | wc -l; }
fail=0
for d in $DAYS; do
  if [ ! -d "$SRC_FR/day$d" ]; then
    # Chunks without frames would point at frames that were never moved.
    if [ -d "$SRC_CHUNKS/day$d" ]; then
      echo "day$d: chunks but no frames on scratch; FAILED (leave the day out of DAYS if intended)"
      fail=1
    else
      echo "day$d: not ingested, skipped"
    fi
    continue
  fi
  mkdir -p "$DST_FR/day$d"
  # One rsync per camera directory, 8 at a time: GPFS metadata is the cost.
  # --delete inside each destination camera dir drops frames thinned away on
  # scratch since an earlier copy; the scratch source is never touched.
  find "$SRC_FR/day$d" -mindepth 1 -maxdepth 1 -type d -print0 \
    | xargs -0 -P 8 -I{} sh -c 'rsync -a --delete "$1/" "$2/$(basename "$1")/"' _ {} "$DST_FR/day$d" \
    || fail=1
  s=$(count "$SRC_FR/day$d"); t=$(count "$DST_FR/day$d")
  echo "day$d frames: $s on scratch, $t in project"; [ "$t" -eq "$s" ] || fail=1
done
mkdir -p "$DST_CHUNKS" "$DST_EMB" "$DST_Q"
rsync -a "$SRC_CHUNKS/" "$DST_CHUNKS/" || fail=1
rsync -a "$SRC_EMB/" "$DST_EMB/" || fail=1
rsync -a "$SRC_Q/" "$DST_Q/" || fail=1
SRCS=("$SRC_CHUNKS" "$SRC_EMB" "$SRC_Q"); DSTS=("$DST_CHUNKS" "$DST_EMB" "$DST_Q")
for i in "${!SRCS[@]}"; do
  s="${SRCS[$i]}"; t="${DSTS[$i]}"
  echo "$(basename "$s"): $(count "$s") files on scratch, $(count "$t") in project"
  [ "$(count "$t")" -ge "$(count "$s")" ] || fail=1
done
du -sh "$DST_FR" "$DST_CHUNKS" "$DST_EMB" "$DST_Q" 2>/dev/null
echo "end $(date -Iseconds) fail=$fail"
[ "$fail" = 0 ]
