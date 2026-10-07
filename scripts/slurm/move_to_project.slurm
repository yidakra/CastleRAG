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

# A job still writing chunks, caches or Qdrant would leave a torn copy.
BUSY="$(squeue -u "$USER" -h -o "%i %j" \
        | grep -E "castle-(ingest|thin|upload|reflag|regen)" | grep -v "^${SLURM_JOB_ID:-x} ")"
if [ -n "$BUSY" ]; then
  echo "ABORT: jobs that write derived data are queued or running:"; echo "$BUSY"; exit 2
fi

src_paths() { castlerag paths --config "$SRC_CONF"; }
dst_paths() { castlerag paths --config "$DST_CONF"; }
S="$(src_paths)" || exit 2; D="$(dst_paths)" || exit 2
get() { printf '%s\n' "$1" | sed -n "s/^$2=//p" | head -1; }
SRC_CHUNKS="$(get "$S" CHUNKS_DIR)"; DST_CHUNKS="$(get "$D" CHUNKS_DIR)"
SRC_EMB="$(get "$S" EMB_DIR)";       DST_EMB="$(get "$D" EMB_DIR)"
SRC_FR="$(get "$S" FRAMES_DIR)";     DST_FR="$(get "$D" FRAMES_DIR)"
SRC_Q="$(get "$S" QDRANT_STORAGE)";  DST_Q="$(get "$D" QDRANT_STORAGE)"
for v in SRC_CHUNKS DST_CHUNKS SRC_EMB DST_EMB SRC_FR DST_FR SRC_Q DST_Q; do
  [ -n "${!v}" ] || { echo "ABORT: $v is empty"; exit 2; }
done
[ "$SRC_CHUNKS" != "$DST_CHUNKS" ] || { echo "ABORT: source and destination are the same"; exit 2; }
echo "start $(date -Iseconds): $SRC_CONF -> $DST_CONF, days: $DAYS"

count() { find "$1" -type f 2>/dev/null | wc -l; }
fail=0
for d in $DAYS; do
  [ -d "$SRC_FR/day$d" ] || { echo "day$d: no frames on scratch, skipped"; continue; }
  mkdir -p "$DST_FR/day$d"
  # One rsync per camera directory, 8 at a time: GPFS metadata is the cost.
  ls "$SRC_FR/day$d" | xargs -P 8 -I{} rsync -a "$SRC_FR/day$d/{}" "$DST_FR/day$d/" || fail=1
  s=$(count "$SRC_FR/day$d"); t=$(count "$DST_FR/day$d")
  echo "day$d frames: $s on scratch, $t in project"; [ "$t" -ge "$s" ] || fail=1
done
mkdir -p "$DST_CHUNKS" "$DST_EMB" "$DST_Q"
rsync -a "$SRC_CHUNKS/" "$DST_CHUNKS/" || fail=1
rsync -a "$SRC_EMB/" "$DST_EMB/" || fail=1
rsync -a "$SRC_Q/" "$DST_Q/" || fail=1
for pair in "$SRC_CHUNKS $DST_CHUNKS" "$SRC_EMB $DST_EMB" "$SRC_Q $DST_Q"; do
  read -r s t <<< "$pair"
  echo "$(basename "$s"): $(count "$s") files on scratch, $(count "$t") in project"
  [ "$(count "$t")" -ge "$(count "$s")" ] || fail=1
done
du -sh "$DST_FR" "$DST_CHUNKS" "$DST_EMB" "$DST_Q" 2>/dev/null
echo "end $(date -Iseconds) fail=$fail"
[ "$fail" = 0 ]
