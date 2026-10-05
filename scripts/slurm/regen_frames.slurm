#!/bin/bash
#SBATCH --job-name=castle-regen-frames
#SBATCH --account=gisr109364
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Re-create a day's thinned frames from the raw video after a scratch purge
# (CPU only; see scripts/regen_frames.py). Needs the day's video downloaded
# again and its chunks in place. Usage, from the repo root:
#   sbatch --export=ALL,DAY=1 scripts/slurm/regen_frames.slurm
# Optional: CONF (default configs/snellius_fixedcams.yaml), VIDEO_ROOT (dir
# holding dayN/ if the video moved since ingest), WORKERS (default 48).
set -uo pipefail
DAY="${DAY:?set DAY=1..4}"
case "$DAY" in 1|2|3|4) ;; *) echo "ABORT: DAY must be 1-4"; exit 1;; esac
CONF="${CONF:-configs/snellius_fixedcams.yaml}"
WORKERS="${WORKERS:-48}"
mkdir -p logs
module purge; module load 2024; module load Python/3.12.3-GCCcore-13.3.0 FFmpeg/7.0.2-GCCcore-13.3.0
source "${HOME}/castlerag_venv/bin/activate"
ARGS=(--config "$CONF" --day "$DAY")
[ -n "${VIDEO_ROOT:-}" ] && ARGS+=(--video-root "$VIDEO_ROOT")
python scripts/regen_frames.py "${ARGS[@]}" || exit 2   # dry run: what is missing
python scripts/regen_frames.py "${ARGS[@]}" --apply --workers "$WORKERS"
