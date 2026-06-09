#!/bin/bash
# preprocess_aux.slurm
# Normalize auxiliary modalities: heartrate, gaze, photo, thermal, auxiliary video.
# Can run in parallel with caption_ocr.slurm (no dependency on it).
#
# Usage:
#   sbatch --account=<account> --dependency=afterok:<preprocess_main_job_id> \
#          scripts/slurm/preprocess_aux.slurm

#SBATCH --job-name=castle-preprocess-aux
#SBATCH --partition=thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG="${CONFIG:-configs/snellius.yaml}"
VENV="${VENV:-${HOME}/castlerag_venv}"

echo "[preprocess_aux]  host=$(hostname)  start=$(date -Iseconds)"

module purge
module load 2024
module load Python/3.11.3-GCCcore-12.3.0

source "${VENV}/bin/activate"

echo "preprocess_aux not yet implemented — see issue #4"
exit 1
