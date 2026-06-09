#!/bin/bash
# embed_video.slurm
# OmniEmbed encoding for main-clip video points and auxiliary video points.
# Batch size: 4 (video is the most expensive modality) — see SPEC §3.2.
# ~48,000 main clips across 10 ego cameras.
# Estimated: 32–48 GPU-hours total; dominant cost in the offline build.
# Array: one job per day (SLURM_ARRAY_TASK_ID maps to day 1-4).
# To shard further by camera, expand to --array=1-40 and add camera-sharding
# logic (map task id to day and camera index) once issue #5 is implemented.
#
# Usage:
#   sbatch --account=<account> --dependency=afterok:<preprocess_main_job_id> \
#          scripts/slurm/embed_video.slurm

#SBATCH --job-name=castle-embed-video
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

DAY="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIG:-configs/snellius.yaml}"
VENV="${VENV:-${HOME}/castlerag_venv}"

echo "[embed_video] day=${DAY}  host=$(hostname)  start=$(date -Iseconds)"

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.11.3-GCCcore-12.3.0

source "${VENV}/bin/activate"

castlerag embed \
    --config "${CONFIG}" \
    --modality video \
    --day "${DAY}"

echo "[embed_video] day=${DAY}  done=$(date -Iseconds)"
