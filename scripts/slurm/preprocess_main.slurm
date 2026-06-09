#!/bin/bash
# preprocess_main.slurm
# CASTLE main-video preprocessing: clip windowing, subclip extraction, 1 fps frame sampling.
# Array: one job per day (1–4).  Override with --array=1 to process only day1.
#
# Usage:
#   sbatch --account=<account> scripts/slurm/preprocess_main.slurm
#   sbatch --account=<account> --array=2 scripts/slurm/preprocess_main.slurm

#SBATCH --job-name=castle-preprocess
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

DAY="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIG:-configs/snellius.yaml}"
VENV="${VENV:-${HOME}/castlerag_venv}"

echo "[preprocess_main] day=${DAY}  host=$(hostname)  start=$(date -Iseconds)"

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.11.3-GCCcore-12.3.0
module load FFmpeg/6.0-GCCcore-12.3.0

source "${VENV}/bin/activate"

mkdir -p logs

castlerag preprocess \
    --config "${CONFIG}" \
    --day "${DAY}"

echo "[preprocess_main] day=${DAY}  done=$(date -Iseconds)"
