#!/bin/bash
# caption_ocr.slurm
# Offline frame captioning and OCR over 1 fps sampled frames for all main clips.
# Array: one job per day (1–4).
# Depends on: preprocess_main.slurm completing successfully.
#
# Estimated GPU cost: 16–32 GPU-hours total (see SPEC §3.5).
#
# Usage:
#   sbatch --account=<account> --dependency=afterok:<preprocess_job_id> \
#          scripts/slurm/caption_ocr.slurm

#SBATCH --job-name=castle-caption-ocr
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

echo "[caption_ocr] day=${DAY}  host=$(hostname)  start=$(date -Iseconds)"

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.11.3-GCCcore-12.3.0

source "${VENV}/bin/activate"

if [[ -z "${VLLM_BASE_URL:-}" ]]; then
  echo "[caption_ocr] ERROR: VLLM_BASE_URL is not set. Export it before submitting this job." >&2
  exit 1
fi

castlerag preprocess --caption --config "${CONFIG}" --day "${DAY}"

echo "[caption_ocr] done  day=${DAY}  end=$(date -Iseconds)"
