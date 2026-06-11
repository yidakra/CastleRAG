#!/bin/bash
# compress_events.slurm
# Build 2-minute event summaries from groups of 4 adjacent captioned clips.
# Array: one job per day (1–4).
# Depends on: caption_ocr.slurm completing successfully.
#
# Estimated GPU cost: 8–16 GPU-hours total (see SPEC §3.5).
#
# Usage:
#   sbatch --account=<account> --dependency=afterok:<caption_ocr_job_id> \
#          scripts/slurm/compress_events.slurm

#SBATCH --job-name=castle-compress-events
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

echo "[compress_events] day=${DAY}  host=$(hostname)  start=$(date -Iseconds)"

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.11.3-GCCcore-12.3.0

source "${VENV}/bin/activate"

if [[ -z "${VLLM_BASE_URL:-}" ]]; then
  echo "[compress_events] ERROR: VLLM_BASE_URL is not set. Export it before submitting this job." >&2
  exit 1
fi

castlerag preprocess --events --config "${CONFIG}" --day "${DAY}"

echo "[compress_events] done  day=${DAY}  end=$(date -Iseconds)"
