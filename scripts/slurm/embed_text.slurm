#!/bin/bash
# embed_text.slurm
# OmniEmbed encoding for event-summary text points and auxiliary text records.
# Batch size: 64 (event summaries), 128 (transcript windows) — see SPEC §3.2.
# Array: one job per day.
#
# Usage:
#   sbatch --account=<account> --dependency=afterok:<compress_events_job_id> \
#          scripts/slurm/embed_text.slurm

#SBATCH --job-name=castle-embed-text
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

DAY="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIG:-configs/snellius.yaml}"
VENV="${VENV:-${HOME}/castlerag_venv}"

echo "[embed_text] day=${DAY}  host=$(hostname)  start=$(date -Iseconds)"

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.11.3-GCCcore-12.3.0

source "${VENV}/bin/activate"

castlerag embed \
    --config "${CONFIG}" \
    --modality event_summary \
    --day "${DAY}"

echo "[embed_text] day=${DAY}  done=$(date -Iseconds)"
