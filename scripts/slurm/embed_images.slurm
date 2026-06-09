#!/bin/bash
# embed_images.slurm
# OmniEmbed encoding for photos and thermal images.
# Batch size: 16 (image modality) — see SPEC §3.2.
#
# Usage:
#   sbatch --account=<account> --dependency=afterok:<preprocess_aux_job_id> \
#          scripts/slurm/embed_images.slurm

#SBATCH --job-name=castle-embed-images
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG="${CONFIG:-configs/snellius.yaml}"
VENV="${VENV:-${HOME}/castlerag_venv}"

echo "[embed_images]  host=$(hostname)  start=$(date -Iseconds)"

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.11.3-GCCcore-12.3.0

source "${VENV}/bin/activate"

castlerag embed \
    --config "${CONFIG}" \
    --modality image

echo "[embed_images]  done=$(date -Iseconds)"
