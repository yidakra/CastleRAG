#!/bin/bash
# index_transcripts.slurm
# Build the transcript BM25 index and dense transcript OmniEmbed embeddings
# from normalized utterance windows.
# This is a CPU-heavy job (BM25) plus moderate GPU work (dense embeddings).
# Estimated: 2–4 CPU-hours + 4–10 GPU-hours total (see SPEC §3.5).
#
# Usage:
#   sbatch --account=<account> --dependency=afterok:<preprocess_job_id> \
#          scripts/slurm/index_transcripts.slurm

#SBATCH --job-name=castle-index-transcripts
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG="${CONFIG:-configs/snellius.yaml}"
VENV="${VENV:-${HOME}/castlerag_venv}"

echo "[index_transcripts]  host=$(hostname)  start=$(date -Iseconds)"

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.11.3-GCCcore-12.3.0

source "${VENV}/bin/activate"

echo "index_transcripts not yet implemented — see issues #5, #6"
exit 1
