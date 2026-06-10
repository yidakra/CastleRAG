#!/bin/bash
# index_transcripts.slurm
# Build dense transcript OmniEmbed embeddings from normalized utterance windows.
# Array: one job per day (1-4). Transcript BM25 is finalized during the
# collection build in index_qdrant.slurm from the full transcript corpus.
# This is the GPU part of transcript indexing; BM25 itself is CPU-heavy.
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
#SBATCH --array=1-4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

DAY="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIG:-configs/snellius.yaml}"
VENV="${VENV:-${HOME}/castlerag_venv}"

echo "[index_transcripts] day=${DAY}  host=$(hostname)  start=$(date -Iseconds)"

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.11.3-GCCcore-12.3.0

source "${VENV}/bin/activate"

castlerag embed \
    --config "${CONFIG}" \
    --modality transcript \
    --day "${DAY}"

echo "[index_transcripts] day=${DAY}  done=$(date -Iseconds)"
