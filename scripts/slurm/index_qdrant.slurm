#!/bin/bash
# index_qdrant.slurm
# Collection creation, payload index creation, and batched upsert into Qdrant.
# Run after all embed_*.slurm jobs finish.
# Qdrant must be started on the compute node before this job runs.
#
# Usage:
#   # 1. Start Qdrant (adjust path to your binary or Docker socket):
#   qdrant &
#   sleep 5
#
#   # 2. Submit indexing job:
#   sbatch --account=<account> \
#          --dependency=afterok:<embed_text_id>:<embed_video_id>:<embed_images_id> \
#          scripts/slurm/index_qdrant.slurm

#SBATCH --job-name=castle-index-qdrant
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG="${CONFIG:-configs/snellius.yaml}"
VENV="${VENV:-${HOME}/castlerag_venv}"
CREATE_COLLECTION="${CREATE_COLLECTION:-}"  # pass --create-collection to recreate

echo "[index_qdrant]  host=$(hostname)  start=$(date -Iseconds)"

module purge
module load 2024
# 3.12.3/GCCcore-13.3.0 matches the venv AND provides the GLIBCXX_3.4.30 that the
# Qdrant binary needs on this RHEL9 node (the 3.11.3/GCCcore-12.3.0 libstdc++ is
# not placed on LD_LIBRARY_PATH and Qdrant fails to start under it).
module load Python/3.12.3-GCCcore-13.3.0

source "${VENV}/bin/activate"

# Start Qdrant for the lifetime of this job, with storage on shared scratch so
# the collection survives the job and is reachable from the eval node.  No `cd`
# (keeps the relative --config path valid); storage path set via env override.
export QDRANT__STORAGE__STORAGE_PATH="/scratch-shared/${USER}/qdrant_storage/storage"
"${HOME}/qdrant/qdrant" &
QDRANT_PID=$!
sleep 5
trap 'kill $QDRANT_PID 2>/dev/null' EXIT

# --day "${CASTLE_DAY}" matches the day-suffixed embed caches (*_dayN.npz) so the
# cache short-circuit skips re-embedding — no vLLM embed server needed here.
castlerag index \
    --config "${CONFIG}" \
    ${CASTLE_DAY:+--day "${CASTLE_DAY}"} \
    ${CREATE_COLLECTION:+--create-collection}

echo "[index_qdrant]  done=$(date -Iseconds)"
