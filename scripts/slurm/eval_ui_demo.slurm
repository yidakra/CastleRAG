#!/bin/bash
#SBATCH --job-name=castle-eval-ui-demo
#SBATCH --partition=gpu_a100
#SBATCH --account=gpuuva082
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail
SC="/scratch-shared/${USER}"
CONF="configs/snellius_me.yaml"
QFILE="data/smoke_day1.csv"   # 40 day-1 questions with ground truth

export HF_HOME="${SC}/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_LOGGING_LEVEL=WARNING
module purge; module load 2024; module load Python/3.12.3-GCCcore-13.3.0; module load CUDA/12.6.0
source "${HOME}/castlerag_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

wait_ready() {
  for i in $(seq 1 120); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 10; done; return 1
}

# Hard gate: a server that never binds must fail the job immediately (and dump its
# log) instead of letting set -e silently swallow the failure inside an && chain.
require_ready() {  # name url logfile
  if wait_ready "$2"; then
    echo "[eval] $1 ready"
  else
    echo "[eval] FATAL: $1 never became ready at $2 — last 30 log lines:" >&2
    grep -av "Loading weights" "$3" 2>/dev/null | tail -30 >&2 || true
    exit 1
  fi
}

CUDA_VISIBLE_DEVICES=0 python scripts/omniembed_server.py --port 8200 > logs/eval_embed_${SLURM_JOB_ID}.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8201 \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct qwen3vl \
    --gpu-memory-utilization 0.90 --max-model-len 49152 --trust-remote-code \
    > logs/eval_vllm_${SLURM_JOB_ID}.log 2>&1 &
export QDRANT__STORAGE__STORAGE_PATH="${SC}/qdrant_storage/storage"
"${HOME}/qdrant/qdrant" > logs/eval_qdrant_${SLURM_JOB_ID}.log 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null' EXIT

require_ready qdrant http://localhost:6333/readyz   logs/eval_qdrant_${SLURM_JOB_ID}.log
require_ready embed  http://localhost:8200/v1/models logs/eval_embed_${SLURM_JOB_ID}.log
require_ready vllm   http://localhost:8201/v1/models logs/eval_vllm_${SLURM_JOB_ID}.log

export VLLM_BASE_URL=http://localhost:8201/v1
OUTDIR="${SC}/castle_outputs/eval_ui_demo_${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"

castlerag answer "${QFILE}" --config "${CONF}" --out "${OUTDIR}/predictions.json"
castlerag eval "${QFILE}" "${OUTDIR}/predictions.json" --config "${CONF}"
echo "[eval] DONE — predictions at ${OUTDIR}/predictions.json"
