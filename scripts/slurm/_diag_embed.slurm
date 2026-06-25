#!/bin/bash
# _diag_embed.slurm — pinpoint WHERE the OmniEmbed load hangs. Runs alone on one
# GPU, probes a bare CUDA op first, then loads the model under faulthandler so a
# stall dumps the exact Python stack instead of going silent.
#SBATCH --job-name=castle-diag-embed
#SBATCH --partition=gpu_a100
#SBATCH --account=gpuuva082
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
export HF_HOME="/scratch-shared/${USER}/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
module purge; module load 2024; module load Python/3.12.3-GCCcore-13.3.0; module load CUDA/12.6.0
source "${HOME}/castlerag_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

echo "[diag] host=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>&1 | head -3

python - <<'PY'
import os, sys, faulthandler, threading, time
faulthandler.dump_traceback_later(90, repeat=True)  # every 90s, print stacks of all threads
def stamp(m): print(f"[diag] {time.strftime('%H:%M:%S')} {m}", flush=True)

import torch
stamp(f"torch={torch.__version__} cuda_avail={torch.cuda.is_available()} "
      f"torch_cuda={torch.version.cuda} dev={torch.cuda.get_device_name(0)}")

stamp("STEP 1: bare CUDA op — torch.zeros(1024,1024).cuda()")
t = torch.zeros(1024, 1024).cuda(); torch.cuda.synchronize()
stamp(f"STEP 1 OK — tensor on {t.device}, freeing")
del t; torch.cuda.empty_cache()

import transformers, peft
stamp(f"transformers={transformers.__version__} peft={peft.__version__}")
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
BASE = "Tevatron/Qwen2.5-Omni-7B-Thinker"

stamp("STEP 2: from_pretrained(bf16, disable_mmap=True) on CPU (no .to yet)")
m = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    BASE, dtype=torch.bfloat16, attn_implementation="sdpa", disable_mmap=True)
stamp("STEP 2 OK — model built on CPU")

stamp("STEP 3: .to('cuda')  <-- prime suspect")
m = m.to("cuda")
torch.cuda.synchronize()
stamp("STEP 3 OK — model on GPU")

stamp("STEP 4: PeftModel.from_pretrained(adapter)")
from peft import PeftModel
m = PeftModel.from_pretrained(m, "Tevatron/OmniEmbed-v0.1-multivent")
torch.cuda.synchronize()
stamp("STEP 4 OK — adapter applied. ALL STEPS PASSED.")
faulthandler.cancel_dump_traceback_later()
PY
echo "[diag] exit=$? end=$(date -Iseconds)"
