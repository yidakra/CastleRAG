#!/bin/bash
# test_omniembed.slurm — cheap validation that the custom OmniEmbed /v1/embeddings
# server loads and produces sane embeddings before any full pipeline run.
#SBATCH --job-name=castle-test-omni
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail
export HF_HOME="/scratch-shared/${USER}/hf_cache"
export HF_HUB_OFFLINE=1
echo "[test-omni] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1) start=$(date -Iseconds)"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
source "${HOME}/castlerag_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p logs

python scripts/omniembed_server.py --port 8200 > "logs/omni_test_server_${SLURM_JOB_ID}.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

echo "[test-omni] waiting for server..."
for i in $(seq 1 90); do
  curl -sf http://localhost:8200/v1/models >/dev/null 2>&1 && { echo "[test-omni] server ready ($((i*10))s)"; break; }
  if ! kill -0 $SPID 2>/dev/null; then echo "[test-omni] SERVER DIED"; tail -40 "logs/omni_test_server_${SLURM_JOB_ID}.log"; exit 1; fi
  sleep 10
done
curl -sf http://localhost:8200/v1/models >/dev/null 2>&1 \
  || { echo "[test-omni] ERROR: server not ready after 900s"; tail -40 "logs/omni_test_server_${SLURM_JOB_ID}.log"; exit 1; }

python - <<'PY'
import requests, numpy as np
texts = [
  "Query: sausages for breakfast in the kitchen",
  "Query: an Irish breakfast with sausages",
  "Query: a car driving fast on a highway at night",
]
r = requests.post("http://localhost:8200/v1/embeddings",
                  json={"input": texts, "model": "Tevatron/OmniEmbed-v0.1-multivent"}, timeout=120)
r.raise_for_status()
V = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
print("embedding shape:", V.shape)
print("all finite:", bool(np.isfinite(V).all()))
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
print(f"cos(similar  0,1) = {cos(V[0],V[1]):.4f}")
print(f"cos(different 0,2) = {cos(V[0],V[2]):.4f}")
assert V.shape[0]==3 and V.shape[1]>=256, "unexpected embedding shape"
assert cos(V[0],V[1]) > cos(V[0],V[2]), "similar pair should score higher than different pair"
print("OMNIEMBED TEST PASSED")
PY
echo "[test-omni] DONE $(date -Iseconds)"
