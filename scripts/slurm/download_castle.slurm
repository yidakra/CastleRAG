#!/bin/bash
#SBATCH --job-name=castle-download
#SBATCH --account=gisr109364
#SBATCH --partition=staging
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/castle-download_%j.out
# Download CASTLE 2024 days into /scratch-shared (14-day retention: ingest the
# days promptly). Usage:
#   sbatch --export=ALL,DAYS="1 2" scripts/slurm/download_castle.slurm
#   sbatch --export=ALL,DAYS="3 4",AUX=0 scripts/slurm/download_castle.slurm
# The auxiliary data is fetched as the single auxiliary.zip and unzipped here:
# its ~20k loose files trip Hugging Face's anonymous rate limit (HTTP 429).
# Needs huggingface_hub >= 0.25 in $HF_VENV (default ~/hfdl_venv).
set -uo pipefail
DAYS="${DAYS:?set DAYS, e.g. DAYS=\"1 2\"}"
AUX="${AUX:-1}"
HF_VENV="${HF_VENV:-$HOME/hfdl_venv}"
DEST="${DEST:-/scratch-shared/$USER/castle2024}"
export HF_HUB_DISABLE_TELEMETRY=1
read -r -a DAY_ARR <<< "${DAYS}"
for d in "${DAY_ARR[@]}"; do
  case "$d" in 1|2|3|4) ;; *) echo "ABORT: bad day '$d' (1-4)"; exit 1;; esac
done
mkdir -p "$DEST" logs
PATTERNS="README.md files.txt"
for d in "${DAY_ARR[@]}"; do PATTERNS="$PATTERNS main/day${d}/**"; done
[ "$AUX" = 1 ] && PATTERNS="$PATTERNS auxiliary.zip auxiliary.zip.sha256"
echo "start $(date -Iseconds) on $(hostname): days=${DAYS} aux=${AUX} -> ${DEST}"
ok=0
for attempt in 1 2 3 4 5; do
  if "$HF_VENV/bin/python" - "$DEST" ${PATTERNS} <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("CASTLE-Dataset/CASTLE2024", repo_type="dataset",
                  local_dir=sys.argv[1], allow_patterns=sys.argv[2:], max_workers=4)
PY
  then ok=1; echo "download ok after attempt $attempt"; break; fi
  echo "attempt $attempt failed $(date -Iseconds); retrying in 120s"; sleep 120
done
if [ "$ok" = 1 ] && [ "$AUX" = 1 ] && [ -f "$DEST/auxiliary.zip" ]; then
  (cd "$DEST" && sha256sum -c auxiliary.zip.sha256) || echo "WARN: auxiliary.zip checksum not verified"
  unzip -q -n "$DEST/auxiliary.zip" -d "$DEST" && echo "auxiliary unzipped"
fi
for d in "${DAY_ARR[@]}"; do
  echo "day${d}: $(find "$DEST/main/day${d}" -name '*.mp4' 2>/dev/null | wc -l) mp4 files, $(du -sh "$DEST/main/day${d}" 2>/dev/null | cut -f1)"
done
echo "end $(date -Iseconds) ok=$ok"
[ "$ok" = 1 ]
