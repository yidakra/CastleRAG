#!/bin/bash
# fixedcams_day1.slurm — kept so the old commands keep working. It is the
# day-1 additive fixed-camera ingest of #50 Bug B, which now lives in the
# day-generic scripts/slurm/ingest_day.slurm; this wrapper pins DAY=1 and keeps
# the old job name, so the log is still logs/castle-fixedcams-day1_<job>.out.
# CAMS / SKIP_BASE / SPLIT / TARGET_WORKERS / HOURS / SNAPSHOT / MIN_POINTS /
# CONF pass straight through. The default CAMS=auto still selects only the day-1
# cameras that have 0 main_clip points (the missing fixed cameras).
#
#   sbatch --account=gisr109364 scripts/slurm/fixedcams_day1.slurm
#   sbatch --account=gisr109364 --export=ALL,CAMS="Reading" scripts/slurm/fixedcams_day1.slurm
#
# The account is per user (`accinfo` shows yours). Other days:
# docs/ingest_all_days.md. Day-1 runbook: docs/fixedcams_reingest.md

#SBATCH --job-name=castle-fixedcams-day1
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=360G
#SBATCH --gres=gpu:3
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

if [ -n "${DAY:-}" ] && [ "${DAY}" != "1" ]; then
  echo "[fixed] ABORT: fixedcams_day1.slurm is day 1 only (got DAY=${DAY});"
  echo "[fixed]        use scripts/slurm/ingest_day.slurm --export=ALL,DAY=${DAY}"
  exit 1
fi
# Under sbatch, $0 is a spool copy, so locate the repo via the submit dir.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
DAY=1 exec bash scripts/slurm/ingest_day.slurm
