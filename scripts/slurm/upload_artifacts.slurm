#!/bin/bash
#SBATCH --job-name=castle-upload
#SBATCH --account=gisr109364
#SBATCH --partition=staging
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/castle-upload_%j.out
# Back up one ingested day to a PRIVATE Hugging Face dataset repo, so losing
# Snellius scratch (14-day purge) or the account doesn't mean re-running the
# captioning/embedding (~15k SBU per day). Usage, after the day's ingest and
# scripts/thin_frames.py --apply:
#   sbatch --export=ALL,DAY=1 scripts/slurm/upload_artifacts.slurm
# Optional: REPO=<owner/name> (default yidakra/castlerag-artifacts), FRAMES=0
# to skip frames, CONF=<config> (default configs/snellius_fixedcams.yaml).
#
# CASTLE's terms forbid distributing the dataset or derivative works, so the
# job refuses to upload unless the repo exists and is private. Create it once
# (private) on huggingface.co and run `hf auth login` (write token) on a login
# node first; the job never handles the token itself.
set -uo pipefail
DAY="${DAY:?set DAY=1..4}"
case "$DAY" in 1|2|3|4) ;; *) echo "ABORT: DAY must be 1-4"; exit 1;; esac
REPO="${REPO:-yidakra/castlerag-artifacts}"
FRAMES="${FRAMES:-1}"
CONF="${CONF:-configs/snellius_fixedcams.yaml}"
HF_VENV="${HF_VENV:-$HOME/hfdl_venv}"
SC="/scratch-shared/$USER"
DERIVED="$SC/castle_derived"
CHUNKS="$DERIVED/chunks"
EMB="$DERIVED/embeddings"
FRAMES_DIR="$DERIVED/frames_1fps"
STAGE="$SC/hf_stage"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p logs "$STAGE/day${DAY}/frames" "$STAGE/common"
echo "start $(date -Iseconds): day${DAY} -> ${REPO} (frames=${FRAMES}, conf=${CONF})"

# --- refuse anything but an existing private repo -------------------------
if ! "$HF_VENV/bin/python" - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
api = HfApi()
try:
    who = api.whoami()["name"]
except Exception as exc:
    sys.exit(f"ABORT: not logged in to Hugging Face ({exc}); run `hf auth login` on a login node")
info = api.repo_info(sys.argv[1], repo_type="dataset")
if not info.private:
    sys.exit(f"ABORT: {sys.argv[1]} is PUBLIC; CASTLE derivatives may only go to a private repo")
print(f"ok: {sys.argv[1]} is private, logged in as {who}")
PY
then exit 2; fi

# --- stage --------------------------------------------------------------------
[ -d "$CHUNKS/day${DAY}" ] || { echo "ABORT: no chunks for day${DAY} in $CHUNKS"; exit 2; }
ls "$EMB"/*_day"${DAY}".npz >/dev/null 2>&1 || { echo "ABORT: no day${DAY} embedding caches in $EMB"; exit 2; }
[ -s "$EMB/manifest_day${DAY}.json" ] || { echo "ABORT: manifest_day${DAY}.json missing or empty"; exit 2; }
tar -C "$CHUNKS" -czf "$STAGE/day${DAY}/chunks_day${DAY}.tar.gz" "day${DAY}" || exit 2
(cd "$EMB" && tar -cf "$STAGE/day${DAY}/embeddings_day${DAY}.tar" ./*_day"${DAY}".npz "manifest_day${DAY}.json") || exit 2
# Lexical indexes are rebuilt from all days' chunks; upload the current ones.
for f in transcripts.pkl visual_text.json; do
  [ -e "$EMB/$f" ] && cp -p "$EMB/$f" "$STAGE/common/$f"
done
cp -p "$CONF" "$STAGE/common/$(basename "$CONF")"
if [ "$FRAMES" = 1 ]; then
  [ -d "$FRAMES_DIR/day${DAY}" ] || { echo "ABORT: no frames for day${DAY} in $FRAMES_DIR (FRAMES=0 to skip)"; exit 2; }
  # One tar per camera-hour keeps the repo's file count low (a day of thinned
  # frames is ~170k JPEGs).
  for camdir in "$FRAMES_DIR/day${DAY}"/*/; do
    cam="$(basename "$camdir")"
    for hourdir in "$camdir"*/; do
      hh="$(basename "$hourdir")"
      tar -C "$FRAMES_DIR" -cf "$STAGE/day${DAY}/frames/${cam}_${hh}.tar" "day${DAY}/${cam}/${hh}" || exit 2
    done
  done
fi
(cd "$STAGE/day${DAY}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
du -sh "$STAGE/day${DAY}" "$STAGE/common"

# --- upload (resumable: files already on the Hub are skipped) -----------------
ok=0
for attempt in 1 2 3; do
  if "$HF_VENV/bin/python" - "$REPO" "$STAGE" "day${DAY}" <<'PY'
# huggingface_hub 2.x has no upload_large_folder: upload in batches of files
# not yet on the Hub, one commit per batch, so a re-run resumes where it left.
import sys
from pathlib import Path
from huggingface_hub import HfApi
repo, stage, day = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
api = HfApi()
remote = set(api.list_repo_files(repo, repo_type="dataset"))
todo = sorted(
    p.relative_to(stage).as_posix()
    for sub in (day, "common")
    for p in (stage / sub).rglob("*")
    if p.is_file()
)
# Shared files change as days are added; always refresh them.
pending = [f for f in todo if f.startswith("common/") or f not in remote]
print(f"{len(todo)} staged, {len(pending)} to upload")
BATCH = 25
for i in range(0, len(pending), BATCH):
    batch = pending[i:i + BATCH]
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(stage),
                      allow_patterns=batch,
                      commit_message=f"{day}: upload {i + 1}-{i + len(batch)} of {len(pending)}")
    print(f"uploaded {i + len(batch)}/{len(pending)}", flush=True)
PY
  then ok=1; break; fi
  echo "upload attempt $attempt failed $(date -Iseconds); retrying in 300s"; sleep 300
done
echo "end $(date -Iseconds) ok=$ok"
# Free the staging copy once it is safely on the Hub.
[ "$ok" = 1 ] && rm -rf "$STAGE/day${DAY}/frames"
[ "$ok" = 1 ]
