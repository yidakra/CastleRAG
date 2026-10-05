#!/bin/bash
# Copy the expensive CastleRAG artifacts from Snellius scratch to this machine,
# as a second backup next to the private Hugging Face repo. Run locally:
#   DAYS="1 2" scripts/pull_artifacts.sh            # chunks, caches, lexical indexes
#   DAYS="1" FRAMES=1 scripts/pull_artifacts.sh     # plus thinned frames (~150 GB/day)
# Optional: HOST (ssh alias, default snellius), DEST (default ~/CastleRAG-artifacts).
set -euo pipefail
DAYS="${DAYS:?set DAYS, e.g. DAYS=\"1 2\"}"
HOST="${HOST:-snellius}"
DEST="${DEST:-$HOME/CastleRAG-artifacts}"
FRAMES="${FRAMES:-0}"
# Resolve the remote login here: rsync won't expand $USER inside a remote path.
RUSER="${RUSER:-$(ssh "$HOST" 'echo "$USER"')}"
[ -n "$RUSER" ] || { echo "could not resolve the remote user on $HOST"; exit 1; }
REMOTE="/scratch-shared/$RUSER/castle_derived"
mkdir -p "$DEST/chunks" "$DEST/embeddings" "$DEST/frames_1fps"
for d in $DAYS; do
  case "$d" in 1|2|3|4) ;; *) echo "bad day '$d'"; exit 1;; esac
  rsync -a --partial "$HOST:$REMOTE/chunks/day$d" "$DEST/chunks/"
  rsync -a --partial --include="*_day$d.npz" --include="manifest_day$d.json" \
        --exclude="*" "$HOST:$REMOTE/embeddings/" "$DEST/embeddings/"
  if [ "$FRAMES" = 1 ]; then
    rsync -a --partial "$HOST:$REMOTE/frames_1fps/day$d" "$DEST/frames_1fps/"
  fi
done
rsync -a --include="transcripts.pkl" --include="visual_text.json" --exclude="*" \
      "$HOST:$REMOTE/embeddings/" "$DEST/embeddings/"
du -sh "$DEST"/*
