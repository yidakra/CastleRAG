#!/bin/bash
# archive_bugb.sh — preserve the day-1 ingest's DERIVED artifacts so they survive
# the /scratch-shared auto-purge (~14 days). Archives only what is expensive to
# regenerate (the ~12 h Bug B fixed-camera ingest): the chunk records, the
# embedding caches, the BM25 index, and a tar of the Qdrant collection storage.
# Frames (~330 G, re-extractable from raw via ffmpeg) and raw video
# (re-downloadable) are intentionally skipped.
#
# Run on the LOGIN node AFTER the ingest finishes, while NO job is using the
# Qdrant index (so the storage tar is consistent).
#
#   bash scripts/archive_bugb.sh [DEST_DIR]
#   # default DEST_DIR=$HOME/castle_archives/bugb_day1_<YYYYmmdd>
#
# Restore instructions are written into <DEST>/RESTORE.md.

set -euo pipefail

SC="/scratch-shared/${USER}"
DERIVED="${SC}/castle_derived"
EMB="${DERIVED}/embeddings"
QSTORE="${SC}/qdrant_storage/storage"
COLL="castle_multimodal_v1"
STAMP="$(date +%Y%m%d)"
DEST="${1:-${HOME}/castle_archives/bugb_day1_${STAMP}}"

echo "[archive] source scratch : ${SC}"
echo "[archive] destination    : ${DEST}"

# --- sanity ---
[ -d "${DERIVED}/chunks" ] || { echo "[archive] ABORT: ${DERIVED}/chunks missing"; exit 2; }
[ -d "${EMB}" ]           || { echo "[archive] ABORT: ${EMB} missing"; exit 2; }
[ -d "${QSTORE}" ]        || { echo "[archive] ABORT: ${QSTORE} missing"; exit 2; }
if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -qiE "bugb|ui-live|smoke|castle"; then
  echo "[archive] WARNING: a castle job appears to be running — the Qdrant storage"
  echo "[archive]          tar may be inconsistent. Prefer archiving when idle."
fi

mkdir -p "${DEST}"

# --- preflight: fail before writing if DEST can't hold the source (+1 GB) ---
# Sized from the uncompressed sources (chunks compress, so this over-estimates,
# which is the safe direction). Aborts up front instead of dying mid-tar and
# leaving corrupt partials on a near-full home quota.
NEED_KB=$(du -sk "${DERIVED}/chunks" "${EMB}" "${QSTORE}" 2>/dev/null | awk '{s+=$1} END{print s+0}')
NEED_KB=$(( NEED_KB + 1048576 ))                       # +1 GB margin
AVAIL_KB=$(df -Pk "${DEST}" | awk 'NR==2 {print $4}')
echo "[archive] space: need ~$(( NEED_KB / 1048576 )) GB, free $(( ${AVAIL_KB:-0} / 1048576 )) GB at $(dirname "${DEST}")"
if [ "${AVAIL_KB:-0}" -lt "${NEED_KB}" ]; then
  echo "[archive] ABORT: insufficient space (need ~$(( NEED_KB / 1048576 )) GB). Pass a DEST_DIR on a larger filesystem."
  exit 2
fi

# --- 1/3: chunk records (JSON; compress well) ---
echo "[archive] 1/3 chunks (day1) ..."
tar -C "${DERIVED}" -czf "${DEST}/chunks_day1.tar.gz" chunks

# --- 2/3: embedding caches + BM25 index (binary; no double-compression) ---
echo "[archive] 2/3 embedding caches + BM25 index ..."
(
  cd "${EMB}"
  files=()
  for f in *_day1.npz manifest_day1.json transcripts.pkl query_cache.npz; do
    [ -e "${f}" ] && files+=("${f}")
  done
  [ "${#files[@]}" -gt 0 ] || { echo "[archive] ABORT: no embedding caches in ${EMB}"; exit 2; }
  echo "[archive]     including: ${files[*]}"
  tar -cf "${DEST}/embeddings_day1.tar" "${files[@]}"
)

# --- 3/3: Qdrant collection storage (binary) ---
echo "[archive] 3/3 Qdrant storage (${COLL}) ..."
tar -C "$(dirname "${QSTORE}")" -cf "${DEST}/qdrant_storage.tar" "$(basename "${QSTORE}")"

# --- integrity + restore notes ---
( cd "${DEST}" && sha256sum ./*.tar ./*.tar.gz > SHA256SUMS )
cat > "${DEST}/RESTORE.md" <<EOF
# Day-1 ingest archive (${STAMP})

Derived artifacts for the day-1 index, including the fixed room cameras
(Kitchen/Living1/Living2/Meeting/Reading) from the Bug B ingest. Preserved
against the /scratch-shared purge. Frames and raw video are NOT included
(regenerable from raw / re-downloadable).

Contents:
- chunks_day1.tar.gz   — preprocessed chunk records (captions/OCR/events)
- embeddings_day1.tar  — OmniEmbed caches (*_day1.npz), manifest, BM25 index
- qdrant_storage.tar   — Qdrant storage for collection ${COLL}

## Restore to scratch
Run these from THIS archive directory (where the tars and SHA256SUMS live):

    cd <this archive directory>
    sha256sum -c SHA256SUMS                 # verify integrity first
    SC="/scratch-shared/\$USER"
    mkdir -p "\$SC/castle_derived/embeddings" "\$SC/qdrant_storage"
    tar -C "\$SC/castle_derived"            -xzf chunks_day1.tar.gz
    tar -C "\$SC/castle_derived/embeddings" -xf  embeddings_day1.tar
    tar -C "\$SC/qdrant_storage"            -xf  qdrant_storage.tar

After restore, ui_live.slurm / smoke / eval boot Qdrant against the restored
storage (collection ${COLL}) with the fixed cameras already indexed — no
re-ingest needed.
EOF

echo "[archive] === contents ==="
du -sh "${DEST}"/* 2>/dev/null || true
echo "[archive] TOTAL: $(du -sh "${DEST}" | cut -f1)   (check home quota with 'myquota')"
echo "[archive] DONE -> ${DEST}"
