#!/usr/bin/env python3
"""Recompute ``is_placeholder`` for a day's clips with the current rule.

Days ingested before the test-card detector (#68) were flagged by stillness, so
real footage of an idle fixed-camera room counted as a placeholder and was left
out of event summaries, while some test-card clips were not flagged. This
re-applies ``mark_placeholder_windows``' rule to the frames still on disk
(thinned days keep 8 per clip, enough for the >80 % test) and rewrites the
flag in the day's ``clips.jsonl`` files. Dry run by default:

    python scripts/reflag_placeholders.py \
        --config configs/snellius_fixedcams.yaml --day 1
    python scripts/reflag_placeholders.py --config configs/snellius_fixedcams.yaml \
        --day 1 --apply

Then rebuild events and the index for that day, without recaptioning:

    sbatch --export=ALL,DAY=1,CAMS="<all 15>",SKIP_BASE=1,CAPTION=0,MIN_POINTS=1 \
        scripts/slurm/ingest_day.slurm

Its index step deletes the clip points that are now placeholders and the
events that re-grouping replaced (``prune_stale_points``). Each rewritten
JSONL keeps a ``.preflag`` copy of the original.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from castlerag.config import load_config
from castlerag.frame_encoding import available_frames
from castlerag.preprocess.media import is_placeholder_or_card

PLACEHOLDER_THRESHOLD = 0.80  # same default as mark_placeholder_windows
IO_WORKERS = 16


def clip_is_placeholder(frame_paths: List[str]) -> Optional[bool]:
    """Apply the window rule to the frames on disk; None if none exist."""
    frames = [p for p in available_frames(frame_paths) if Path(p).exists()]
    if not frames:
        return None
    hits = sum(1 for f in frames if is_placeholder_or_card(Path(f)))
    return hits / len(frames) > PLACEHOLDER_THRESHOLD


def _rewrite_jsonl(path: Path, rows: List[dict]) -> None:
    backup = path.with_name(path.name + ".preflag")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def reflag_day(
    chunks_day: Path, apply: bool, workers: int = IO_WORKERS
) -> Dict[str, int]:
    """Recompute the flag for every clip under ``chunks_day``; return counters."""
    stats = {
        "clips": 0,
        "placeholder": 0,
        "newly_flagged": 0,
        "unflagged": 0,
        "no_frames": 0,
        "files": 0,
    }
    files = []
    for clips_file in sorted(chunks_day.rglob("clips.jsonl")):
        lines = clips_file.read_text().splitlines()
        files.append((clips_file, [json.loads(line) for line in lines if line.strip()]))
    rows = [row for _, file_rows in files for row in file_rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        flags = list(
            pool.map(
                lambda r: clip_is_placeholder(r.get("sampled_frame_paths") or []), rows
            )
        )
    new_flag = {id(row): flag for row, flag in zip(rows, flags)}
    for clips_file, file_rows in files:
        changed = False
        for row in file_rows:
            stats["clips"] += 1
            flag = new_flag[id(row)]
            old = bool(row.get("is_placeholder"))
            if flag is None:  # no frames left: keep the old flag
                stats["no_frames"] += 1
                flag = old
            stats["placeholder"] += flag
            if flag != old:
                stats["newly_flagged" if flag else "unflagged"] += 1
                row["is_placeholder"] = flag
                changed = True
        if changed and apply:
            _rewrite_jsonl(clips_file, file_rows)
            stats["files"] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--day", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--apply", action="store_true", help="rewrite clips.jsonl")
    args = ap.parse_args()
    if not args.config.is_file():
        ap.error(f"--config {args.config} does not exist")
    cfg = load_config(override_path=args.config)
    chunks_day = Path(cfg.preprocessing.chunks_dir) / f"day{args.day}"
    if not chunks_day.is_dir():
        print(f"ABORT: no chunks at {chunks_day}", file=sys.stderr)
        return 2
    s = reflag_day(chunks_day, args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(
        f"[{mode}] day{args.day}: {s['clips']} clips, {s['placeholder']} placeholders "
        f"(+{s['newly_flagged']} newly flagged, -{s['unflagged']} unflagged), "
        f"{s['no_frames']} without frames kept as they were, "
        f"{s['files']} chunk files rewritten"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
