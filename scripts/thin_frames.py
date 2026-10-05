#!/usr/bin/env python3
"""Thin a day's sampled frames down to the ones read at question time.

Each 30 s clip is sampled at 1 fps (~30 JPEGs), but after captioning only a
few frames are ever read again: the reranker and the generator sample frames
evenly across a clip (``castlerag.frame_encoding.sample_frames_evenly``). This
keeps ``--keep`` evenly spaced frames per clip (default 8, the generator's
per-clip maximum), deletes the rest, and rewrites ``sampled_frame_paths`` in
the day's chunk JSONLs to the kept frames. Readers sample from the frames that
still exist (``available_frames``), so already-indexed payloads keep working
without a re-index.

Run it only after the day's captioning and events are done; both read the
full frame set. Dry run by default:

    python scripts/thin_frames.py --config configs/snellius_fixedcams.yaml --day 1
    python scripts/thin_frames.py --config configs/snellius_fixedcams.yaml \
        --day 1 --apply

Each rewritten JSONL keeps a ``.prethin`` copy of the original.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

from castlerag.config import load_config
from castlerag.frame_encoding import sample_frames_evenly


def plan_clip(paths: List[str], keep: int) -> Tuple[List[str], List[str]]:
    """Return (kept, dropped) frame paths for one clip."""
    kept = sample_frames_evenly(paths, keep)
    kept_set = set(kept)
    return kept, [p for p in paths if p not in kept_set]


def _rewrite_jsonl(path: Path, rows: List[dict]) -> None:
    """Atomically replace ``path`` with ``rows``, keeping a .prethin copy."""
    backup = path.with_name(path.name + ".prethin")
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


def thin_day(chunks_day: Path, keep: int, apply: bool) -> Dict[str, int]:
    """Thin every clip under ``chunks_day``; return counters."""
    stats = {"clips": 0, "kept": 0, "dropped": 0, "dropped_bytes": 0, "files": 0}
    dropped_all: Set[str] = set()
    kept_by_clip: Dict[str, List[str]] = {}
    for clips_file in sorted(chunks_day.rglob("clips.jsonl")):
        lines = clips_file.read_text().splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        changed = False
        for row in rows:
            paths = row.get("sampled_frame_paths") or []
            if len(paths) <= keep:
                stats["kept"] += len(paths)
                continue
            kept, dropped = plan_clip(paths, keep)
            stats["clips"] += 1
            stats["kept"] += len(kept)
            stats["dropped"] += len(dropped)
            for p in dropped:
                try:
                    stats["dropped_bytes"] += Path(p).stat().st_size
                except OSError:
                    pass
            dropped_all.update(dropped)
            kept_by_clip[row.get("clip_id", "")] = kept
            row["sampled_frame_paths"] = kept
            changed = True
        if changed and apply:
            _rewrite_jsonl(clips_file, rows)
            stats["files"] += 1
    # Other chunk files (events, ...) may list the same frames; drop deleted
    # paths from them so no record points at a removed file.
    for other in sorted(chunks_day.rglob("*.jsonl")):
        if other.name == "clips.jsonl":
            continue
        lines = other.read_text().splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        changed = False
        for row in rows:
            paths = row.get("sampled_frame_paths")
            if isinstance(paths, list) and any(p in dropped_all for p in paths):
                row["sampled_frame_paths"] = [p for p in paths if p not in dropped_all]
                changed = True
        if changed and apply:
            _rewrite_jsonl(other, rows)
            stats["files"] += 1
    if apply:
        for p in dropped_all:
            Path(p).unlink(missing_ok=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--day", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument(
        "--keep", type=int, default=8, help="frames kept per clip (default 8)"
    )
    ap.add_argument(
        "--apply", action="store_true", help="delete frames and rewrite chunks"
    )
    args = ap.parse_args()
    if args.keep < 1:
        ap.error("--keep must be >= 1")
    cfg = load_config(override_path=args.config)
    chunks_day = Path(cfg.preprocessing.chunks_dir) / f"day{args.day}"
    if not chunks_day.is_dir():
        print(f"ABORT: no chunks at {chunks_day}", file=sys.stderr)
        return 2
    stats = thin_day(chunks_day, args.keep, args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(
        f"[{mode}] day{args.day}: {stats['clips']} clips thinned, "
        f"{stats['kept']} frames kept, {stats['dropped']} dropped "
        f"({stats['dropped_bytes'] / 1e9:.1f} GB), "
        f"{stats['files']} chunk files rewritten"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
