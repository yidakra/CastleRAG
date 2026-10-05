#!/usr/bin/env python3
"""Re-create a day's sampled frames from the raw video.

Frames live only on Snellius scratch (cleared every 14 days); the HF backup
skips them because the free plan has 100 GB of private storage. They are cheap
to rebuild: every clip record keeps ``source_video_path``, ``start_seconds``,
``end_seconds`` and the exact ``sampled_frame_paths``, and
``preprocess.media.extract_frames_1fps`` names frames deterministically
(``-ss <start>``, ``fps=1``, ``%04d.jpg``). For each clip with a missing
frame, this extracts the clip at 1 fps into a temp dir and moves only the
listed frames into place, so a thinned day comes back thinned.

Needs the day's raw video (download it again with download_castle.slurm) and
the chunk JSONLs (restored from the HF backup). CPU only. Dry run by default:

    python scripts/regen_frames.py --config configs/snellius_fixedcams.yaml --day 1
    python scripts/regen_frames.py --config configs/snellius_fixedcams.yaml \
        --day 1 --apply --workers 32

Re-running skips clips whose frames all exist.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from castlerag.config import load_config
from castlerag.preprocess.media import extract_frames_1fps


def iter_missing(chunks_day: Path) -> Iterator[dict]:
    """Yield clip rows from ``chunks_day`` that list at least one missing frame."""
    for clips_file in sorted(chunks_day.rglob("clips.jsonl")):
        for line in clips_file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            paths = row.get("sampled_frame_paths") or []
            if any(not Path(p).exists() for p in paths):
                yield row


def regen_clip(row: dict, video_root: Optional[Path] = None) -> int:
    """Rebuild the listed frames of one clip; return how many were written."""
    video = Path(row["source_video_path"])
    if video_root is not None:
        # The chunks may name an old dataset root; keep the part from dayN/.
        parts = video.parts
        tail = next((i for i, p in enumerate(parts) if p.startswith("day")), None)
        if tail is None:
            raise ValueError(f"no dayN/ in {video}")
        video = video_root.joinpath(*parts[tail:])
    if not video.exists():
        raise FileNotFoundError(f"raw video missing: {video}")
    wanted = [Path(p) for p in row["sampled_frame_paths"]]
    out_dir = wanted[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir.parent, prefix=".regen_") as tmp:
        extract_frames_1fps(
            video, Path(tmp), float(row["start_seconds"]), float(row["end_seconds"])
        )
        written = 0
        for dst in wanted:
            if dst.exists():
                continue
            src = Path(tmp) / dst.name
            if not src.exists():
                raise FileNotFoundError(
                    f"ffmpeg did not produce {dst.name} for {video}"
                )
            shutil.move(str(src), str(dst))
            written += 1
    return written


def regen_day(
    chunks_day: Path,
    apply: bool,
    workers: int = 8,
    video_root: Optional[Path] = None,
) -> Dict[str, int]:
    """Rebuild every missing frame under ``chunks_day``; return counters."""
    rows = list(iter_missing(chunks_day))
    stats = {
        "clips": len(rows),
        "frames": sum(
            1 for r in rows for p in r["sampled_frame_paths"] if not Path(p).exists()
        ),
        "written": 0,
        "failed": 0,
    }
    if not apply:
        return stats
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(regen_clip, r, video_root): r["clip_id"] for r in rows}
        for fut in as_completed(futures):
            try:
                stats["written"] += fut.result()
            except Exception as exc:  # report every failure, keep going
                stats["failed"] += 1
                errors.append(f"{futures[fut]}: {exc}")
    for err in errors[:20]:
        print(f"FAILED {err}", file=sys.stderr)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--day", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--apply", action="store_true", help="extract the frames")
    ap.add_argument("--workers", type=int, default=8, help="parallel ffmpeg runs")
    ap.add_argument(
        "--video-root",
        type=Path,
        help="dir holding dayN/ if the video moved since ingest (default: as recorded)",
    )
    args = ap.parse_args()
    if not args.config.is_file():
        ap.error(f"--config {args.config} does not exist")
    cfg = load_config(override_path=args.config)
    chunks_day = Path(cfg.preprocessing.chunks_dir) / f"day{args.day}"
    if not chunks_day.is_dir():
        print(f"ABORT: no chunks at {chunks_day}", file=sys.stderr)
        return 2
    stats = regen_day(chunks_day, args.apply, args.workers, args.video_root)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(
        f"[{mode}] day{args.day}: {stats['clips']} clips with missing frames, "
        f"{stats['frames']} frames missing, {stats['written']} written, "
        f"{stats['failed']} clips failed"
    )
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
