"""ffmpeg-based subclip extraction and 1 fps frame sampling.

Preservation rule (SPEC §2.3):
  - keep source resolution (3840x2160) on disk
  - resize only at model-input time (never here)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

FFMPEG_TIMEOUT_SECONDS = 120


def get_video_duration(source_path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=FFMPEG_TIMEOUT_SECONDS,
    )
    return float(result.stdout.strip())


def extract_frames_1fps(
    source_path: Path,
    out_dir: Path,
    start_seconds: float,
    end_seconds: float,
    fps: int = 1,
) -> List[Path]:
    """Extract JPEG frames at `fps` into out_dir, returning sorted frame paths.

    Uses ffmpeg via subprocess.  Preserves source resolution — no -vf scale.
    Frames are named %04d.jpg (1-indexed by ffmpeg).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()
    duration = end_seconds - start_seconds
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_seconds),
            "-i",
            str(source_path),
            "-t",
            str(duration),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            str(out_dir / "%04d.jpg"),
        ],
        capture_output=True,
        check=True,
        timeout=FFMPEG_TIMEOUT_SECONDS,
    )
    return sorted(out_dir.glob("*.jpg"))


def extract_subclip(
    source_path: Path,
    out_path: Path,
    start_seconds: float,
    end_seconds: float,
) -> Path:
    """Extract a 30-second MP4 subclip with audio, returning out_path.

    Uses accurate seeking and resets timestamps so the derived subclip aligns
    with transcript and frame metadata. This re-encodes the clip instead of
    stream-copying because `-c copy` with pre-input `-ss` is not frame-accurate
    for non-keyframe boundaries.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_seconds - start_seconds
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-ss",
            str(start_seconds),
            "-t",
            str(duration),
            "-reset_timestamps",
            "1",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            str(out_path),
        ],
        capture_output=True,
        check=True,
        timeout=FFMPEG_TIMEOUT_SECONDS,
    )
    return out_path


def is_placeholder_frame(frame_path: Path) -> bool:
    """Return True if the frame matches the CASTLE test-card placeholder.

    The CASTLE test card is a low-variance static card (near-uniform color or
    simple test pattern).  We use grayscale standard deviation < 8 as the
    heuristic; real scene frames consistently exceed 20.  This threshold can
    be tightened once the exact test-card image is available from the dataset.
    """
    import numpy as np
    from PIL import Image

    with Image.open(frame_path) as img:
        arr = np.array(img.convert("L"), dtype=np.float32)
    return float(arr.std()) < 8.0
