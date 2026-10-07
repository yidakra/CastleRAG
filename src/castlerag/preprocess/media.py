"""ffmpeg-based subclip extraction and 1 fps frame sampling.

Preservation rule (SPEC §2.3):
  - keep source resolution (3840x2160) on disk
  - resize only at model-input time (never here)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

FFMPEG_TIMEOUT_SECONDS = 120
# A 30 s clip normally takes ~10 s, but reads from the shared scratch
# filesystem can stall for minutes; the first day-1 ingest on Snellius lost all
# ten workers to one such stall. Give frame extraction more time and retry a
# clip that still times out, rather than failing the whole camera.
FRAME_TIMEOUT_SECONDS = 600
FRAME_ATTEMPTS = 3


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
    cmd = [
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
    ]
    for attempt in range(1, FRAME_ATTEMPTS + 1):
        try:
            subprocess.run(
                cmd, capture_output=True, check=True, timeout=FRAME_TIMEOUT_SECONDS
            )
            break
        except subprocess.TimeoutExpired:
            if attempt == FRAME_ATTEMPTS:
                raise
            print(
                f"ffmpeg timed out on {source_path} @ {start_seconds}s "
                f"(attempt {attempt}/{FRAME_ATTEMPTS}); retrying",
                file=sys.stderr,
            )
            for partial in out_dir.glob("*.jpg"):
                partial.unlink()
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


def frame_stats(frame_path: Path) -> "tuple[float, float]":
    """Return (grayscale std, flatness) for one frame, from a single decode.

    JPEG frames are decoded at reduced scale (draft mode: 3840x2160 decodes
    straight to 960x540, several times faster than a full decode). Flatness is
    the fraction of horizontally adjacent pixels with exactly equal values.
    Smaller images are measured as they are (never upscaled).
    """
    import numpy as np
    from PIL import Image

    with Image.open(frame_path) as img:
        if img.format == "JPEG" and img.width > 960:
            img.draft("L", (960, 540))
        gray = img.convert("L")
        if gray.width > 960:
            gray = gray.resize((960, round(960 * gray.height / gray.width)))
        arr = np.asarray(gray, dtype=np.int16)
    flat = float((np.diff(arr, axis=1) == 0).mean()) if arr.shape[1] > 1 else 0.0
    return float(arr.std()), flat


# Grayscale std below this is a blank frame (black, lights off); real scenes
# consistently exceed 20. The CASTLE test card is colourful (std ~70).
BLANK_STD = 8.0

# Flatness above this is the CASTLE test card, a synthetic graphic (flat grey
# grid, colour bars). Measured on days 1-3 with draft decoding: card frames
# score ~0.91; real frames have sensor noise and scored median 0.35-0.44, with
# a rare overexposed frame up to 0.81. A clip needs >80 % such frames to be a
# placeholder; on 600 sampled real clips (300 still, 300 normal) none was,
# apart from one near-black covered-lens clip that is blank anyway.
TEST_CARD_FLAT_FRACTION = 0.65


def is_placeholder_frame(frame_path: Path) -> bool:
    """Return True if the frame is blank (near-uniform, e.g. a black frame)."""
    return frame_stats(frame_path)[0] < BLANK_STD


def is_test_card_frame(frame_path: Path) -> bool:
    """Return True if the frame shows the CASTLE test card (a flat graphic).

    Detected positively from flatness rather than from stillness: a fixed
    camera filming an empty room is still too, but it is real footage.
    """
    return frame_stats(frame_path)[1] > TEST_CARD_FLAT_FRACTION


def is_placeholder_or_card(frame_path: Path) -> bool:
    """Blank or test card, from one decode (used per frame by the window rule)."""
    std, flat = frame_stats(frame_path)
    return std < BLANK_STD or flat > TEST_CARD_FLAT_FRACTION
