"""ffmpeg-based subclip extraction and 1 fps frame sampling.

Preservation rule (SPEC §2.3):
  - keep source resolution (3840x2160) on disk
  - resize only at model-input time (never here)
"""
from __future__ import annotations

from pathlib import Path
from typing import List


def extract_frames_1fps(
    source_path: Path,
    out_dir: Path,
    start_seconds: float,
    end_seconds: float,
    fps: int = 1,
) -> List[Path]:
    """Extract JPEG frames at `fps` into out_dir, returning sorted frame paths.

    Uses ffmpeg via subprocess.  Preserves source resolution.
    """
    raise NotImplementedError("Implemented in issue #4")


def extract_subclip(
    source_path: Path,
    out_path: Path,
    start_seconds: float,
    end_seconds: float,
) -> Path:
    """Extract a 30-second MP4 subclip with audio, returning out_path.

    Keeps original frame rate for archival traceability.
    """
    raise NotImplementedError("Implemented in issue #4")


def get_video_duration(source_path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    raise NotImplementedError("Implemented in issue #4")


def is_placeholder_frame(frame_path: Path) -> bool:
    """Return True if the frame matches the CASTLE test-card placeholder."""
    raise NotImplementedError("Implemented in issue #4")
