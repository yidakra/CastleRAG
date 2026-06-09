"""Sliding-window creation for main video chunks.

Policy (fixed by spec):
  window_size = 30 s
  stride      = 30 s  (no overlap)
  fps         = 1 (for derived retrieval frames)

Placeholder detection:
  skip windows where >80% of sampled frames match the CASTLE test-card.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List


@dataclass
class VideoWindow:
    camera_id: str
    day: str
    hour: int
    clip_index: int       # 0-based within the hour
    start_seconds: float
    end_seconds: float
    source_video_path: Path
    is_placeholder: bool = False


def iter_windows(
    video_path: Path,
    camera_id: str,
    day: str,
    hour: int,
    duration_seconds: float,
    clip_seconds: int = 30,
    stride_seconds: int = 30,
) -> Iterator[VideoWindow]:
    """Yield VideoWindow records for a single hour video.

    Does not touch the file system beyond reading the provided duration.
    Placeholder detection is deferred to media.py (requires frame access).
    """
    raise NotImplementedError("Implemented in issue #4")
