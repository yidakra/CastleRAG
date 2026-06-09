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

    Windows are non-overlapping (stride == window size by default).  A trailing
    window shorter than 1 second is discarded.  Placeholder detection is deferred
    to media.py (requires frame access).
    """
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be > 0")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be > 0")
    start = 0.0
    clip_index = 0
    while start < duration_seconds:
        end = min(start + clip_seconds, duration_seconds)
        if end - start < 1.0:
            break
        yield VideoWindow(
            camera_id=camera_id,
            day=day,
            hour=hour,
            clip_index=clip_index,
            start_seconds=start,
            end_seconds=end,
            source_video_path=video_path,
        )
        start += stride_seconds
        clip_index += 1


def mark_placeholder_windows(
    windows: List[VideoWindow],
    frame_dir: Path,
    placeholder_threshold: float = 0.80,
) -> List[VideoWindow]:
    """Return windows with is_placeholder set based on per-frame checks.

    A window is marked placeholder when the fraction of frames that match
    the CASTLE test-card exceeds placeholder_threshold (default 0.80).

    frame_dir must contain per-clip sub-directories named by clip_index
    (e.g. frame_dir/0/, frame_dir/1/, ...).
    """
    if not (0.0 <= placeholder_threshold <= 1.0):
        raise ValueError("placeholder_threshold must be between 0.0 and 1.0")
    from castlerag.preprocess.media import is_placeholder_frame

    result: List[VideoWindow] = []
    for w in windows:
        clip_dir = frame_dir / str(w.clip_index)
        frames = sorted(clip_dir.glob("*.jpg")) if clip_dir.exists() else []
        if not frames:
            result.append(w)
            continue
        n_placeholder = sum(1 for f in frames if is_placeholder_frame(f))
        frac = n_placeholder / len(frames)
        result.append(VideoWindow(
            camera_id=w.camera_id,
            day=w.day,
            hour=w.hour,
            clip_index=w.clip_index,
            start_seconds=w.start_seconds,
            end_seconds=w.end_seconds,
            source_video_path=w.source_video_path,
            is_placeholder=frac > placeholder_threshold,
        ))
    return result
