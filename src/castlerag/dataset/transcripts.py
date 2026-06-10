"""Transcript JSON parsing and absolute timestamp alignment.

Official format:
  {"chunks": [{"timestamp": [start_s, end_s], "text": "..."}, ...]}

Timestamps are relative to the enclosing hour file (e.g. 08.json → 08:00 base).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from castlerag.schemas import TranscriptSegment, TranscriptWindow

# CASTLE recording epoch: actual recording calendar date is not part of this
# skeleton.  The caller must supply the base_unix_ms for the day+hour.


def load_raw_segments(path: Path) -> List[TranscriptSegment]:
    """Parse an official CASTLE transcript JSON into raw segments."""
    data = json.loads(path.read_text())
    segments: List[TranscriptSegment] = []
    for i, chunk in enumerate(data.get("chunks", [])):
        ts = chunk.get("timestamp", [0, 0])
        if not isinstance(ts, (list, tuple)) or len(ts) < 2:
            raise ValueError(
                f"Malformed timestamp in chunk {i} of {path}: "
                f"expected [start, end], got {ts!r}"
            )
        segments.append(
            TranscriptSegment(
                start=float(ts[0]),
                end=float(ts[1]),
                text=chunk.get("text", "").strip(),
            )
        )
    return segments


def merge_into_windows(
    segments: List[TranscriptSegment],
    base_unix_ms: int,
    camera_id: str,
    camera_type: str,
    participant_id: str | None,
    room: str | None,
    day: str,
    hour: int,
    max_seconds: float = 15.0,
    max_chars: int = 96 * 4,  # approx token→char factor
    version: str = "0.1.0",
) -> List[TranscriptWindow]:
    """Merge adjacent ASR segments into utterance windows.

    Returns transcript windows with absolute UTC millisecond timestamps.
    Each window carries its raw constituent segments for downstream BM25 and
    dense retrieval.  Windows with no speech text are still emitted so that
    silent stretches are represented in the time index.
    """
    windows: List[TranscriptWindow] = []

    bucket_segs: List[TranscriptSegment] = []
    bucket_start: float | None = None
    bucket_end: float = 0.0
    bucket_chars: int = 0

    def _flush() -> None:
        nonlocal bucket_segs, bucket_start, bucket_end, bucket_chars
        if not bucket_segs:
            return
        text = " ".join(s.text for s in bucket_segs if s.text).strip()
        abs_start = base_unix_ms + int(bucket_start * 1000)  # type: ignore[operator]
        abs_end = base_unix_ms + int(bucket_end * 1000)
        if abs_end <= abs_start:
            abs_end = abs_start + 1
        win_id = f"{day}_{camera_id}_{hour:02d}_{int(bucket_start * 1000):010d}"  # type: ignore[operator]
        windows.append(
            TranscriptWindow(
                transcript_window_id=win_id,
                day=day,
                camera_id=camera_id,
                camera_type=camera_type,  # type: ignore[arg-type]
                participant_id=participant_id,
                room=room,
                hour=hour,
                transcript_text=text,
                transcript_segments=list(bucket_segs),
                has_speech=bool(text),
                transcript_char_len=len(text),
                absolute_start=abs_start,
                absolute_end=abs_end,
                version=version,
            )
        )
        bucket_segs = []
        bucket_start = None
        bucket_end = 0.0
        bucket_chars = 0

    for seg in segments:
        if bucket_start is not None:
            span = seg.end - bucket_start
            new_chars = bucket_chars + len(seg.text)
            if span > max_seconds or new_chars > max_chars:
                _flush()
        if bucket_start is None:
            bucket_start = seg.start
        bucket_end = seg.end
        bucket_chars += len(seg.text)
        bucket_segs.append(seg)
        # If this single segment already exceeds limits, emit immediately so
        # we never accumulate further into an already-over-limit window.
        if len(bucket_segs) == 1:
            span = bucket_end - bucket_start
            if span > max_seconds or bucket_chars > max_chars:
                _flush()

    _flush()
    return windows
