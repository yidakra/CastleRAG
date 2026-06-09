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
        segments.append(TranscriptSegment(
            start=float(ts[0]),
            end=float(ts[1]),
            text=chunk.get("text", "").strip(),
        ))
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
    """Merge adjacent ASR segments into utterance windows (≤15 s, ≤96 token-equiv chars).

    Returns transcript windows with absolute UTC millisecond timestamps.
    """
    raise NotImplementedError("Implemented in issue #4 (transcript normalization)")
