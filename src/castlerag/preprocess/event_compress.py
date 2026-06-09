"""Compression of 4 adjacent clip notes into a searchable event summary.

One event-summary block covers 2 minutes (4 × 30 s clips).
The compression model must be a local open-weight summarizer (no hosted API).
"""
from __future__ import annotations

from typing import List, Optional

from castlerag.schemas import ClipRecord, EventSummaryRecord


def compress_clips_to_event(
    clips: List[ClipRecord],
    model_name: str,
    vllm_base_url: Optional[str] = None,
    version: str = "0.1.0",
) -> EventSummaryRecord:
    """Generate an EventSummaryRecord from exactly 4 adjacent ClipRecords.

    The output event_summary is the primary text artifact used for dense
    retrieval over long videos (MARS pattern, SPEC §2.7).
    """
    if len(clips) != 4:
        raise ValueError(f"Expected 4 clips, got {len(clips)}")
    raise NotImplementedError("Implemented in issue #4")
