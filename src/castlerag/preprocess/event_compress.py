"""Compression of 4 adjacent clip notes into a searchable event summary.

One event-summary block covers 2 minutes (4 × 30 s clips).
The compression model must be a local open-weight summarizer (no hosted API).
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

from castlerag.preprocess.caption_ocr import _vllm_chat
from castlerag.schemas import ClipRecord, EventSummaryRecord


def _event_summary_id(clip_ids: List[str]) -> str:
    """Deterministic ID from member clip IDs."""
    key = "|".join(sorted(clip_ids))
    return "evt_" + hashlib.sha1(key.encode()).hexdigest()[:16]


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

    clips_sorted = sorted(clips, key=lambda c: c.absolute_start)
    first = clips_sorted[0]

    # Validate all clips share the same identity fields
    for c in clips_sorted[1:]:
        if (c.day, c.camera_id, c.camera_type, c.participant_id, c.room) != (
            first.day,
            first.camera_id,
            first.camera_type,
            first.participant_id,
            first.room,
        ):
            raise ValueError(
                f"All clips must share the same (day, camera_id, camera_type, "
                f"participant_id, room); mismatch on {c.clip_id!r}"
            )

    # Validate strictly adjacent, non-overlapping order
    for prev, curr in zip(clips_sorted, clips_sorted[1:]):
        if curr.absolute_start != prev.absolute_end:
            raise ValueError(
                f"Clips must be adjacent; {prev.clip_id!r} ends at "
                f"{prev.absolute_end} but {curr.clip_id!r} starts at "
                f"{curr.absolute_start}"
            )

    member_clip_ids = [c.clip_id for c in clips_sorted]
    abs_start = clips_sorted[0].absolute_start
    abs_end = clips_sorted[-1].absolute_end
    aggregated_ocr = " ".join(c.ocr_text for c in clips_sorted if c.ocr_text) or None

    event_summary_text: Optional[str] = None
    if vllm_base_url:
        clip_notes = "\n".join(
            f"[{i + 1}] {c.clip_caption or c.transcript_text or '(no content)'}"
            for i, c in enumerate(clips_sorted)
        )
        prompt = (
            "You are summarising a 2-minute egocentric video segment from "
            "the CASTLE dataset.\n\n"
            f"Clip notes:\n{clip_notes}\n\n"
            "Write a 3–5 sentence event summary describing what happened during this "
            "2-minute period. Focus on people, objects, actions, and locations."
        )
        event_summary_text = _vllm_chat(
            vllm_base_url,
            model_name,
            [{"role": "user", "content": prompt}],
            max_tokens=256,
        )

    evt_id = _event_summary_id(member_clip_ids)
    return EventSummaryRecord(
        event_summary_id=evt_id,
        day=first.day,
        camera_id=first.camera_id,
        camera_type=first.camera_type,
        participant_id=first.participant_id,
        room=first.room,
        absolute_start=abs_start,
        absolute_end=abs_end,
        member_clip_ids=member_clip_ids,
        event_summary=event_summary_text,
        aggregated_ocr_text=aggregated_ocr,
        version=version,
    )
