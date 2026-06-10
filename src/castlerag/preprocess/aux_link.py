"""Link AuxRecords to overlapping ClipRecords and EventSummaryRecords.

Overlap criterion (standard interval overlap):
    clip.absolute_start < aux.absolute_end and clip.absolute_end > aux.absolute_start

Records without a reliable timestamp (has_reliable_timestamp=False) are left
unchanged — their linked_* lists remain empty so that retrieval can safely
filter them out.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from castlerag.schemas import AuxRecord, ClipRecord, EventSummaryRecord

log = logging.getLogger(__name__)


def _load_clip_records(path: Path) -> List[ClipRecord]:
    records: List[ClipRecord] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(ClipRecord.model_validate_json(line))
    return records


def _load_event_summary_records(path: Path) -> List[EventSummaryRecord]:
    records: List[EventSummaryRecord] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(EventSummaryRecord.model_validate_json(line))
    return records


def link_aux_records(
    aux_records: List[AuxRecord],
    clips_jsonl: Optional[Union[str, Path]] = None,
    events_jsonl: Optional[Union[str, Path]] = None,
) -> List[AuxRecord]:
    """Return a new list of AuxRecords with linked_main_clip_ids and
    linked_event_summary_ids populated by temporal overlap.

    Parameters
    ----------
    aux_records:
        Input auxiliary records produced by the preprocess/auxiliary.py
        iterators.
    clips_jsonl:
        Path to a JSONL file of ClipRecord rows.  If None or the file does not
        exist the clip-linking step is skipped and records are returned
        unchanged for that dimension.
    events_jsonl:
        Path to a JSONL file of EventSummaryRecord rows.  Same semantics as
        clips_jsonl.

    Returns
    -------
    List[AuxRecord]
        New list; original objects are not mutated.
    """
    clips: List[ClipRecord] = []
    events: List[EventSummaryRecord] = []

    if clips_jsonl is not None:
        clips_path = Path(clips_jsonl)
        if clips_path.exists():
            try:
                clips = _load_clip_records(clips_path)
            except Exception as exc:
                log.warning("Could not load clips from %s: %s", clips_path, exc)
        else:
            log.debug("clips_jsonl path does not exist, skipping: %s", clips_path)

    if events_jsonl is not None:
        events_path = Path(events_jsonl)
        if events_path.exists():
            try:
                events = _load_event_summary_records(events_path)
            except Exception as exc:
                log.warning("Could not load events from %s: %s", events_path, exc)
        else:
            log.debug("events_jsonl path does not exist, skipping: %s", events_path)

    linked: List[AuxRecord] = []
    for aux in aux_records:
        if not aux.has_reliable_timestamp:
            linked.append(
                aux.model_copy(
                    update={
                        "linked_main_clip_ids": [],
                        "linked_event_summary_ids": [],
                    }
                )
            )
            continue

        clip_ids = [
            c.clip_id
            for c in clips
            if c.absolute_start < aux.absolute_end
            and c.absolute_end > aux.absolute_start
        ]
        event_ids = [
            e.event_summary_id
            for e in events
            if e.absolute_start < aux.absolute_end
            and e.absolute_end > aux.absolute_start
        ]

        linked.append(
            aux.model_copy(
                update={
                    "linked_main_clip_ids": clip_ids,
                    "linked_event_summary_ids": event_ids,
                }
            )
        )

    return linked
