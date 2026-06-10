"""Qdrant payload filter builders for server-side filtering.

Time overlap rule (SPEC §4.5):
  retrieve points where absolute_end >= query_start
                    AND absolute_start <= query_end
  Both sides must use UTC Unix-millisecond integers.
"""

from __future__ import annotations

from typing import Any, Optional


def build_filter(
    day: Optional[str] = None,
    camera_id: Optional[str] = None,
    camera_type: Optional[str] = None,
    participant_id: Optional[str] = None,
    room: Optional[str] = None,
    modality: Optional[str] = None,
    source_type: Optional[str] = None,
    time_range_start_ms: Optional[int] = None,
    time_range_end_ms: Optional[int] = None,
    has_speech: Optional[bool] = None,
) -> Any:
    """Return a qdrant_client Filter object for the given constraints.

    Omitted parameters are not added to the filter (no restriction).
    """
    from qdrant_client.http import models as qm

    conditions = []

    if day is not None:
        conditions.append(qm.FieldCondition(key="day", match=qm.MatchValue(value=day)))
    if camera_id is not None:
        conditions.append(
            qm.FieldCondition(key="camera_id", match=qm.MatchValue(value=camera_id))
        )
    if camera_type is not None:
        conditions.append(
            qm.FieldCondition(key="camera_type", match=qm.MatchValue(value=camera_type))
        )
    if participant_id is not None:
        conditions.append(
            qm.FieldCondition(
                key="participant_id", match=qm.MatchValue(value=participant_id)
            )
        )
    if room is not None:
        conditions.append(
            qm.FieldCondition(key="room", match=qm.MatchValue(value=room))
        )
    if modality is not None:
        conditions.append(
            qm.FieldCondition(key="modality", match=qm.MatchValue(value=modality))
        )
    if source_type is not None:
        conditions.append(
            qm.FieldCondition(key="source_type", match=qm.MatchValue(value=source_type))
        )
    if has_speech is not None:
        conditions.append(
            qm.FieldCondition(key="has_speech", match=qm.MatchValue(value=has_speech))
        )

    # Time overlap: absolute_end >= query_start AND absolute_start <= query_end
    if time_range_start_ms is not None:
        conditions.append(
            qm.FieldCondition(
                key="absolute_end",
                range=qm.Range(gte=time_range_start_ms),
            )
        )
    if time_range_end_ms is not None:
        conditions.append(
            qm.FieldCondition(
                key="absolute_start",
                range=qm.Range(lte=time_range_end_ms),
            )
        )

    if (
        time_range_start_ms is not None
        and time_range_end_ms is not None
        and time_range_start_ms > time_range_end_ms
    ):
        raise ValueError(
            f"time_range_start_ms ({time_range_start_ms}) must be "
            f"<= time_range_end_ms ({time_range_end_ms})"
        )

    if not conditions:
        return None
    return qm.Filter(must=conditions)
