"""Tests for src/castlerag/preprocess/aux_link.py and has_reliable_timestamp."""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import patch

from castlerag.preprocess.aux_link import link_aux_records
from castlerag.schemas import AuxRecord, ClipRecord, EventSummaryRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_MS = 1_700_000_000_000  # arbitrary fixed epoch


def _make_aux(
    abs_start: int,
    abs_end: int,
    participant: str = "Allie",
    reliable: bool = True,
) -> AuxRecord:
    return AuxRecord(
        clip_id=f"aux_photo_test_{abs_start}_{abs_end}",
        source_type="aux_photo",
        modality="image",
        day="day1",
        participant_id=participant,
        aux_owner=participant,
        asset_path=f"/data/photo/{participant}/img.jpg",
        summary_text=f"Photo by {participant}",
        absolute_start=abs_start,
        absolute_end=abs_end,
        has_reliable_timestamp=reliable,
    )


def _make_clip(clip_id: str, abs_start: int, abs_end: int) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        parent_source_id="vid_0",
        day="day1",
        hour=8,
        camera_id="Allie",
        camera_type="ego",
        participant_id="Allie",
        start_seconds=0.0,
        end_seconds=30.0,
        absolute_start=abs_start,
        absolute_end=abs_end,
        source_video_path="/data/main/day1/Allie/video/08.mp4",
    )


def _make_event(event_id: str, abs_start: int, abs_end: int) -> EventSummaryRecord:
    return EventSummaryRecord(
        event_summary_id=event_id,
        day="day1",
        camera_id="Allie",
        camera_type="ego",
        absolute_start=abs_start,
        absolute_end=abs_end,
        member_clip_ids=["c0", "c1", "c2", "c3"],
    )


def _write_clips_jsonl(path: Path, clips: List[ClipRecord]) -> None:
    path.write_text("\n".join(c.model_dump_json() for c in clips) + "\n")


def _write_events_jsonl(path: Path, events: List[EventSummaryRecord]) -> None:
    path.write_text("\n".join(e.model_dump_json() for e in events) + "\n")


# ---------------------------------------------------------------------------
# Overlap detection — clips
# ---------------------------------------------------------------------------


def test_link_aux_overlaps_clip(tmp_path: Path):
    aux = _make_aux(BASE_MS + 5_000, BASE_MS + 6_000)
    clip = _make_clip("clip_0001", BASE_MS, BASE_MS + 30_000)

    clips_file = tmp_path / "clips.jsonl"
    _write_clips_jsonl(clips_file, [clip])

    result = link_aux_records([aux], clips_jsonl=clips_file)
    assert len(result) == 1
    assert "clip_0001" in result[0].linked_main_clip_ids


def test_link_aux_no_overlap_clip(tmp_path: Path):
    # aux is entirely after the clip
    aux = _make_aux(BASE_MS + 40_000, BASE_MS + 41_000)
    clip = _make_clip("clip_0001", BASE_MS, BASE_MS + 30_000)

    clips_file = tmp_path / "clips.jsonl"
    _write_clips_jsonl(clips_file, [clip])

    result = link_aux_records([aux], clips_jsonl=clips_file)
    assert result[0].linked_main_clip_ids == []


def test_link_aux_boundary_touching_no_overlap(tmp_path: Path):
    # aux starts exactly when clip ends → no overlap
    aux = _make_aux(BASE_MS + 30_000, BASE_MS + 30_001)
    clip = _make_clip("clip_0001", BASE_MS, BASE_MS + 30_000)

    clips_file = tmp_path / "clips.jsonl"
    _write_clips_jsonl(clips_file, [clip])

    result = link_aux_records([aux], clips_jsonl=clips_file)
    assert result[0].linked_main_clip_ids == []


def test_link_aux_multiple_clips_only_overlapping_linked(tmp_path: Path):
    aux = _make_aux(BASE_MS + 25_000, BASE_MS + 35_000)
    clip_a = _make_clip("clip_a", BASE_MS, BASE_MS + 30_000)
    clip_b = _make_clip("clip_b", BASE_MS + 30_000, BASE_MS + 60_000)
    clip_c = _make_clip("clip_c", BASE_MS + 60_000, BASE_MS + 90_000)

    clips_file = tmp_path / "clips.jsonl"
    _write_clips_jsonl(clips_file, [clip_a, clip_b, clip_c])

    result = link_aux_records([aux], clips_jsonl=clips_file)
    ids = result[0].linked_main_clip_ids
    assert "clip_a" in ids
    assert "clip_b" in ids
    assert "clip_c" not in ids


# ---------------------------------------------------------------------------
# Overlap detection — events
# ---------------------------------------------------------------------------


def test_link_aux_overlaps_event(tmp_path: Path):
    aux = _make_aux(BASE_MS + 1_000, BASE_MS + 2_000)
    event = _make_event("ev_0001", BASE_MS, BASE_MS + 120_000)

    events_file = tmp_path / "events.jsonl"
    _write_events_jsonl(events_file, [event])

    result = link_aux_records([aux], events_jsonl=events_file)
    assert "ev_0001" in result[0].linked_event_summary_ids


def test_link_aux_no_overlap_event(tmp_path: Path):
    aux = _make_aux(BASE_MS + 200_000, BASE_MS + 201_000)
    event = _make_event("ev_0001", BASE_MS, BASE_MS + 120_000)

    events_file = tmp_path / "events.jsonl"
    _write_events_jsonl(events_file, [event])

    result = link_aux_records([aux], events_jsonl=events_file)
    assert result[0].linked_event_summary_ids == []


# ---------------------------------------------------------------------------
# Missing / non-existent JSONL paths
# ---------------------------------------------------------------------------


def test_link_aux_clips_jsonl_none():
    aux = _make_aux(BASE_MS, BASE_MS + 1_000)
    result = link_aux_records([aux], clips_jsonl=None)
    assert result[0].linked_main_clip_ids == []


def test_link_aux_events_jsonl_none():
    aux = _make_aux(BASE_MS, BASE_MS + 1_000)
    result = link_aux_records([aux], events_jsonl=None)
    assert result[0].linked_event_summary_ids == []


def test_link_aux_clips_jsonl_missing_path(tmp_path: Path):
    aux = _make_aux(BASE_MS, BASE_MS + 1_000)
    missing = tmp_path / "no_such_file.jsonl"
    result = link_aux_records([aux], clips_jsonl=missing)
    assert result[0].linked_main_clip_ids == []


def test_link_aux_events_jsonl_missing_path(tmp_path: Path):
    aux = _make_aux(BASE_MS, BASE_MS + 1_000)
    missing = tmp_path / "no_such_file.jsonl"
    result = link_aux_records([aux], events_jsonl=missing)
    assert result[0].linked_event_summary_ids == []


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_link_aux_empty_aux_records():
    result = link_aux_records([])
    assert result == []


def test_link_aux_empty_clips_file(tmp_path: Path):
    aux = _make_aux(BASE_MS, BASE_MS + 1_000)
    clips_file = tmp_path / "clips.jsonl"
    clips_file.write_text("")

    result = link_aux_records([aux], clips_jsonl=clips_file)
    assert result[0].linked_main_clip_ids == []


def test_link_aux_empty_events_file(tmp_path: Path):
    aux = _make_aux(BASE_MS, BASE_MS + 1_000)
    events_file = tmp_path / "events.jsonl"
    events_file.write_text("")

    result = link_aux_records([aux], events_jsonl=events_file)
    assert result[0].linked_event_summary_ids == []


# ---------------------------------------------------------------------------
# Records without reliable timestamp are left unchanged
# ---------------------------------------------------------------------------


def test_link_aux_unreliable_timestamp_not_linked(tmp_path: Path):
    aux = _make_aux(BASE_MS + 5_000, BASE_MS + 6_000, reliable=False)
    clip = _make_clip("clip_0001", BASE_MS, BASE_MS + 30_000)

    clips_file = tmp_path / "clips.jsonl"
    _write_clips_jsonl(clips_file, [clip])

    result = link_aux_records([aux], clips_jsonl=clips_file)
    # No linking performed for unreliable-timestamp records
    assert result[0].linked_main_clip_ids == []
    assert result[0].has_reliable_timestamp is False


# ---------------------------------------------------------------------------
# has_reliable_timestamp set by auxiliary.py iterators
# ---------------------------------------------------------------------------


def test_photo_record_no_timestamp_marked_unreliable(tmp_path: Path):
    """Photo with no EXIF and no filename hint gets has_reliable_timestamp=False."""
    from castlerag.preprocess.auxiliary import iter_photo_records

    photo_dir = tmp_path / "photo" / "Allie"
    photo_dir.mkdir(parents=True)
    (photo_dir / "IMG_no_ts.jpg").touch()

    with patch("castlerag.preprocess.auxiliary._exif_unix_ms", return_value=None):
        records = list(iter_photo_records(tmp_path, "Allie", "day1"))

    assert len(records) == 1
    # No EXIF and no timestamp in filename → abs_start == 0 → unreliable
    assert records[0].has_reliable_timestamp is False
    assert records[0].absolute_start == 0


def test_photo_record_with_filename_hint_marked_reliable(tmp_path: Path):
    """Photo with a parseable filename timestamp gets has_reliable_timestamp=True."""
    from castlerag.preprocess.auxiliary import iter_photo_records

    photo_dir = tmp_path / "photo" / "Allie"
    photo_dir.mkdir(parents=True)
    (photo_dir / "2023-01-05_08-00-00.jpg").touch()

    with patch("castlerag.preprocess.auxiliary._exif_unix_ms", return_value=None):
        records = list(iter_photo_records(tmp_path, "Allie", "day1"))

    assert len(records) == 1
    assert records[0].has_reliable_timestamp is True
    assert records[0].absolute_start > 0


def test_thermal_record_no_ts_hint_marked_unreliable(tmp_path: Path):
    """Thermal file without a timestamp hint gets has_reliable_timestamp=False."""
    from castlerag.preprocess.auxiliary import iter_thermal_records

    thermal_dir = tmp_path / "thermal"
    thermal_dir.mkdir(parents=True)
    (thermal_dir / "frame_no_ts.bmp").touch()

    records = list(iter_thermal_records(tmp_path, "day2"))
    assert len(records) == 1
    assert records[0].has_reliable_timestamp is False
    assert records[0].absolute_start == 0


def test_thermal_record_with_ts_hint_marked_reliable(tmp_path: Path):
    """Thermal file with a parseable timestamp hint gets has_reliable_timestamp=True."""
    from castlerag.preprocess.auxiliary import iter_thermal_records

    thermal_dir = tmp_path / "thermal"
    thermal_dir.mkdir(parents=True)
    (thermal_dir / "2023-01-05_10-30-00.bmp").touch()

    records = list(iter_thermal_records(tmp_path, "day2"))
    assert len(records) == 1
    assert records[0].has_reliable_timestamp is True
    assert records[0].absolute_start > 0


# ---------------------------------------------------------------------------
# summary_text emitted by iterators
# ---------------------------------------------------------------------------


def test_photo_record_has_summary_text(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_photo_records

    photo_dir = tmp_path / "photo" / "Allie"
    photo_dir.mkdir(parents=True)
    (photo_dir / "IMG_001.jpg").touch()

    with patch("castlerag.preprocess.auxiliary._exif_unix_ms", return_value=None):
        records = list(iter_photo_records(tmp_path, "Allie", "day1"))

    assert records[0].summary_text == "Photo by Allie"


def test_thermal_record_has_summary_text(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_thermal_records

    thermal_dir = tmp_path / "thermal"
    thermal_dir.mkdir(parents=True)
    (thermal_dir / "frame_001.bmp").touch()

    records = list(iter_thermal_records(tmp_path, "day2"))
    assert records[0].summary_text == "Thermal frame"


# ---------------------------------------------------------------------------
# Original objects are not mutated
# ---------------------------------------------------------------------------


def test_link_aux_does_not_mutate_originals(tmp_path: Path):
    aux = _make_aux(BASE_MS + 5_000, BASE_MS + 6_000)
    clip = _make_clip("clip_0001", BASE_MS, BASE_MS + 30_000)

    clips_file = tmp_path / "clips.jsonl"
    _write_clips_jsonl(clips_file, [clip])

    original_ids = list(aux.linked_main_clip_ids)
    result = link_aux_records([aux], clips_jsonl=clips_file)

    # Original unchanged
    assert aux.linked_main_clip_ids == original_ids
    # Result is linked
    assert "clip_0001" in result[0].linked_main_clip_ids
