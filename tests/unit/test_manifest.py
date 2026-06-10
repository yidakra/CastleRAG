"""Tests for src/castlerag/dataset/manifest.py"""

from pathlib import Path

from castlerag.dataset.manifest import (
    AuxAssetRow,
    _extract_timestamp_hint,
    _participant_from_stem,
    discover_aux_manifest,
    discover_main_manifest,
    read_aux_manifest,
    read_main_manifest,
    write_manifest,
)

EGO = ["Allie", "Bjorn"]
EXO = ["Kitchen", "Living1"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video(root: Path, day: str, camera: str, hour: int) -> Path:
    d = root / "main" / day / camera / "video"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{hour:02d}.mp4"
    p.touch()
    return p


def _make_novideo(root: Path, day: str, camera: str, hour: int) -> Path:
    d = root / "main" / day / camera / "video"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{hour:02d}.novideo"
    p.touch()
    return p


def _make_transcript(root: Path, day: str, camera: str, hour: int) -> Path:
    d = root / "main" / day / camera / "transcript"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{hour:02d}.json"
    p.write_text("{}")
    return p


# ---------------------------------------------------------------------------
# Main manifest discovery
# ---------------------------------------------------------------------------


def test_discover_main_manifest_normal_hour(tmp_path: Path):
    _make_video(tmp_path, "day1", "Allie", 8)
    rows = discover_main_manifest(tmp_path, EGO, EXO, days=[1], hours=[8])
    assert len(rows) == 1
    r = rows[0]
    assert r.day == "day1"
    assert r.camera_id == "Allie"
    assert r.camera_type == "ego"
    assert r.participant_id == "Allie"
    assert r.hour == 8
    assert r.missing_video is False
    assert r.video_path.endswith("08.mp4")


def test_discover_main_manifest_includes_novideo(tmp_path: Path):
    _make_novideo(tmp_path, "day1", "Bjorn", 10)
    rows = discover_main_manifest(tmp_path, EGO, EXO, days=[1], hours=[10])
    assert len(rows) == 1
    assert rows[0].missing_video is True
    assert rows[0].camera_id == "Bjorn"


def test_discover_main_manifest_excludes_absent_hours(tmp_path: Path):
    # No file at all for Allie hour 9 — should not appear
    _make_video(tmp_path, "day1", "Allie", 8)
    rows = discover_main_manifest(tmp_path, EGO, EXO, days=[1], hours=[8, 9])
    assert all(r.hour == 8 for r in rows)
    assert len(rows) == 1


def test_discover_main_manifest_transcript_path_present(tmp_path: Path):
    _make_video(tmp_path, "day1", "Allie", 8)
    _make_transcript(tmp_path, "day1", "Allie", 8)
    rows = discover_main_manifest(tmp_path, EGO, EXO, days=[1], hours=[8])
    assert rows[0].transcript_path is not None
    assert rows[0].transcript_path.endswith("08.json")


def test_discover_main_manifest_transcript_absent(tmp_path: Path):
    _make_video(tmp_path, "day1", "Allie", 8)
    rows = discover_main_manifest(tmp_path, EGO, EXO, days=[1], hours=[8])
    assert rows[0].transcript_path is None


def test_discover_main_manifest_ego_scope_only(tmp_path: Path):
    for cam in ("Allie", "Kitchen"):
        _make_video(tmp_path, "day1", cam, 8)
    rows = discover_main_manifest(
        tmp_path,
        ego_cameras=["Allie"],
        exo_cameras=["Kitchen"],
        days=[1],
        hours=[8],
        camera_scope="ego",
    )
    camera_ids = {r.camera_id for r in rows}
    assert "Allie" in camera_ids
    assert "Kitchen" not in camera_ids


def test_discover_main_manifest_all_scope_includes_exo(tmp_path: Path):
    for cam in ("Allie", "Kitchen"):
        _make_video(tmp_path, "day1", cam, 8)
    rows = discover_main_manifest(
        tmp_path,
        ego_cameras=["Allie"],
        exo_cameras=["Kitchen"],
        days=[1],
        hours=[8],
        camera_scope="all",
    )
    camera_ids = {r.camera_id for r in rows}
    assert "Kitchen" in camera_ids


def test_discover_main_manifest_fixed_camera_metadata(tmp_path: Path):
    _make_video(tmp_path, "day2", "Kitchen", 12)
    rows = discover_main_manifest(
        tmp_path,
        ego_cameras=[],
        exo_cameras=["Kitchen"],
        days=[2],
        hours=[12],
        camera_scope="all",
    )
    assert rows[0].camera_type == "fixed"
    assert rows[0].participant_id is None


def test_discover_main_manifest_metadata_dir(tmp_path: Path):
    _make_video(tmp_path, "day1", "Allie", 8)
    rows = discover_main_manifest(tmp_path, EGO, EXO, days=[1], hours=[8])
    assert "metadata" in rows[0].metadata_dir


def test_discover_main_manifest_version_field(tmp_path: Path):
    _make_video(tmp_path, "day1", "Allie", 8)
    rows = discover_main_manifest(
        tmp_path, EGO, EXO, days=[1], hours=[8], version="1.2.3"
    )
    assert rows[0].version == "1.2.3"


def test_discover_main_manifest_multiple_days(tmp_path: Path):
    _make_video(tmp_path, "day1", "Allie", 8)
    _make_video(tmp_path, "day2", "Allie", 8)
    rows = discover_main_manifest(tmp_path, ["Allie"], [], days=[1, 2], hours=[8])
    days = {r.day for r in rows}
    assert days == {"day1", "day2"}


# ---------------------------------------------------------------------------
# Auxiliary manifest discovery
# ---------------------------------------------------------------------------


def test_discover_aux_manifest_empty_root(tmp_path: Path):
    rows = discover_aux_manifest(tmp_path)
    assert rows == []


def test_discover_aux_manifest_heartrate(tmp_path: Path):
    f = tmp_path / "auxiliary" / "heartrate" / "Allie" / "hr_data.csv"
    f.parent.mkdir(parents=True)
    f.touch()
    rows = discover_aux_manifest(tmp_path)
    assert len(rows) == 1
    assert rows[0].source_type == "aux_heartrate"
    assert rows[0].participant_id == "Allie"
    assert rows[0].path.endswith("hr_data.csv")


def test_discover_aux_manifest_gaze(tmp_path: Path):
    f = tmp_path / "auxiliary" / "gaze" / "Bjorn_gaze.csv"
    f.parent.mkdir(parents=True)
    f.touch()
    rows = discover_aux_manifest(tmp_path)
    assert len(rows) == 1
    assert rows[0].source_type == "aux_gaze"
    assert rows[0].participant_id == "Bjorn"


def test_discover_aux_manifest_gaze_no_participant(tmp_path: Path):
    f = tmp_path / "auxiliary" / "gaze" / "session01.csv"
    f.parent.mkdir(parents=True)
    f.touch()
    rows = discover_aux_manifest(tmp_path)
    # "session01" starts with alpha but has digits — participant should be None
    assert rows[0].participant_id is None


def test_discover_aux_manifest_photo(tmp_path: Path):
    f = tmp_path / "auxiliary" / "photo" / "Allie" / "IMG_001.jpg"
    f.parent.mkdir(parents=True)
    f.touch()
    rows = discover_aux_manifest(tmp_path)
    assert rows[0].source_type == "aux_photo"
    assert rows[0].participant_id == "Allie"


def test_discover_aux_manifest_thermal(tmp_path: Path):
    f = tmp_path / "auxiliary" / "thermal" / "frame_001.bmp"
    f.parent.mkdir(parents=True)
    f.touch()
    rows = discover_aux_manifest(tmp_path)
    assert rows[0].source_type == "aux_thermal"
    assert rows[0].participant_id is None


def test_discover_aux_manifest_video(tmp_path: Path):
    f = tmp_path / "auxiliary" / "video" / "Bjorn" / "clip.mp4"
    f.parent.mkdir(parents=True)
    f.touch()
    rows = discover_aux_manifest(tmp_path)
    assert rows[0].source_type == "aux_video"
    assert rows[0].participant_id == "Bjorn"


def test_discover_aux_manifest_multiple_types(tmp_path: Path):
    (tmp_path / "auxiliary" / "heartrate" / "Allie").mkdir(parents=True)
    (tmp_path / "auxiliary" / "heartrate" / "Allie" / "hr.csv").touch()
    (tmp_path / "auxiliary" / "thermal").mkdir(parents=True)
    (tmp_path / "auxiliary" / "thermal" / "t.bmp").touch()
    rows = discover_aux_manifest(tmp_path)
    types = {r.source_type for r in rows}
    assert "aux_heartrate" in types
    assert "aux_thermal" in types


# ---------------------------------------------------------------------------
# Timestamp hint extraction
# ---------------------------------------------------------------------------


def test_extract_timestamp_hint_datetime(tmp_path: Path):
    f = tmp_path / "2023-01-05_14-22-00_clip.mp4"
    assert _extract_timestamp_hint(f) == "2023-01-05_14-22-00"


def test_extract_timestamp_hint_compact(tmp_path: Path):
    f = tmp_path / "20230105T142200.mp4"
    assert _extract_timestamp_hint(f) == "20230105T142200"


def test_extract_timestamp_hint_date_only(tmp_path: Path):
    f = tmp_path / "2023-01-05_data.csv"
    assert _extract_timestamp_hint(f) == "2023-01-05"


def test_extract_timestamp_hint_no_match(tmp_path: Path):
    f = tmp_path / "no_date_here.csv"
    assert _extract_timestamp_hint(f) is None


# ---------------------------------------------------------------------------
# Participant from stem
# ---------------------------------------------------------------------------


def test_participant_from_stem_alpha():
    assert _participant_from_stem("Allie_gaze") == "Allie"


def test_participant_from_stem_mixed():
    assert _participant_from_stem("session01") is None


def test_participant_from_stem_empty():
    assert _participant_from_stem("") is None


# ---------------------------------------------------------------------------
# JSONL write / read roundtrip
# ---------------------------------------------------------------------------


def test_write_read_main_manifest_roundtrip(tmp_path: Path):
    _make_video(tmp_path / "data", "day1", "Allie", 8)
    rows = discover_main_manifest(tmp_path / "data", ["Allie"], [], days=[1], hours=[8])
    out = tmp_path / "manifests" / "main.jsonl"
    write_manifest(rows, out)
    loaded = read_main_manifest(out)
    assert len(loaded) == 1
    assert loaded[0].day == rows[0].day
    assert loaded[0].camera_id == rows[0].camera_id
    assert loaded[0].missing_video == rows[0].missing_video


def test_write_read_aux_manifest_roundtrip(tmp_path: Path):
    f = tmp_path / "auxiliary" / "heartrate" / "Allie" / "hr.csv"
    f.parent.mkdir(parents=True)
    f.touch()
    rows = discover_aux_manifest(tmp_path)
    out = tmp_path / "manifests" / "aux.jsonl"
    write_manifest(rows, out)
    loaded = read_aux_manifest(out)
    assert len(loaded) == 1
    assert loaded[0].source_type == rows[0].source_type
    assert loaded[0].participant_id == rows[0].participant_id


def test_write_manifest_creates_parent_dirs(tmp_path: Path):
    row = AuxAssetRow(source_type="aux_thermal", participant_id=None, path="/aux/t.bmp")
    out = tmp_path / "deep" / "nested" / "dir" / "aux.jsonl"
    write_manifest([row], out)
    assert out.exists()


def test_write_manifest_empty(tmp_path: Path):
    out = tmp_path / "empty.jsonl"
    write_manifest([], out)
    assert out.read_text() == ""


def test_read_main_manifest_skips_blank_lines(tmp_path: Path):
    _make_video(tmp_path / "data", "day1", "Allie", 8)
    rows = discover_main_manifest(tmp_path / "data", ["Allie"], [], days=[1], hours=[8])
    out = tmp_path / "main.jsonl"
    write_manifest(rows, out)
    # inject blank lines
    content = out.read_text()
    out.write_text("\n" + content + "\n\n")
    loaded = read_main_manifest(out)
    assert len(loaded) == 1
