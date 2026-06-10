"""Tests for src/castlerag/dataset/layout.py"""

from pathlib import Path

import pytest

from castlerag.dataset.layout import (
    build_camera_registry,
    discover_hours,
    hour_transcript_path,
    hour_video_path,
    is_novideo,
)

EGO = ["Allie", "Bjorn"]
EXO = ["Kitchen", "Living1"]


def test_build_camera_registry_ego():
    reg = build_camera_registry(EGO, EXO)
    assert reg["Allie"].camera_type == "ego"
    assert reg["Allie"].participant_id == "Allie"
    assert reg["Allie"].room is None


def test_build_camera_registry_fixed():
    reg = build_camera_registry(EGO, EXO)
    assert reg["Kitchen"].camera_type == "fixed"
    assert reg["Kitchen"].participant_id is None
    assert reg["Kitchen"].room == "Kitchen"


def test_hour_video_path():
    root = Path("/data/castle2024")
    p = hour_video_path(root, "day1", "Allie", 8)
    assert p == Path("/data/castle2024/main/day1/Allie/video/08.mp4")


def test_hour_transcript_path():
    root = Path("/data/castle2024")
    p = hour_transcript_path(root, "day2", "Bjorn", 14)
    assert p == Path("/data/castle2024/main/day2/Bjorn/transcript/14.json")


def test_hour_video_path_zero_padding():
    root = Path("/data/castle2024")
    p = hour_video_path(root, "day1", "Allie", 9)
    assert p.name == "09.mp4"


def test_is_novideo_no_file(tmp_path: Path):
    assert not is_novideo(tmp_path, "day1", "Allie", 8)


def test_is_novideo_with_novideo_marker(tmp_path: Path):
    video_dir = tmp_path / "main" / "day1" / "Allie" / "video"
    video_dir.mkdir(parents=True)
    (video_dir / "08.novideo").touch()
    assert is_novideo(tmp_path, "day1", "Allie", 8)


def test_is_novideo_when_mp4_exists(tmp_path: Path):
    video_dir = tmp_path / "main" / "day1" / "Allie" / "video"
    video_dir.mkdir(parents=True)
    (video_dir / "08.novideo").touch()
    (video_dir / "08.mp4").touch()
    # mp4 present → not novideo
    assert not is_novideo(tmp_path, "day1", "Allie", 8)


def test_discover_hours_ego_scope(tmp_path: Path):
    # Create a fake video file for Allie day1 hour 8
    video_dir = tmp_path / "main" / "day1" / "Allie" / "video"
    video_dir.mkdir(parents=True)
    (video_dir / "08.mp4").touch()

    assets = list(
        discover_hours(
            root=tmp_path,
            ego_cameras=["Allie"],
            exo_cameras=["Kitchen"],
            days=[1],
            hours=[8],
            camera_scope="ego",
        )
    )
    assert len(assets) == 1
    assert assets[0].camera_id == "Allie"
    assert assets[0].camera_type == "ego"
    assert assets[0].hour == 8


def test_discover_hours_skips_exo_by_default(tmp_path: Path):
    for cam in ("Allie", "Kitchen"):
        video_dir = tmp_path / "main" / "day1" / cam / "video"
        video_dir.mkdir(parents=True)
        (video_dir / "08.mp4").touch()

    assets = list(
        discover_hours(
            root=tmp_path,
            ego_cameras=["Allie"],
            exo_cameras=["Kitchen"],
            days=[1],
            hours=[8],
            camera_scope="ego",  # default — Kitchen must be skipped
        )
    )
    camera_ids = {a.camera_id for a in assets}
    assert "Allie" in camera_ids
    assert "Kitchen" not in camera_ids


def test_discover_hours_all_scope_includes_exo(tmp_path: Path):
    for cam in ("Allie", "Kitchen"):
        video_dir = tmp_path / "main" / "day1" / cam / "video"
        video_dir.mkdir(parents=True)
        (video_dir / "08.mp4").touch()

    assets = list(
        discover_hours(
            root=tmp_path,
            ego_cameras=["Allie"],
            exo_cameras=["Kitchen"],
            days=[1],
            hours=[8],
            camera_scope="all",
        )
    )
    camera_ids = {a.camera_id for a in assets}
    assert "Allie" in camera_ids
    assert "Kitchen" in camera_ids


def test_build_camera_registry_rejects_overlap():
    with pytest.raises(ValueError, match="unique"):
        build_camera_registry(["Allie", "Kitchen"], ["Kitchen"])


def test_discover_hours_skips_novideo(tmp_path: Path):
    video_dir = tmp_path / "main" / "day1" / "Allie" / "video"
    video_dir.mkdir(parents=True)
    (video_dir / "08.novideo").touch()  # no mp4

    assets = list(
        discover_hours(
            root=tmp_path,
            ego_cameras=["Allie"],
            exo_cameras=[],
            days=[1],
            hours=[8],
        )
    )
    assert len(assets) == 0
