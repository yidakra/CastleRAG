"""Project-space move: config-driven paths and frame path aliases."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from castlerag.cli import app
from castlerag.config import load_config
from castlerag.frame_encoding import (
    available_frames,
    relocate_frame_path,
    resolve_frame_path,
    set_frame_path_aliases,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_aliases_after():
    yield
    set_frame_path_aliases({})


def _frames(root: Path, n: int = 3) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(1, n + 1):
        p = root / f"{i:04d}.jpg"
        p.write_bytes(b"jpg")
        out.append(str(p))
    return out


def test_moved_frames_resolve_through_the_alias(tmp_path: Path):
    old, new = tmp_path / "scratch" / "frames", tmp_path / "project" / "frames"
    stored = [str(old / "day1" / "Allie" / "08" / "0" / f"{i:04d}.jpg") for i in (1, 2)]
    _frames(new / "day1" / "Allie" / "08" / "0", 2)
    assert available_frames(stored) == stored  # no alias: missing, unchanged
    set_frame_path_aliases({str(old): str(new)})
    moved = available_frames(stored)
    assert moved == [p.replace(str(old), str(new)) for p in stored]
    assert all(Path(p).exists() for p in moved)


def test_a_path_that_still_exists_wins_over_the_alias(tmp_path: Path):
    old, new = tmp_path / "a", tmp_path / "b"
    (p,) = _frames(old / "x", 1)
    _frames(new / "x", 1)
    set_frame_path_aliases({str(old): str(new)})
    assert resolve_frame_path(p) == p
    assert relocate_frame_path(p) == str(new / "x" / "0001.jpg")


def test_alias_matches_whole_path_components(tmp_path: Path):
    set_frame_path_aliases({"/s/frames": "/p/frames"})
    assert relocate_frame_path("/s/frames/day1/a.jpg") == "/p/frames/day1/a.jpg"
    assert relocate_frame_path("/s/frames_old/day1/a.jpg") == "/s/frames_old/day1/a.jpg"


def test_load_config_registers_aliases_with_user_expanded_in_keys(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("USER", "tester")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "preprocessing:\n"
        "  frame_path_aliases:\n"
        '    "/scratch/$USER/frames": "/projects/p1/frames"\n'
    )
    cfg = load_config(override_path=cfg_path)
    assert cfg.preprocessing.frame_path_aliases == {
        "/scratch/tester/frames": "/projects/p1/frames"
    }
    assert (
        relocate_frame_path("/scratch/tester/frames/x.jpg")
        == "/projects/p1/frames/x.jpg"
    )


def test_paths_command_prints_project_config_paths(monkeypatch):
    monkeypatch.setenv("USER", "tester")
    result = CliRunner().invoke(
        app, ["paths", "--config", str(REPO / "configs" / "snellius_project.yaml")]
    )
    assert result.exit_code == 0, result.output
    values = dict(line.split("=", 1) for line in result.output.strip().splitlines())
    assert values["CHUNKS_DIR"] == "/projects/prjs2298/castle_derived/chunks"
    assert values["EMB_DIR"] == "/projects/prjs2298/castle_derived/embeddings"
    assert values["QDRANT_STORAGE"] == "/projects/prjs2298/qdrant_storage"
    assert values["DATA_ROOT"] == "/scratch-shared/tester/castle2024"


def test_paths_command_keeps_scratch_for_the_current_config(monkeypatch):
    monkeypatch.setenv("USER", "tester")
    result = CliRunner().invoke(
        app, ["paths", "--config", str(REPO / "configs" / "snellius_fixedcams.yaml")]
    )
    assert result.exit_code == 0, result.output
    assert "CHUNKS_DIR=/scratch-shared/tester/castle_derived/chunks" in result.output
    assert "QDRANT_STORAGE=/scratch-shared/tester/qdrant_storage" in result.output


def test_paths_command_rejects_a_missing_config(tmp_path: Path):
    result = CliRunner().invoke(app, ["paths", "--config", str(tmp_path / "x.yaml")])
    assert result.exit_code == 2


def test_regen_writes_rebuilt_frames_to_the_new_home(tmp_path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "regen_frames", REPO / "scripts" / "regen_frames.py"
    )
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    old, new = tmp_path / "scratch", tmp_path / "project"
    stored = [str(old / "day1" / "Allie" / "08" / "0" / f"{i:04d}.jpg") for i in (1, 5)]
    video = tmp_path / "v.mp4"
    video.write_bytes(b"mp4")
    chunks = tmp_path / "chunks" / "day1" / "Allie" / "08"
    chunks.mkdir(parents=True)
    row = {
        "clip_id": "c0",
        "source_video_path": str(video),
        "start_seconds": 0.0,
        "end_seconds": 30.0,
        "sampled_frame_paths": stored,
    }
    (chunks / "clips.jsonl").write_text(json.dumps(row) + "\n")

    def fake_extract(src, out_dir, start, end, fps=1):
        for i in range(1, 31):
            (out_dir / f"{i:04d}.jpg").write_bytes(b"jpg")

    monkeypatch.setattr(rf, "extract_frames_1fps", fake_extract)
    set_frame_path_aliases({str(old): str(new)})
    stats = rf.regen_day(tmp_path / "chunks" / "day1", apply=True)
    assert stats["written"] == 2
    assert all(Path(p.replace(str(old), str(new))).exists() for p in stored)
    assert not old.exists()  # nothing written back to scratch
