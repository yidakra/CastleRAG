"""regen_frames rebuilds exactly the listed (thinned) frames from the video."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from castlerag.frame_encoding import sample_frames_evenly

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "regen_frames", REPO / "scripts" / "regen_frames.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _day(tmp_path: Path, video: Path, frames: list[str]) -> Path:
    chunks = tmp_path / "chunks" / "day1" / "Allie" / "08"
    chunks.mkdir(parents=True)
    row = {
        "clip_id": "day1_Allie_08_0000",
        "source_video_path": str(video),
        "start_seconds": 0.0,
        "end_seconds": 30.0,
        "sampled_frame_paths": frames,
    }
    (chunks / "clips.jsonl").write_text(json.dumps(row) + "\n")
    return tmp_path / "chunks" / "day1"


def _thinned(tmp_path: Path) -> list[str]:
    clip_dir = tmp_path / "frames" / "day1" / "Allie" / "08" / "0"
    all_frames = [str(clip_dir / f"{i:04d}.jpg") for i in range(1, 31)]
    return sample_frames_evenly(all_frames, 8)


def test_regen_restores_only_listed_frames(tmp_path: Path, monkeypatch):
    rf = _load()
    video = tmp_path / "day1" / "Allie" / "08.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"mp4")

    def fake_extract(src, out_dir, start, end, fps=1):
        for i in range(1, 31):
            (out_dir / f"{i:04d}.jpg").write_bytes(b"jpg")

    monkeypatch.setattr(rf, "extract_frames_1fps", fake_extract)
    kept = _thinned(tmp_path)
    day = _day(tmp_path, video, kept)

    dry = rf.regen_day(day, apply=False)
    assert dry == {"clips": 1, "frames": 8, "written": 0, "failed": 0}
    done = rf.regen_day(day, apply=True)
    assert done["written"] == 8 and done["failed"] == 0
    clip_dir = Path(kept[0]).parent
    assert sorted(str(p) for p in clip_dir.glob("*.jpg")) == sorted(kept)
    assert not list(clip_dir.parent.glob(".regen_*"))  # temp dir cleaned up
    assert rf.regen_day(day, apply=True)["clips"] == 0  # nothing left to do


def test_regen_reports_missing_video(tmp_path: Path):
    rf = _load()
    day = _day(tmp_path, tmp_path / "gone.mp4", _thinned(tmp_path))
    stats = rf.regen_day(day, apply=True)
    assert stats["failed"] == 1 and stats["written"] == 0


def test_video_root_remaps_moved_dataset(tmp_path: Path, monkeypatch):
    rf = _load()
    new_root = tmp_path / "new" / "main"
    video = new_root / "day1" / "Allie" / "08.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"mp4")
    seen = []

    def fake_extract(src, out_dir, start, end, fps=1):
        seen.append(src)
        for i in range(1, 31):
            (out_dir / f"{i:04d}.jpg").write_bytes(b"jpg")

    monkeypatch.setattr(rf, "extract_frames_1fps", fake_extract)
    day = _day(tmp_path, Path("/old/root/main/day1/Allie/08.mp4"), _thinned(tmp_path))
    assert rf.regen_day(day, apply=True, video_root=new_root)["failed"] == 0
    assert seen == [video]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_regen_matches_original_extraction(tmp_path: Path):
    from castlerag.preprocess.media import extract_frames_1fps

    rf = _load()
    video = tmp_path / "day1" / "Allie" / "08.mp4"
    video.parent.mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=40:size=64x48:rate=5",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        capture_output=True,
        check=True,
    )
    clip_dir = tmp_path / "frames" / "day1" / "Allie" / "08" / "0"
    original = extract_frames_1fps(video, clip_dir, 0.0, 30.0)
    kept = sample_frames_evenly([str(p) for p in original], 8)
    before = {p: Path(p).read_bytes() for p in kept}
    shutil.rmtree(clip_dir)  # scratch purge
    day = _day(tmp_path, video, kept)
    assert rf.regen_day(day, apply=True)["written"] == 8
    assert {p: Path(p).read_bytes() for p in kept} == before


def test_frame_extraction_caps_decoder_threads(tmp_path: Path, monkeypatch):
    from castlerag.preprocess import media

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["timeout"] = cmd, kw.get("timeout")

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    media.extract_frames_1fps(tmp_path / "v.mp4", tmp_path / "out", 0.0, 30.0)
    cmd = seen["cmd"]
    # -threads must come before -i to limit the decoder
    assert cmd.index("-threads") < cmd.index("-i")
    assert cmd[cmd.index("-threads") + 1] == str(media.FRAME_DECODE_THREADS)
    assert seen["timeout"] == media.FRAME_TIMEOUT_SECONDS
