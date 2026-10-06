"""Frame thinning: readers sample surviving frames; thin_frames keeps them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from castlerag.frame_encoding import available_frames, sample_frames_evenly
from castlerag.generation.answer import _gather_frame_paths
from castlerag.retrieval.candidate_expand import _collect_frame_paths

REPO = Path(__file__).resolve().parents[2]


def _load_thin_frames():
    spec = importlib.util.spec_from_file_location(
        "thin_frames", REPO / "scripts" / "thin_frames.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clip_frames(root: Path, clip: str, n: int = 30) -> list[str]:
    d = root / clip
    d.mkdir(parents=True)
    paths = []
    for i in range(1, n + 1):
        p = d / f"{i:04d}.jpg"
        p.write_bytes(b"jpg")
        paths.append(str(p))
    return paths


def test_available_frames_keeps_existing_or_falls_back(tmp_path: Path):
    paths = _clip_frames(tmp_path, "c0", n=4)
    Path(paths[1]).unlink()
    assert available_frames(paths) == [paths[0], paths[2], paths[3]]
    ghost = ["/no/such/0001.jpg", "/no/such/0002.jpg"]
    assert available_frames(ghost) == ghost  # nothing on disk: unchanged


def test_readers_sample_only_surviving_frames(tmp_path: Path):
    paths = _clip_frames(tmp_path, "c0")
    kept = sample_frames_evenly(paths, 8)
    for p in paths:
        if p not in kept:
            Path(p).unlink()
    row = SimpleNamespace(sampled_frame_paths=paths)  # payload still lists all 30
    pack = _collect_frame_paths([row], max_frames=32)
    assert pack == kept
    assert sample_frames_evenly(pack, 4) == sample_frames_evenly(kept, 4)
    gen = _gather_frame_paths([row], max_frames=8)
    assert gen == kept and all(Path(p).exists() for p in gen)


def test_thin_day_dry_run_then_apply(tmp_path: Path):
    tf = _load_thin_frames()
    frames_root = tmp_path / "frames" / "day1" / "Allie" / "08"
    paths = _clip_frames(frames_root, "clip0")
    chunks = tmp_path / "chunks" / "day1" / "Allie" / "08"
    chunks.mkdir(parents=True)
    clip = {"clip_id": "c0", "sampled_frame_paths": paths}
    event = {"event_summary_id": "e0", "sampled_frame_paths": paths[:3]}
    (chunks / "clips.jsonl").write_text(json.dumps(clip) + "\n")
    (chunks / "events.jsonl").write_text(json.dumps(event) + "\n")

    dry = tf.thin_day(tmp_path / "chunks" / "day1", keep=8, apply=False)
    assert dry["dropped"] == 22 and dry["files"] == 0
    assert all(Path(p).exists() for p in paths)

    done = tf.thin_day(tmp_path / "chunks" / "day1", keep=8, apply=True)
    kept = sample_frames_evenly(paths, 8)
    assert done["kept"] == 8 and done["dropped"] == 22
    assert sorted(str(p) for p in Path(paths[0]).parent.glob("*.jpg")) == sorted(kept)
    rewritten = json.loads((chunks / "clips.jsonl").read_text())
    assert rewritten["sampled_frame_paths"] == kept
    ev = json.loads((chunks / "events.jsonl").read_text())
    assert all(p in kept for p in ev["sampled_frame_paths"])
    assert (chunks / "clips.jsonl.prethin").exists()

    again = tf.thin_day(tmp_path / "chunks" / "day1", keep=8, apply=True)
    assert again["dropped"] == 0  # idempotent


def _day_with_clip(tmp_path: Path, paths: list[str]) -> Path:
    chunks = tmp_path / "chunks" / "day1" / "Allie" / "08"
    chunks.mkdir(parents=True)
    clip = {"clip_id": "c0", "sampled_frame_paths": paths}
    event = {"event_summary_id": "e0", "sampled_frame_paths": paths}
    (chunks / "clips.jsonl").write_text(json.dumps(clip) + "\n")
    (chunks / "events.jsonl").write_text(json.dumps(event) + "\n")
    return tmp_path / "chunks" / "day1"


def test_thin_day_keeps_only_existing_frames(tmp_path: Path):
    tf = _load_thin_frames()
    paths = _clip_frames(tmp_path / "frames", "clip0")
    for p in paths[8:]:
        Path(p).unlink()  # only the first 8 of 30 listed frames survive
    day = _day_with_clip(tmp_path, paths)
    tf.thin_day(day, keep=8, apply=True)
    row = json.loads(next(day.rglob("clips.jsonl")).read_text())
    assert row["sampled_frame_paths"] == paths[:8]
    assert all(Path(p).exists() for p in row["sampled_frame_paths"])


def test_thin_day_resumes_an_interrupted_apply(tmp_path: Path, monkeypatch):
    tf = _load_thin_frames()
    paths = _clip_frames(tmp_path / "frames", "clip0")
    day = _day_with_clip(tmp_path, paths)
    real_rewrite = tf._rewrite_jsonl

    def crash_after_clips(path, rows):
        real_rewrite(path, rows)
        if path.name == "clips.jsonl":
            raise RuntimeError("killed")

    monkeypatch.setattr(tf, "_rewrite_jsonl", crash_after_clips)
    try:
        tf.thin_day(day, keep=8, apply=True)
    except RuntimeError:
        pass
    assert (day / tf.PLAN_NAME).exists()
    monkeypatch.setattr(tf, "_rewrite_jsonl", real_rewrite)
    tf.thin_day(day, keep=8, apply=True)
    kept = sample_frames_evenly(paths, 8)
    ev = json.loads(next(day.rglob("events.jsonl")).read_text())
    assert ev["sampled_frame_paths"] == kept
    assert sorted(str(p) for p in Path(paths[0]).parent.glob("*.jpg")) == sorted(kept)
    assert not (day / tf.PLAN_NAME).exists()


def test_main_rejects_missing_config(tmp_path: Path, monkeypatch, capsys):
    tf = _load_thin_frames()
    monkeypatch.setattr(
        "sys.argv",
        ["thin_frames.py", "--config", str(tmp_path / "nope.yaml"), "--day", "1"],
    )
    try:
        tf.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("missing --config was accepted")
    assert "does not exist" in capsys.readouterr().err


def test_bounded_map_keeps_order_across_batches():
    from concurrent.futures import ThreadPoolExecutor

    tf = _load_thin_frames()
    items = [str(i) for i in range(10)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        out = list(tf._bounded_map(pool, lambda x: int(x) * 2, items, batch=4))
    assert out == [i * 2 for i in range(10)]
