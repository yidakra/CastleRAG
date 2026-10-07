"""Placeholder clips (#68): skipped by captioning, left out of the index,
pruned from Qdrant when re-flagged, and re-flaggable on ingested days."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from castlerag.cli import app
from castlerag.index.io import load_clip_records, write_jsonl_records
from castlerag.index.pipeline import (
    LoadedArtifacts,
    filter_records,
    prune_stale_points,
)
from castlerag.schemas import EventSummaryRecord
from tests.unit.test_fixedcams import EGO, EXO, _cfg, _clip
from tests.unit.test_preprocessing import _save_card, _save_still_room

REPO = Path(__file__).resolve().parents[2]


def _event(cam: str, eid: str) -> EventSummaryRecord:
    return EventSummaryRecord(
        event_summary_id=eid,
        day="day1",
        camera_id=cam,
        camera_type="fixed",
        room=cam,
        absolute_start=0,
        absolute_end=120_000,
    )


def _cfg_file(tmp_path: Path, chunks: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "dataset:\n"
        f"  ego_cameras: {EGO}\n"
        f"  exo_cameras: {EXO}\n"
        "  camera_scope: all\n"
        "preprocessing:\n"
        f"  chunks_dir: {chunks}\n"
        "embedding:\n"
        f"  cache_dir: {tmp_path / 'embeddings'}\n"
    )
    return cfg


def test_caption_skips_placeholder_clips(tmp_path: Path, monkeypatch):
    from castlerag.preprocess import caption_ocr

    chunks = tmp_path / "chunks"
    path = chunks / "day1" / "Reading" / "08" / "clips.jsonl"
    clips = [
        _clip("Reading", 0, fixed=True).model_copy(update={"clip_caption": None}),
        _clip("Reading", 1, fixed=True).model_copy(
            update={"clip_caption": None, "is_placeholder": True}
        ),
    ]
    write_jsonl_records(clips, path)
    seen = []

    def _annotate(**kw):
        seen.append(kw["clip_id"])
        return SimpleNamespace(clip_caption="NEW", ocr_text=None, scene_graph_text=None)

    monkeypatch.setattr(caption_ocr, "annotate_clip", _annotate)
    monkeypatch.setenv("VLLM_BASE_URL", "http://stub/v1")
    result = CliRunner().invoke(
        app,
        [
            "preprocess",
            "--config",
            str(_cfg_file(tmp_path, chunks)),
            "--day",
            "1",
            "--skip-base",
            "--caption",
            "--camera",
            "Reading",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen == [clips[0].clip_id]  # the placeholder costs no caption call
    assert "1 placeholders skipped" in result.output
    after = load_clip_records(path)
    assert [c.clip_caption for c in after] == ["NEW", None]
    assert after[1].is_placeholder is True


def test_filter_records_leaves_placeholders_out(tmp_path: Path):
    cfg = _cfg(tmp_path, scope="all")
    real = _clip("Reading", 0, fixed=True)
    card = _clip("Reading", 1, fixed=True).model_copy(update={"is_placeholder": True})
    loaded = LoadedArtifacts(transcripts=[], clips=[real, card], events=[], aux=[])
    assert [c.clip_id for c in filter_records(loaded, cfg).clips] == [real.clip_id]


class _FakeQdrant:
    """Stores points as payload dicts and answers count/delete by filter."""

    def __init__(self, points):
        self.points = list(points)

    @staticmethod
    def _cond(p, c):
        v = p.get(c.key)
        m = c.match
        return v in m.any if hasattr(m, "any") and m.any is not None else v == m.value

    def _match(self, p, flt):
        return all(self._cond(p, c) for c in flt.must or []) and not any(
            self._cond(p, c) for c in flt.must_not or []
        )

    def count(self, collection, count_filter, exact=True):
        return SimpleNamespace(
            count=sum(self._match(p, count_filter) for p in self.points)
        )

    def delete(self, collection, points_selector, wait=True):
        self.points = [
            p for p in self.points if not self._match(p, points_selector.filter)
        ]


def _pt(cam, source, rid):
    return {"day": "day1", "camera_id": cam, "source_type": source, "record_id": rid}


def test_prune_removes_placeholder_clips_and_replaced_events(tmp_path: Path):
    cfg = _cfg(tmp_path, scope="all")
    real = _clip("Reading", 0, fixed=True)
    card = _clip("Reading", 1, fixed=True).model_copy(update={"is_placeholder": True})
    records = LoadedArtifacts(
        transcripts=[], clips=[real, card], events=[_event("Reading", "ev_new")], aux=[]
    )
    client = _FakeQdrant(
        [
            _pt("Reading", "main_clip", real.clip_id),
            _pt("Reading", "main_clip", card.clip_id),  # re-flagged: goes
            _pt("Reading", "main_event_summary", "ev_new"),
            _pt("Reading", "main_event_summary", "ev_old"),  # replaced: goes
            _pt("Reading", "transcript_window", "tw1"),
            _pt("Kitchen", "main_clip", "day1_Kitchen_08_0000"),  # no chunks: kept
            _pt("Kitchen", "main_event_summary", "ev_k"),  # no chunks: kept
        ]
    )
    removed = prune_stale_points(client, cfg, records, day=1)
    assert removed == 2
    left = {(p["camera_id"], p["record_id"]) for p in client.points}
    assert ("Reading", card.clip_id) not in left
    assert ("Reading", "ev_old") not in left
    assert {
        ("Reading", real.clip_id),
        ("Reading", "ev_new"),
        ("Reading", "tw1"),
    } <= left
    assert {("Kitchen", "day1_Kitchen_08_0000"), ("Kitchen", "ev_k")} <= left


def _load_reflag():
    spec = importlib.util.spec_from_file_location(
        "reflag_placeholders", REPO / "scripts" / "reflag_placeholders.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reflag_day_unflags_still_rooms_and_flags_cards(tmp_path: Path):
    rf = _load_reflag()
    frames = tmp_path / "frames"
    frames.mkdir()
    still = [str(_save_still_room(frames / f"s{i}.jpg", seed=i)) for i in range(3)]
    card = [str(_save_card(frames / f"c{i}.jpg")) for i in range(3)]
    chunks = tmp_path / "chunks" / "day1" / "Reading" / "09"
    rows = [
        _clip("Reading", 0, fixed=True).model_copy(
            update={"sampled_frame_paths": still, "is_placeholder": True}
        ),
        _clip("Reading", 1, fixed=True).model_copy(
            update={"sampled_frame_paths": card, "is_placeholder": False}
        ),
        _clip("Reading", 2, fixed=True).model_copy(
            update={"sampled_frame_paths": ["/gone.jpg"], "is_placeholder": True}
        ),
    ]
    write_jsonl_records(rows, chunks / "clips.jsonl")
    day = tmp_path / "chunks" / "day1"

    dry = rf.reflag_day(day, apply=False)
    assert (dry["unflagged"], dry["newly_flagged"], dry["no_frames"]) == (1, 1, 1)
    assert dry["files"] == 0

    done = rf.reflag_day(day, apply=True)
    assert done["files"] == 1
    flags = [
        json.loads(line)["is_placeholder"] for line in (chunks / "clips.jsonl").open()
    ]
    assert flags == [False, True, True]  # the frameless clip keeps its flag
    assert (chunks / "clips.jsonl.preflag").exists()
    assert rf.reflag_day(day, apply=True)["files"] == 0  # idempotent


def test_events_rebuild_clears_events_no_longer_possible(tmp_path: Path):
    """Re-flagging can leave an hour with no 4 usable clips: the old events go."""
    chunks = tmp_path / "chunks"
    hour = chunks / "day1" / "Reading" / "08"
    clips = [
        _clip("Reading", i, fixed=True).model_copy(update={"is_placeholder": True})
        for i in range(4)
    ]
    write_jsonl_records(clips, hour / "clips.jsonl")
    write_jsonl_records([_event("Reading", "ev_stale")], hour / "events.jsonl")
    result = CliRunner().invoke(
        app,
        [
            "preprocess",
            "--config",
            str(_cfg_file(tmp_path, chunks)),
            "--day",
            "1",
            "--skip-base",
            "--events",
            "--camera",
            "Reading",
        ],
        env={"VLLM_BASE_URL": "http://stub/v1"},
    )
    assert result.exit_code == 0, result.output
    assert (hour / "events.jsonl").read_text() == ""


def test_index_prunes_a_day_with_nothing_left_to_index(tmp_path: Path, monkeypatch):
    """All clips re-flagged, no events or transcripts: prune instead of failing."""
    import castlerag.cli as cli

    chunks = tmp_path / "chunks"
    card = _clip("Reading", 0, fixed=True).model_copy(update={"is_placeholder": True})
    write_jsonl_records([card], chunks / "day1" / "Reading" / "08" / "clips.jsonl")
    calls = []

    def _prune(cfg, records, day):
        calls.append((day, [c.clip_id for c in records.clips]))
        return 3

    monkeypatch.setattr(cli, "prune_day_without_index", _prune)
    result = CliRunner().invoke(
        app,
        ["index", "--config", str(_cfg_file(tmp_path, chunks)), "--day", "1"],
    )
    assert result.exit_code == 0, result.output
    assert calls == [(1, [card.clip_id])]
    assert "pruned 3 stale points" in result.output
    # lexical indexes rebuilt after pruning, without the placeholder clip
    visual = json.loads((tmp_path / "embeddings" / "visual_text.json").read_text())
    assert card.clip_id not in json.dumps(visual)


def test_index_still_fails_for_a_day_without_chunks(tmp_path: Path):
    chunks = tmp_path / "chunks"
    write_jsonl_records(
        [_clip("Reading", 0, fixed=True)],
        chunks / "day1" / "Reading" / "08" / "clips.jsonl",
    )
    result = CliRunner().invoke(
        app,
        ["index", "--config", str(_cfg_file(tmp_path, chunks)), "--day", "2"],
    )
    assert result.exit_code == 1
    assert "No chunk records found for day 2" in result.output


def test_transcript_bm25_handles_an_empty_corpus(tmp_path: Path):
    from castlerag.index.transcript_lexical import build_bm25_index, load_bm25_index
    from castlerag.retrieval.transcript_lexical import score_windows

    path = tmp_path / "transcripts.pkl"
    built = build_bm25_index([], path)
    assert built.bm25 is None
    loaded = load_bm25_index(path)
    assert score_windows(loaded, [], "what did she say", {}) == []
