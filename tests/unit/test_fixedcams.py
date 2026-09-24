"""Fixed room-camera ingest + retrieval (issue #50 Bug B).

Covers the pieces needed to add the 5 fixed cameras to an existing ego-only
day-1 collection: camera scoping for preprocess, incremental embedding caches,
scope-tolerant cache loading, deterministic/distinct point ids, a participant
filter that does not drop fixed cameras, and the UI padding roster.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from castlerag.cli import app, iter_scoped_clip_paths
from castlerag.config import CastleRAGConfig, load_config
from castlerag.dataset.layout import discover_hours, scoped_cameras
from castlerag.index.io import load_embedding_cache, write_embedding_cache
from castlerag.index.pipeline import (
    build_qdrant_index,
    cache_dense_embeddings,
    load_chunk_records,
    load_dense_caches,
)
from castlerag.index.qdrant import record_to_qdrant_point
from castlerag.rerank.llm_reranker import format_candidate_pack
from castlerag.retrieval.filters import build_filter
from castlerag.retrieval.search import _dense_search
from castlerag.routing.question_router import route_question
from castlerag.schemas import ClipRecord, EvidencePack, RetrievalHit
from castlerag.ui.rag_engine import padding_roster

REPO = Path(__file__).resolve().parents[2]
EGO = ["Allie", "Bjorn"]
EXO = ["Kitchen", "Reading"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _clip(camera_id: str, idx: int = 0, *, fixed: bool = False) -> ClipRecord:
    return ClipRecord(
        clip_id=f"day1_{camera_id}_08_{idx:04d}",
        parent_source_id=f"day1_{camera_id}_08",
        day="day1",
        hour=8,
        camera_id=camera_id,
        camera_type="fixed" if fixed else "ego",
        participant_id=None if fixed else camera_id,
        room=camera_id if fixed else None,
        start_seconds=30.0 * idx,
        end_seconds=30.0 * (idx + 1),
        absolute_start=28_800_000 + 30_000 * idx,
        absolute_end=28_800_000 + 30_000 * (idx + 1),
        source_video_path=f"/data/main/day1/{camera_id}/video/08.mp4",
        clip_caption=f"{camera_id} clip {idx}",
    )


def _cfg(tmp_path: Path, scope: str = "ego") -> CastleRAGConfig:
    return CastleRAGConfig.model_validate(
        {
            "dataset": {
                "ego_cameras": EGO,
                "exo_cameras": EXO,
                "camera_scope": scope,
            },
            "preprocessing": {"chunks_dir": str(tmp_path / "chunks")},
            "embedding": {
                "cache_dir": str(tmp_path / "embeddings"),
                "backend": "transformers",
                "batch_sizes": {
                    "transcript": 2,
                    "event_summary": 2,
                    "image": 2,
                    "video": 2,
                },
            },
            "qdrant": {"collection": "castle_test"},
            "version": "0.1.0",
        }
    )


class _CountingEmbed:
    """Deterministic fake: vector = [len(payload), call_no]."""

    def __init__(self) -> None:
        self.dim = 2
        self.video_calls: list[list[str]] = []

    def embed_texts(self, payloads):
        return np.asarray([[float(len(p)), 0.0] for p in payloads], dtype=np.float32)

    def embed_images(self, payloads):
        return self.embed_texts(payloads)

    def embed_videos(self, payloads):
        self.video_calls.append(list(payloads))
        n = float(len(self.video_calls))
        return np.asarray([[float(len(p)), n] for p in payloads], dtype=np.float32)


def _records(clips: list[ClipRecord]):
    records = load_chunk_records(Path("/nonexistent"))
    records.transcripts, records.events, records.aux = [], [], []
    records.clips = clips
    return records


# ---------------------------------------------------------------------------
# camera scoping (preprocess)
# ---------------------------------------------------------------------------


def test_scoped_cameras_ego_all_and_subset():
    assert scoped_cameras(EGO, EXO, "ego") == EGO
    assert scoped_cameras(EGO, EXO, "all") == EGO + EXO
    # Subset keeps roster order regardless of the order requested.
    assert scoped_cameras(EGO, EXO, "all", only=["Reading", "Kitchen"]) == EXO


def test_scoped_cameras_rejects_fixed_under_ego_scope():
    with pytest.raises(ValueError, match="camera_scope='all'"):
        scoped_cameras(EGO, EXO, "ego", only=["Kitchen"])
    with pytest.raises(ValueError, match="not in scope"):
        scoped_cameras(EGO, EXO, "all", only=["Kitchn"])


def test_discover_hours_cameras_subset_yields_only_fixed(tmp_path: Path):
    for cam in EGO + EXO:
        video_dir = tmp_path / "main" / "day1" / cam / "video"
        video_dir.mkdir(parents=True)
        (video_dir / "08.mp4").touch()
    assets = list(
        discover_hours(
            root=tmp_path,
            ego_cameras=EGO,
            exo_cameras=EXO,
            days=[1],
            hours=[8],
            camera_scope="all",
            cameras=EXO,
        )
    )
    assert [a.camera_id for a in assets] == EXO
    assert all(a.camera_type == "fixed" for a in assets)
    assert [a.room for a in assets] == EXO
    assert all(a.participant_id is None for a in assets)


def test_iter_scoped_clip_paths_skips_other_cameras(tmp_path: Path):
    day_root = tmp_path / "chunks" / "day1"
    for cam in ("Allie", "Kitchen", "Reading"):
        hour_dir = day_root / cam / "08"
        hour_dir.mkdir(parents=True)
        (hour_dir / "clips.jsonl").write_text("")
    paths = list(iter_scoped_clip_paths([day_root], ["Kitchen", "Reading"]))
    assert [p.parent.parent.name for p in paths] == ["Kitchen", "Reading"]
    # Missing day roots are ignored rather than raising.
    assert list(iter_scoped_clip_paths([tmp_path / "nope"], ["Kitchen"])) == []


def test_iter_scoped_clip_paths_filters_hours(tmp_path: Path):
    day_root = tmp_path / "chunks" / "day1"
    for hh in ("08", "09", "10"):
        hour_dir = day_root / "Reading" / hh
        hour_dir.mkdir(parents=True)
        (hour_dir / "clips.jsonl").write_text("")
    paths = list(iter_scoped_clip_paths([day_root], ["Reading"], hours=[8, 10]))
    assert [p.parent.name for p in paths] == ["08", "10"]


def test_preprocess_caption_touches_only_requested_camera_and_hour(
    tmp_path: Path, monkeypatch
):
    """--skip-base --caption --camera X --hour H rewrites only X/H clips.jsonl."""
    from castlerag.index.io import load_clip_records, write_jsonl_records
    from castlerag.preprocess import caption_ocr

    chunks = tmp_path / "chunks"
    for cam, fixed in (("Allie", False), ("Reading", True)):
        for hh in (8, 9):
            write_jsonl_records(
                [_clip(cam, 0, fixed=fixed).model_copy(update={"hour": hh})],
                chunks / "day1" / cam / f"{hh:02d}" / "clips.jsonl",
            )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "dataset:\n"
        f"  ego_cameras: {EGO}\n"
        f"  exo_cameras: {EXO}\n"
        "  camera_scope: all\n"
        "preprocessing:\n"
        f"  chunks_dir: {chunks}\n"
    )
    seen: list[str] = []

    def _fake_annotate(**kwargs):
        seen.append(kwargs["clip_id"])
        return SimpleNamespace(clip_caption="NEW", ocr_text=None, scene_graph_text=None)

    monkeypatch.setattr(caption_ocr, "annotate_clip", _fake_annotate)
    monkeypatch.setenv("VLLM_BASE_URL", "http://stub/v1")
    result = CliRunner().invoke(
        app,
        [
            "preprocess", "--config", str(cfg_path), "--day", "1",
            "--skip-base", "--caption", "--camera", "Reading", "--hour", "9",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(seen) == 1

    def _caption(cam: str, hh: str) -> str:
        path = chunks / "day1" / cam / hh / "clips.jsonl"
        return load_clip_records(path)[0].clip_caption

    assert _caption("Reading", "09") == "NEW"
    assert _caption("Reading", "08") == "Reading clip 0"
    assert _caption("Allie", "08") == "Allie clip 0"
    assert _caption("Allie", "09") == "Allie clip 0"


def test_preprocess_rejects_fixed_camera_under_ego_config():
    result = CliRunner().invoke(
        app, ["preprocess", "--dry-run", "--camera", "Kitchen"]
    )
    assert result.exit_code == 1
    assert "not in scope" in result.output


def test_preprocess_accepts_fixed_cameras_with_fixedcams_config():
    result = CliRunner().invoke(
        app,
        [
            "preprocess",
            "--dry-run",
            "--config",
            str(REPO / "configs" / "snellius_fixedcams.yaml"),
            "--camera",
            "Kitchen",
            "--camera",
            "Reading",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "0 ego + 2 fixed" in result.output


def test_fixedcams_config_matches_snellius_me_except_scope():
    me = load_config(override_path=REPO / "configs" / "snellius_me.yaml")
    fixed = load_config(override_path=REPO / "configs" / "snellius_fixedcams.yaml")
    assert me.dataset.camera_scope == "ego"
    assert fixed.dataset.camera_scope == "all"
    assert fixed.dataset.exo_cameras == [
        "Kitchen",
        "Living1",
        "Living2",
        "Meeting",
        "Reading",
    ]
    # Everything else (paths, collection, endpoints) must be identical so the
    # fixed cameras land in the SAME collection / cache dir / chunks tree.
    me_d = me.model_dump()
    fixed_d = fixed.model_dump()
    me_d["dataset"].pop("camera_scope")
    fixed_d["dataset"].pop("camera_scope")
    assert me_d == fixed_d


# ---------------------------------------------------------------------------
# incremental embedding caches
# ---------------------------------------------------------------------------


def test_cache_appends_new_fixed_records_without_reembedding_ego(tmp_path: Path):
    ego_clips = [_clip("Allie", 0), _clip("Bjorn", 0)]
    fixed_clips = [_clip("Kitchen", 0, fixed=True), _clip("Reading", 0, fixed=True)]
    embed = _CountingEmbed()

    # 1) The existing ego-only ingest.
    cache_dense_embeddings(
        _records(ego_clips), _cfg(tmp_path, "ego"), embed, modality="video", day=1
    )
    cache = tmp_path / "embeddings" / "clips_day1.npz"
    ego_ids, ego_vecs = load_embedding_cache(cache)
    assert ego_ids == [c.clip_id for c in ego_clips]

    # 2) Same chunks tree now also holds fixed clips; re-run under scope=all.
    all_records = _records(ego_clips + fixed_clips)
    cache_dense_embeddings(
        all_records, _cfg(tmp_path, "all"), embed, modality="video", day=1
    )
    ids, vecs = load_embedding_cache(cache)
    assert ids == ego_ids + [c.clip_id for c in fixed_clips]
    # Ego rows are kept verbatim; only the fixed clips hit the embedder.
    np.testing.assert_array_equal(vecs[: len(ego_ids)], ego_vecs)
    assert len(embed.video_calls) == 2
    assert len(embed.video_calls[1]) == len(fixed_clips)

    # 3) A third run is a complete no-op (idempotent).
    cache_dense_embeddings(
        all_records, _cfg(tmp_path, "all"), embed, modality="video", day=1
    )
    assert len(embed.video_calls) == 2
    assert load_embedding_cache(cache)[0] == ids


def test_cache_preserves_out_of_scope_rows(tmp_path: Path):
    """An ego-scope run must never drop fixed rows already in the cache."""
    ego_clips = [_clip("Allie", 0)]
    fixed_clips = [_clip("Kitchen", 0, fixed=True)]
    embed = _CountingEmbed()
    all_records = _records(ego_clips + fixed_clips)
    cache_dense_embeddings(
        all_records, _cfg(tmp_path, "all"), embed, modality="video", day=1
    )
    cache_dense_embeddings(
        all_records, _cfg(tmp_path, "ego"), embed, modality="video", day=1
    )
    ids, _ = load_embedding_cache(tmp_path / "embeddings" / "clips_day1.npz")
    assert set(ids) == {"day1_Allie_08_0000", "day1_Kitchen_08_0000"}


def test_cache_force_reembeds_scope_but_keeps_out_of_scope_rows(tmp_path: Path):
    """``--force`` under the ego config must not delete cached fixed rows."""
    ego_clips = [_clip("Allie", 0)]
    fixed_clips = [_clip("Kitchen", 0, fixed=True)]
    embed = _CountingEmbed()
    cache_dense_embeddings(
        _records(ego_clips + fixed_clips),
        _cfg(tmp_path, "all"),
        embed,
        modality="video",
        day=1,
    )
    cache = tmp_path / "embeddings" / "clips_day1.npz"
    _, before = load_embedding_cache(cache)
    cache_dense_embeddings(
        _records(ego_clips + fixed_clips),
        _cfg(tmp_path, "ego"),
        embed,
        modality="video",
        day=1,
        force=True,
    )
    ids, vecs = load_embedding_cache(cache)
    assert sorted(ids) == ["day1_Allie_08_0000", "day1_Kitchen_08_0000"]
    # The ego row was re-embedded (second call), the fixed row kept verbatim.
    assert len(embed.video_calls[-1]) == 1
    rows = dict(zip(ids, vecs))
    assert rows["day1_Allie_08_0000"][1] == 2.0
    np.testing.assert_array_equal(rows["day1_Kitchen_08_0000"], before[1])


def test_cache_force_prunes_stale_ids(tmp_path: Path):
    """``--force`` stays the repair path: ids with no chunk record are dropped."""
    cache = tmp_path / "embeddings" / "clips_day1.npz"
    write_embedding_cache(
        ["day1_Allie_08_9999", "day1_Kitchen_08_0000"],
        np.zeros((2, 2), dtype=np.float32),
        cache,
    )
    ego_clips = [_clip("Allie", 0)]
    fixed_clips = [_clip("Kitchen", 0, fixed=True)]
    cache_dense_embeddings(
        _records(ego_clips + fixed_clips),
        _cfg(tmp_path, "ego"),
        _CountingEmbed(),
        modality="video",
        day=1,
        force=True,
    )
    ids, _ = load_embedding_cache(cache)
    # Stale ego id gone, live out-of-scope fixed row kept, in-scope re-embedded.
    assert sorted(ids) == ["day1_Allie_08_0000", "day1_Kitchen_08_0000"]


def test_write_embedding_cache_is_atomic(tmp_path: Path, monkeypatch):
    """A crash mid-write must leave the previous good cache untouched."""
    cache = tmp_path / "clips_day1.npz"
    write_embedding_cache(["a"], np.ones((1, 2), dtype=np.float32), cache)

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", _boom)
    with pytest.raises(OSError):
        write_embedding_cache(["a", "b"], np.ones((2, 2), dtype=np.float32), cache)
    assert load_embedding_cache(cache)[0] == ["a"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clips_day1.npz"]


def test_cache_rejects_dim_mismatch(tmp_path: Path):
    cache_dir = tmp_path / "embeddings"
    write_embedding_cache(
        ["day1_Allie_08_0000"],
        np.zeros((1, 5), dtype=np.float32),
        cache_dir / "clips_day1.npz",
    )
    records = _records([_clip("Allie", 0), _clip("Kitchen", 0, fixed=True)])
    with pytest.raises(ValueError, match="dim mismatch"):
        cache_dense_embeddings(
            records, _cfg(tmp_path, "all"), _CountingEmbed(), modality="video", day=1
        )


# ---------------------------------------------------------------------------
# indexing: scope-tolerant loads + additive upsert
# ---------------------------------------------------------------------------


def _capture_index(monkeypatch) -> list[dict]:
    upserts: list[dict] = []
    monkeypatch.setattr(
        "castlerag.index.pipeline.bootstrap_collection",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "castlerag.index.pipeline.upsert_batch",
        lambda **kwargs: upserts.append(kwargs),
    )
    return upserts


def test_index_scope_all_upserts_fixed_with_room(tmp_path: Path, monkeypatch):
    clips = [_clip("Allie", 0), _clip("Kitchen", 0, fixed=True)]
    records = _records(clips)
    cfg = _cfg(tmp_path, "all")
    cache_dense_embeddings(records, cfg, _CountingEmbed(), modality="video", day=1)
    upserts = _capture_index(monkeypatch)

    build_qdrant_index(cfg, records, day=1)
    payloads = [p for batch in upserts for p in batch["payloads"]]
    by_cam = {p["camera_id"]: p for p in payloads}
    assert set(by_cam) == {"Allie", "Kitchen"}
    assert by_cam["Kitchen"]["camera_type"] == "fixed"
    assert by_cam["Kitchen"]["room"] == "Kitchen"
    assert "participant_id" not in by_cam["Kitchen"]  # None -> excluded
    assert by_cam["Allie"]["camera_type"] == "ego"
    assert "room" not in by_cam["Allie"]


def test_index_ego_scope_ignores_fixed_rows_in_cache(tmp_path: Path, monkeypatch):
    """After the fixed ingest, an ego-scope `index --day 1` must not KeyError."""
    clips = [_clip("Allie", 0), _clip("Kitchen", 0, fixed=True)]
    records = _records(clips)
    cache_dense_embeddings(
        records, _cfg(tmp_path, "all"), _CountingEmbed(), modality="video", day=1
    )
    upserts = _capture_index(monkeypatch)

    build_qdrant_index(_cfg(tmp_path, "ego"), records, day=1)
    payloads = [p for batch in upserts for p in batch["payloads"]]
    assert [p["camera_id"] for p in payloads] == ["Allie"]


def test_load_dense_caches_still_flags_stale_ids(tmp_path: Path):
    cache_dir = tmp_path / "embeddings"
    write_embedding_cache(
        ["ghost_clip"], np.zeros((1, 2), dtype=np.float32), cache_dir / "clips.npz"
    )
    records = _records([_clip("Allie", 0)])
    with pytest.raises(KeyError, match="ghost_clip"):
        load_dense_caches(cache_dir, records, scope=records)


def test_point_ids_stable_for_ego_and_distinct_for_fixed():
    ego = record_to_qdrant_point(_clip("Allie", 0), model_version="0.1.0")
    ego_again = record_to_qdrant_point(_clip("Allie", 0), model_version="0.1.0")
    fixed = record_to_qdrant_point(_clip("Kitchen", 0, fixed=True), "0.1.0")
    assert ego.point_id == ego_again.point_id  # re-upsert overwrites in place
    assert fixed.point_id != ego.point_id


# ---------------------------------------------------------------------------
# retrieval: participant hint must not drop fixed cameras
# ---------------------------------------------------------------------------


def test_build_filter_participant_includes_fixed_uses_should():
    f = build_filter(
        participant_id="Werner",
        source_type="main_clip",
        participant_includes_fixed=True,
    )
    assert [c.key for c in f.must] == ["source_type"]
    should = {(c.key, c.match.value) for c in f.should}
    assert should == {("participant_id", "Werner"), ("camera_type", "fixed")}


def test_build_filter_participant_only_with_fixed_is_not_none():
    f = build_filter(participant_id="Werner", participant_includes_fixed=True)
    assert f is not None
    assert f.must is None
    assert len(f.should) == 2


def test_dense_search_admits_fixed_cameras_for_participant_hint():
    captured: dict = {}

    class _Client:
        def query_points(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(points=[])

    _dense_search(
        qdrant_client=_Client(),
        collection_name="c",
        query_vector=[0.0, 1.0],
        limit=5,
        source_type="main_clip",
        modality="video",
        participant_id="Werner",
    )
    flt = captured["query_filter"]
    assert "participant_id" not in [c.key for c in flt.must]
    assert ("camera_type", "fixed") in {(c.key, c.match.value) for c in flt.should}


def test_router_maps_reading_area_to_reading_room():
    hints = route_question(
        "What family name is on the coat of arms in the reading area?",
        {"a": "x", "b": "y", "c": "z", "d": "w"},
    )
    assert hints.room == "Reading"


# ---------------------------------------------------------------------------
# rerank prompt + UI roster
# ---------------------------------------------------------------------------


def _fixed_hit() -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        score=0.9,
        point_id="pt",
        record_id="day1_Kitchen_08_0000",
        source_type="main_clip",
        modality="video",
        day="day1",
        camera_id="Kitchen",
        room="Kitchen",
        absolute_start=28_800_000,
        absolute_end=28_830_000,
    )


def test_reranker_pack_names_room_for_fixed_camera():
    hit = _fixed_hit()
    pack = EvidencePack(
        pack_id="p1",
        route="static_visual",
        primary_hit=hit,
        retrieval_score=0.9,
        evidence_rows=[hit],
    )
    text = format_candidate_pack(pack, rank=1)
    assert "Room: Kitchen (fixed room camera)" in text
    assert "Participant: N/A" in text


def test_reranker_pack_has_no_room_line_for_ego_camera():
    hit = _fixed_hit().model_copy(
        update={"camera_id": "Allie", "room": None, "participant_id": "Allie"}
    )
    pack = EvidencePack(
        pack_id="p1",
        route="static_visual",
        primary_hit=hit,
        retrieval_score=0.9,
        evidence_rows=[hit],
    )
    assert "Room:" not in format_candidate_pack(pack, rank=1)


def test_padding_roster_adds_fixed_cameras_only_in_all_scope():
    ego_ds = SimpleNamespace(ego_cameras=EGO, exo_cameras=EXO, camera_scope="ego")
    all_ds = SimpleNamespace(ego_cameras=EGO, exo_cameras=EXO, camera_scope="all")
    assert padding_roster(ego_ds) == tuple(EGO)
    assert padding_roster(all_ds) == tuple(EGO + EXO)
    # Duck-typed configs without the newer fields keep working.
    assert padding_roster(SimpleNamespace(ego_cameras=())) == ()


# ---------------------------------------------------------------------------
# participant hints naming someone outside the indexed ego roster
# ---------------------------------------------------------------------------


def _run_retrieve(question_text: str, known_participants, windows=()):
    from castlerag.retrieval.search import retrieve
    from castlerag.schemas import EvalQuestion

    calls: list = []

    class _Client:
        def query_points(self, **kwargs):
            calls.append(kwargs["query_filter"])
            return SimpleNamespace(points=[])

    class _BM25:
        def get_scores(self, tokens):
            return np.zeros(len(windows), dtype=np.float32)

    class _Embed:
        def embed_texts(self, texts):
            return np.ones((len(texts), 2), dtype=np.float32)

    question = EvalQuestion(
        question_id="q",
        query=question_text,
        answers={"a": "x", "b": "y", "c": "z", "d": "w"},
    )
    retrieval_cfg = SimpleNamespace(
        transcript_top_k=5,
        event_summary_top_k=5,
        video_top_k=5,
        photo_top_k=5,
        aux_video_top_k=5,
        heartrate_top_k=5,
        gaze_top_k=5,
        thermal_top_k=5,
        rrf_k=60,
        max_candidate_videos=4,
        frames_per_candidate=8,
        max_aux_images=4,
        max_evidence_rows=10,
        modality_score_thresholds={},
    )
    retrieve(
        question=question,
        hints=route_question(question.query, question.answers),
        qdrant_client=_Client(),
        collection_name="c",
        bm25_index=SimpleNamespace(bm25=_BM25(), windows=list(windows)),
        embed_client=_Embed(),
        retrieval_cfg=retrieval_cfg,
        known_participants=known_participants,
    )
    return calls


def _participant_values(flt) -> set:
    conds = list(flt.must or []) + list(flt.should or [])
    return {c.match.value for c in conds if c.key == "participant_id"}


def test_retrieve_drops_participant_filter_for_unindexed_name():
    # Bao is a router participant but has no day-1 ego stream in the roster.
    calls = _run_retrieve("What did Bao cook?", known_participants=["Allie", "Werner"])
    assert calls
    assert all(not _participant_values(f) for f in calls)


def test_retrieve_keeps_soft_participant_filter_for_indexed_name():
    calls = _run_retrieve(
        "What organisation's logo is on Werner's apron?",
        known_participants=["Allie", "Werner"],
    )
    assert calls
    for flt in calls:
        assert _participant_values(flt) == {"Werner"}
        # ...but OR-ed with the fixed cameras, never a hard `must`.
        assert "participant_id" not in [c.key for c in (flt.must or [])]
        assert ("camera_type", "fixed") in {
            (c.key, c.match.value) for c in (flt.should or [])
        }


def test_retrieve_without_roster_keeps_hint():
    calls = _run_retrieve("What did Bao cook?", known_participants=None)
    assert calls
    assert all(_participant_values(f) == {"Bao"} for f in calls)


def _window(participant: str, day: str = "day1"):
    return SimpleNamespace(
        participant_id=participant,
        day=day,
        camera_id=participant,
        hour=8,
        room=None,
        transcript_text="",
        transcript_window_id=f"{day}_{participant}_08_w0",
        absolute_start=0,
        absolute_end=1,
    )


def test_retrieve_uses_indexed_windows_over_config_roster():
    # Bao is in the configured roster but has no indexed day-1 windows.
    calls = _run_retrieve(
        "On day 1, what did Bao cook?",
        known_participants=["Allie", "Bao", "Werner"],
        windows=[_window("Allie"), _window("Werner"), _window("Bao", "day2")],
    )
    assert calls
    assert all(not _participant_values(f) for f in calls)


def test_retrieve_keeps_hint_for_participant_with_indexed_windows():
    calls = _run_retrieve(
        "On day 1, what organisation's logo is on Werner's apron?",
        known_participants=["Allie"],  # stale roster is ignored
        windows=[_window("Werner")],
    )
    assert calls
    assert all(_participant_values(f) == {"Werner"} for f in calls)


def test_retrieve_drops_hint_for_camera_not_yet_ingested():
    # A --camera scoped ingest indexed only Allie so far.
    calls = _run_retrieve(
        "What organisation's logo is on Werner's apron?",
        known_participants=["Allie", "Werner"],
        windows=[_window("Allie")],
    )
    assert calls
    assert all(not _participant_values(f) for f in calls)


def test_participant_or_fixed_filter_is_binding_in_qdrant():
    """A top-level ``should`` next to ``must`` is mandatory in Qdrant, not a boost.

    Runs the real filter through qdrant-client's in-memory engine: another
    participant's ego clip on the same day must be excluded, while the named
    participant's clip and the fixed-camera clip both pass.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    client = QdrantClient(":memory:")
    client.create_collection(
        "t", vectors_config=qm.VectorParams(size=2, distance=qm.Distance.COSINE)
    )
    rows = [
        ("Werner", "ego", "Werner"),
        ("Allie", "ego", "Allie"),
        (None, "fixed", "Kitchen"),
    ]
    client.upsert(
        "t",
        [
            qm.PointStruct(
                id=i,
                vector=[1.0, 0.0],
                payload={
                    "participant_id": participant,
                    "camera_type": ctype,
                    "camera_id": cam,
                    "day": "day1",
                    "source_type": "main_clip",
                },
            )
            for i, (participant, ctype, cam) in enumerate(rows)
        ],
    )
    flt = build_filter(
        day="day1",
        source_type="main_clip",
        participant_id="Werner",
        participant_includes_fixed=True,
    )
    hits = client.query_points("t", query=[1.0, 0.0], query_filter=flt, limit=10).points
    assert sorted(h.payload["camera_id"] for h in hits) == ["Kitchen", "Werner"]
