"""Tests for the visual-text BM25 lane (captions + OCR + scene graphs, #50)."""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from castlerag.cli import app
from castlerag.config import CastleRAGConfig
from castlerag.eval.run_eval import (
    IndexArtifactReport,
    PipelineDependencyError,
    _load_optional_visual_index,
)
from castlerag.index.io import write_jsonl_records
from castlerag.index.pipeline import LoadedArtifacts, build_visual_bm25_artifact
from castlerag.index.visual_lexical import (
    VISUAL_TEXT_INDEX_NAME,
    build_visual_bm25_index,
    build_visual_docs,
    load_visual_bm25_index,
    load_visual_bm25_index_if_present,
)
from castlerag.retrieval.search import retrieve
from castlerag.retrieval.visual_lexical import (
    DEFAULT_VISUAL_TEXT_ROUTE_WEIGHTS,
    DEFAULT_VISUAL_TEXT_TOP_K,
    score_visual_docs,
    visual_lane_top_k,
    visual_lane_weight,
)
from castlerag.routing.question_router import route_question
from castlerag.schemas import (
    ClipRecord,
    EvalQuestion,
    EventSummaryRecord,
    TranscriptWindow,
)

run_eval_module = importlib.import_module("castlerag.eval.run_eval")

_BASE = 1_672_531_200_000


def _clip(
    clip_id: str,
    *,
    camera_id: str = "Allie",
    caption: str | None = None,
    ocr: str | None = None,
    scene_graph: str | None = None,
    transcript: str | None = None,
    offset_ms: int = 0,
    day: str = "day1",
    room: str | None = None,
) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        parent_source_id=f"vid_{camera_id}",
        day=day,
        hour=8,
        camera_id=camera_id,
        camera_type="ego",
        participant_id=camera_id,
        room=room,
        start_seconds=float(offset_ms // 1000),
        end_seconds=float(offset_ms // 1000 + 30),
        absolute_start=_BASE + offset_ms,
        absolute_end=_BASE + offset_ms + 30_000,
        source_video_path=f"/data/main/{day}/{camera_id}/video/08.mp4",
        retrieval_clip_path=f"/data/derived/clips/{clip_id}.mp4",
        sampled_frame_paths=[f"/tmp/{clip_id}_0.jpg"],
        transcript_text=transcript,
        clip_caption=caption,
        ocr_text=ocr,
        scene_graph_text=scene_graph,
    )


def _event(event_id: str, summary: str, ocr: str | None = None) -> EventSummaryRecord:
    return EventSummaryRecord(
        event_summary_id=event_id,
        day="day1",
        camera_id="Allie",
        camera_type="ego",
        participant_id="Allie",
        absolute_start=_BASE,
        absolute_end=_BASE + 120_000,
        member_clip_ids=["clip_fridge"],
        event_summary=summary,
        aggregated_ocr_text=ocr,
    )


def _corpus() -> tuple[list[ClipRecord], list[EventSummaryRecord]]:
    clips = [
        # The clip we want: the brand is only visible on screen (OCR), and the
        # transcript is about something else entirely.
        _clip(
            "clip_fridge",
            caption="A person opens a large silver fridge in the kitchen.",
            ocr="SAMSUNG",
            scene_graph="person - opens - refrigerator",
            transcript="so anyway I told him we would meet at nine",
            room="Kitchen",
        ),
        # Distractor: another appliance with a different brand on screen.
        _clip(
            "clip_bosch",
            camera_id="Bjorn",
            caption="Someone loads plates into a dishwasher covered in magnets.",
            ocr="BOSCH",
            offset_ms=60_000,
            room="Kitchen",
        ),
        # Pure transcript clip with no visual text — must not enter the index.
        _clip(
            "clip_speech_only",
            camera_id="Cathal",
            transcript="Samsung Samsung Samsung the fridge brand is Samsung",
            offset_ms=120_000,
        ),
    ]
    events = [
        _event(
            "evt_kitchen",
            "Allie prepares breakfast and takes milk from the fridge.",
            ocr="SAMSUNG | MILK",
        ),
        _event("evt_empty", "", None),
    ]
    return clips, events


def _fridge_question() -> EvalQuestion:
    return EvalQuestion(
        question_id="q_fridge",
        query="What brand is the fridge in the kitchen?",
        answers={"a": "Samsung", "b": "Whirlpool", "c": "Bosch", "d": "Liebherr"},
    )


# ---------------------------------------------------------------------------
# Index build / load
# ---------------------------------------------------------------------------


def test_build_visual_docs_skips_records_without_visual_text():
    clips, events = _corpus()
    docs = build_visual_docs(clips, events)
    ids = [doc.record_id for doc in docs]
    assert ids == ["clip_fridge", "clip_bosch", "evt_kitchen"]
    fridge = docs[0]
    # Indexed surface = caption + OCR + scene graph, never the transcript.
    assert "SAMSUNG" in fridge.text
    assert "refrigerator" in fridge.text
    assert "meet at nine" not in fridge.text
    # ...but the transcript is still carried for downstream evidence rows.
    assert fridge.transcript_text == "so anyway I told him we would meet at nine"
    assert fridge.source_type == "main_clip"
    assert fridge.modality == "video"
    assert fridge.asset_path == "/data/derived/clips/clip_fridge.mp4"
    event = docs[2]
    assert event.source_type == "main_event_summary"
    assert event.modality == "text"
    assert "MILK" in event.text


def test_visual_bm25_index_roundtrip(tmp_path: Path):
    clips, events = _corpus()
    index_path = tmp_path / VISUAL_TEXT_INDEX_NAME
    bundle = build_visual_bm25_index(clips, events, index_path)
    assert index_path.exists()
    assert len(bundle.docs) == 3
    scores = bundle.bm25.get_scores(["samsung"])
    assert scores[0] > scores[1]  # clip_fridge beats clip_bosch

    loaded = load_visual_bm25_index(index_path)
    assert [doc.record_id for doc in loaded.docs] == [
        doc.record_id for doc in bundle.docs
    ]
    assert loaded.tokenized_corpus == bundle.tokenized_corpus
    np.testing.assert_allclose(
        loaded.bm25.get_scores(["samsung"]), scores, rtol=1e-6
    )


def test_visual_bm25_index_empty_corpus_roundtrip(tmp_path: Path):
    index_path = tmp_path / VISUAL_TEXT_INDEX_NAME
    bundle = build_visual_bm25_index([], [], index_path)
    assert bundle.docs == [] and bundle.bm25 is None
    loaded = load_visual_bm25_index(index_path)
    assert loaded.docs == [] and loaded.bm25 is None
    assert score_visual_docs(loaded, "what brand", {"a": "Samsung"}) == []


def test_build_visual_bm25_artifact_writes_next_to_transcript_pickle(tmp_path: Path):
    clips, events = _corpus()
    records = LoadedArtifacts(transcripts=[], clips=clips, events=events, aux=[])
    out = build_visual_bm25_artifact(records, tmp_path / "embeddings")
    assert out == tmp_path / "embeddings" / "visual_text.pkl"
    assert len(load_visual_bm25_index(out).docs) == 3


def test_load_visual_index_if_present_is_optional(tmp_path: Path):
    assert load_visual_bm25_index_if_present(tmp_path) is None
    clips, events = _corpus()
    build_visual_bm25_index(clips, events, tmp_path / VISUAL_TEXT_INDEX_NAME)
    bundle = load_visual_bm25_index_if_present(tmp_path)
    assert bundle is not None
    assert len(bundle.docs) == 3


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def test_score_visual_docs_ranks_ocr_brand_match_first(tmp_path: Path):
    clips, events = _corpus()
    bundle = build_visual_bm25_index(clips, events, tmp_path / VISUAL_TEXT_INDEX_NAME)
    q = _fridge_question()
    hits = score_visual_docs(
        bundle,
        query=q.query,
        choices=q.answers,
        day_hint="day1",
        room_hint="Kitchen",
        top_k=10,
    )
    assert hits, "expected lexical hits"
    assert hits[0].record_id == "clip_fridge"
    assert hits[0].source_type == "main_clip"
    assert hits[0].modality == "video"
    assert hits[0].point_id == "visual_lexical:clip_fridge"
    assert hits[0].ocr_text == "SAMSUNG"
    assert hits[0].asset_path == "/data/derived/clips/clip_fridge.mp4"
    assert hits[0].sampled_frame_paths == ["/tmp/clip_fridge_0.jpg"]
    # The transcript-only clip never enters this lane, however often it says
    # "Samsung": that is what the transcript lane is for.
    assert all(hit.record_id != "clip_speech_only" for hit in hits)
    # Event summaries surface too, as their own source type.
    event_hits = [h for h in hits if h.source_type == "main_event_summary"]
    assert event_hits and event_hits[0].record_id == "evt_kitchen"
    assert event_hits[0].modality == "text"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_score_visual_docs_answer_option_overlap_breaks_caption_ties(tmp_path: Path):
    # Same caption on both clips; only the on-screen text differs. The clip
    # whose OCR carries one of the answer options must win.
    clips = [
        _clip("c_lg", camera_id="Bjorn", caption="a person opens the fridge", ocr="LG"),
        _clip(
            "c_samsung",
            caption="a person opens the fridge",
            ocr="SAMSUNG",
            offset_ms=60_000,  # later start: loses the tie-break without OCR
        ),
    ]
    bundle = build_visual_bm25_index(clips, [], tmp_path / VISUAL_TEXT_INDEX_NAME)
    q = _fridge_question()
    hits = score_visual_docs(bundle, q.query, q.answers)
    assert [h.record_id for h in hits] == ["c_samsung", "c_lg"]


def test_score_visual_docs_applies_metadata_bonuses(tmp_path: Path):
    clips = [
        _clip("c_day1", caption="a red mug on the table", day="day1"),
        _clip(
            "c_day2",
            camera_id="Bjorn",
            caption="a red mug on the table",
            day="day2",
            offset_ms=86_400_000,
        ),
    ]
    bundle = build_visual_bm25_index(clips, [], tmp_path / VISUAL_TEXT_INDEX_NAME)
    choices = {"a": "red", "b": "blue", "c": "green", "d": "black"}
    by_day = score_visual_docs(
        bundle, "what colour is the mug", choices, day_hint="day2"
    )
    assert by_day[0].record_id == "c_day2"
    by_person = score_visual_docs(
        bundle, "what colour is the mug", choices, person_hint="Allie"
    )
    assert by_person[0].record_id == "c_day1"


def test_score_visual_docs_handles_empty_query_and_top_k(tmp_path: Path):
    clips, events = _corpus()
    bundle = build_visual_bm25_index(clips, events, tmp_path / VISUAL_TEXT_INDEX_NAME)
    assert score_visual_docs(bundle, "???", {"a": "x"}) == []
    assert score_visual_docs(bundle, "fridge", {"a": "x"}, top_k=0) == []
    assert len(score_visual_docs(bundle, "fridge kitchen", {"a": "x"}, top_k=1)) == 1


def test_visual_lane_config_helpers_fall_back_to_defaults():
    legacy_cfg = SimpleNamespace(rrf_k=60)  # predates the lane
    assert visual_lane_top_k(legacy_cfg) == DEFAULT_VISUAL_TEXT_TOP_K
    for route, weight in DEFAULT_VISUAL_TEXT_ROUTE_WEIGHTS.items():
        assert visual_lane_weight(route, legacy_cfg) == weight
    assert visual_lane_weight("unknown_route", legacy_cfg) == 1.0

    cfg = CastleRAGConfig().retrieval
    assert visual_lane_top_k(cfg) == 20
    assert visual_lane_weight("static_visual", cfg) == 2.0
    assert visual_lane_weight("speech_text", cfg) == 0.5

    custom = SimpleNamespace(
        visual_text_top_k=7, visual_text_route_weights={"speech_text": 0.1}
    )
    assert visual_lane_top_k(custom) == 7
    assert visual_lane_weight("speech_text", custom) == 0.1
    # Routes missing from the override still use the built-in default.
    assert visual_lane_weight("mixed", custom) == 1.5


# ---------------------------------------------------------------------------
# Fusion in retrieve()
# ---------------------------------------------------------------------------


class _FakeBM25:
    def __init__(self, n: int) -> None:
        self.n = n

    def get_scores(self, tokens: list[str]) -> np.ndarray:
        return np.zeros(self.n, dtype=np.float32)


class _FakeEmbedClient:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0]] * len(texts), dtype=np.float32)


class _FakePoint:
    def __init__(self, pid: str, score: float, payload: dict) -> None:
        self.id = pid
        self.score = score
        self.payload = payload


class _FakeQdrant:
    """Dense lanes only ever return one unrelated clip."""

    def query_points(self, **kwargs):
        source_type = next(
            c.match.value for c in kwargs["query_filter"].must if c.key == "source_type"
        )
        if source_type != "main_clip":
            return SimpleNamespace(points=[])
        return SimpleNamespace(
            points=[
                _FakePoint(
                    "pt_dense",
                    0.8,
                    {
                        "record_id": "clip_dense_only",
                        "source_type": "main_clip",
                        "modality": "video",
                        "day": "day1",
                        "camera_id": "Allie",
                        "participant_id": "Allie",
                        "absolute_start": _BASE + 300_000,
                        "absolute_end": _BASE + 330_000,
                    },
                )
            ]
        )


def _windows() -> list[TranscriptWindow]:
    return [
        TranscriptWindow(
            transcript_window_id="tw_1",
            day="day1",
            camera_id="Allie",
            camera_type="ego",
            participant_id="Allie",
            room="Kitchen",
            hour=8,
            transcript_text="so anyway I told him we would meet at nine",
            absolute_start=_BASE,
            absolute_end=_BASE + 15_000,
        )
    ]


def _legacy_retrieval_cfg() -> SimpleNamespace:
    # Deliberately has no visual_text_* keys: retrieve() must cope.
    return SimpleNamespace(
        transcript_top_k=30,
        event_summary_top_k=20,
        video_top_k=20,
        photo_top_k=16,
        aux_video_top_k=8,
        heartrate_top_k=8,
        gaze_top_k=8,
        thermal_top_k=8,
        rrf_k=60,
        max_candidate_videos=4,
        frames_per_candidate=32,
        max_aux_images=16,
        max_evidence_rows=50,
        modality_score_thresholds={},
    )


def _run_retrieve(**extra):
    q = _fridge_question()
    hints = route_question(q.query, q.answers)
    return retrieve(
        question=q,
        hints=hints,
        qdrant_client=_FakeQdrant(),
        collection_name="castle_test",
        bm25_index=SimpleNamespace(bm25=_FakeBM25(1), windows=_windows()),
        embed_client=_FakeEmbedClient(),
        retrieval_cfg=_legacy_retrieval_cfg(),
        **extra,
    )


def test_retrieve_without_visual_index_is_unchanged(tmp_path: Path):
    baseline = _run_retrieve()
    assert [h.record_id for h in baseline] == ["clip_dense_only", "tw_1"]
    assert _run_retrieve(visual_index=None) == baseline
    empty = build_visual_bm25_index([], [], tmp_path / VISUAL_TEXT_INDEX_NAME)
    assert _run_retrieve(visual_index=empty) == baseline


def test_retrieve_fuses_visual_lexical_lane(tmp_path: Path):
    clips, events = _corpus()
    bundle = build_visual_bm25_index(clips, events, tmp_path / VISUAL_TEXT_INDEX_NAME)
    hits = _run_retrieve(visual_index=bundle)
    ids = [h.record_id for h in hits]
    # The OCR-matched clip is now retrievable at all — and outranks the
    # unrelated dense-only clip on a visual route (lane weight 2.0 vs 1.0/0.7).
    assert "clip_fridge" in ids
    assert ids.index("clip_fridge") < ids.index("clip_dense_only")
    fridge = hits[ids.index("clip_fridge")]
    assert fridge.source_type == "main_clip"
    assert fridge.ocr_text == "SAMSUNG"
    assert "evt_kitchen" in ids
    assert "clip_speech_only" not in ids


def test_retrieve_visual_lane_respects_camera_exclusions(tmp_path: Path):
    clips, events = _corpus()
    bundle = build_visual_bm25_index(clips, events, tmp_path / VISUAL_TEXT_INDEX_NAME)
    q = _fridge_question()
    hints = route_question(q.query, q.answers)
    hints.exclude_cameras = ("Bjorn",)
    hits = retrieve(
        question=q,
        hints=hints,
        qdrant_client=_FakeQdrant(),
        collection_name="castle_test",
        bm25_index=SimpleNamespace(bm25=_FakeBM25(1), windows=_windows()),
        embed_client=_FakeEmbedClient(),
        retrieval_cfg=_legacy_retrieval_cfg(),
        visual_index=bundle,
    )
    ids = [h.record_id for h in hits]
    assert "clip_fridge" in ids
    assert "clip_bosch" not in ids  # Bjorn's clip dropped from the lexical lane


def test_retrieve_visual_lane_honours_config_top_k(tmp_path: Path):
    clips, events = _corpus()
    bundle = build_visual_bm25_index(clips, events, tmp_path / VISUAL_TEXT_INDEX_NAME)
    q = _fridge_question()
    hints = route_question(q.query, q.answers)
    cfg = _legacy_retrieval_cfg()
    cfg.visual_text_top_k = 1
    cfg.visual_text_route_weights = {}
    hits = retrieve(
        question=q,
        hints=hints,
        qdrant_client=_FakeQdrant(),
        collection_name="castle_test",
        bm25_index=SimpleNamespace(bm25=_FakeBM25(1), windows=_windows()),
        embed_client=_FakeEmbedClient(),
        retrieval_cfg=cfg,
        visual_index=bundle,
    )
    lexical_ids = {
        h.record_id for h in hits if h.point_id.startswith("visual_lexical:")
    }
    assert lexical_ids == {"clip_fridge"}


# ---------------------------------------------------------------------------
# Optional loading in run_eval and the CLI
# ---------------------------------------------------------------------------


def _report(tmp_path: Path) -> IndexArtifactReport:
    return IndexArtifactReport(
        bm25_path=tmp_path / "transcripts.pkl",
        chunks_dir=tmp_path / "chunks",
        cache_dir=tmp_path,
        chunk_files={"transcripts": [], "clips": [], "events": [], "aux": []},
        embedding_caches={},
    )


def test_run_eval_loads_visual_index_only_when_present(tmp_path: Path):
    cfg = CastleRAGConfig()
    cfg.embedding.cache_dir = str(tmp_path)
    assert _load_optional_visual_index(cfg, _report(tmp_path)) is None
    clips, events = _corpus()
    build_visual_bm25_index(clips, events, tmp_path / VISUAL_TEXT_INDEX_NAME)
    bundle = _load_optional_visual_index(cfg, _report(tmp_path))
    assert bundle is not None and len(bundle.docs) == 3


def test_run_eval_reports_corrupt_visual_index(tmp_path: Path):
    cfg = CastleRAGConfig()
    cfg.embedding.cache_dir = str(tmp_path)
    (tmp_path / VISUAL_TEXT_INDEX_NAME).write_bytes(b"not a pickle")
    with pytest.raises(PipelineDependencyError, match="visual-text BM25 index"):
        _load_optional_visual_index(cfg, _report(tmp_path))


def test_build_default_pipeline_passes_visual_index_to_retrieve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = CastleRAGConfig()
    cfg.embedding.cache_dir = str(tmp_path)
    clips, events = _corpus()
    build_visual_bm25_index(clips, events, tmp_path / VISUAL_TEXT_INDEX_NAME)
    seen: dict = {}

    monkeypatch.setattr(
        run_eval_module,
        "_prepare_default_runtime",
        lambda c: (object(), object(), _report(tmp_path)),
    )
    monkeypatch.setattr(run_eval_module, "_build_vllm_chat_client", lambda: object())
    monkeypatch.setattr(run_eval_module, "OmniEmbedClient", lambda **kw: object())

    def _fake_retrieve(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(run_eval_module, "retrieve_evidence", _fake_retrieve)
    pipeline = run_eval_module._build_default_pipeline(cfg)
    q = _fridge_question()
    pipeline.retrieve(q, route_question(q.query, q.answers))
    assert seen["visual_index"] is not None
    assert len(seen["visual_index"].docs) == 3


def test_cli_index_lexical_only_builds_both_pickles(tmp_path: Path):
    chunks = tmp_path / "chunks" / "day1"
    write_jsonl_records(_windows(), chunks / "transcripts.jsonl")
    clips, events = _corpus()
    write_jsonl_records(clips, chunks / "clips.jsonl")
    write_jsonl_records(events, chunks / "events.jsonl")
    cache_dir = tmp_path / "embeddings"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "preprocessing:\n"
        f"  chunks_dir: {tmp_path / 'chunks'}\n"
        "embedding:\n"
        f"  cache_dir: {cache_dir}\n"
        "dataset:\n"
        "  camera_scope: all\n"
    )
    result = CliRunner().invoke(
        app, ["index", "--lexical-only", "--config", str(cfg_path)]
    )
    assert result.exit_code == 0, result.output
    assert "skipped (--lexical-only)" in result.output
    assert (cache_dir / "transcripts.pkl").exists()
    visual = load_visual_bm25_index(cache_dir / VISUAL_TEXT_INDEX_NAME)
    assert [d.record_id for d in visual.docs] == [
        "clip_fridge",
        "clip_bosch",
        "evt_kitchen",
    ]
    with (cache_dir / "transcripts.pkl").open("rb") as fh:
        assert len(pickle.load(fh)["windows"]) == 1


# ---------------------------------------------------------------------------
# review follow-ups (#63): exclusion before truncation, no metadata-only hits,
# cached doc features, CLI error boundary, dry-run + --lexical-only
# ---------------------------------------------------------------------------


class _ScoresBM25:
    def __init__(self, scores):
        self.scores = list(scores)

    def get_scores(self, tokens):
        return np.asarray(self.scores, dtype=np.float32)


def _vdoc(record_id: str, camera_id: str, text: str, day: str = "day1"):
    from castlerag.index.visual_lexical import VisualTextDoc

    return VisualTextDoc(
        record_id=record_id,
        source_type="main_clip",
        modality="video",
        day=day,
        camera_id=camera_id,
        participant_id=camera_id,
        absolute_start=0,
        absolute_end=1,
        text=text,
    )


_CHOICES = {"a": "Samsung", "b": "Bosch", "c": "Miele", "d": "LG"}


def test_score_visual_docs_excludes_cameras_before_truncation():
    from castlerag.retrieval.visual_lexical import score_visual_docs

    docs = [
        _vdoc("k1", "Kitchen", "fridge with SAMSUNG logo"),
        _vdoc("k2", "Kitchen", "fridge door SAMSUNG"),
        _vdoc("a1", "Allie", "a fridge in the corner"),
    ]
    index = SimpleNamespace(bm25=_ScoresBM25([5.0, 4.0, 1.0]), docs=docs)
    hits = score_visual_docs(
        visual_index=index,
        query="What brand is the fridge?",
        choices=_CHOICES,
        top_k=1,
        exclude_cameras=["Kitchen"],
    )
    # Both Kitchen docs outrank Allie's, but they must not eat the single slot.
    assert [h.record_id for h in hits] == ["a1"]


def test_score_visual_docs_ignores_docs_with_no_lexical_evidence():
    from castlerag.retrieval.visual_lexical import score_visual_docs

    docs = [
        _vdoc("x1", "Allie", "someone reads a book"),
        _vdoc("x2", "Bjorn", "SAMSUNG sticker on the fridge"),
    ]
    index = SimpleNamespace(bm25=_ScoresBM25([0.0, 0.0]), docs=docs)
    hits = score_visual_docs(
        visual_index=index,
        query="What brand is the fridge?",
        choices=_CHOICES,
        day_hint="day1",
        person_hint="Allie",
    )
    # x1 only gets the day + person bonuses; x2 matches an answer token.
    assert [h.record_id for h in hits] == ["x2"]


def test_score_visual_docs_caches_doc_features_on_index():
    from castlerag.retrieval.visual_lexical import score_visual_docs

    docs = [_vdoc("x2", "Bjorn", "SAMSUNG sticker on the fridge")]
    index = SimpleNamespace(bm25=_ScoresBM25([1.0]), docs=docs)
    score_visual_docs(visual_index=index, query="fridge brand", choices=_CHOICES)
    first = index._scorer_features
    score_visual_docs(visual_index=index, query="fridge brand", choices=_CHOICES)
    assert index._scorer_features is first
    assert first[1][0] >= {"samsung", "fridge"}


def _cli_cfg(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "preprocessing:\n"
        f"  chunks_dir: {tmp_path / 'chunks'}\n"
        "embedding:\n"
        f"  cache_dir: {cache_dir}\n"
        "dataset:\n"
        "  camera_scope: all\n"
    )
    return cfg_path


def test_cli_retrieve_reports_unreadable_visual_index(tmp_path: Path, monkeypatch):
    import castlerag.cli as cli_module

    cfg_path = _cli_cfg(tmp_path)
    cache_dir = tmp_path / "embeddings"
    (cache_dir / "transcripts.pkl").write_bytes(b"placeholder")
    (cache_dir / VISUAL_TEXT_INDEX_NAME).write_bytes(b"not a pickle")
    monkeypatch.setattr(cli_module, "load_bm25_index", lambda path: object())
    result = CliRunner().invoke(
        app,
        [
            "retrieve", "What brand is the fridge?",
            "--a", "Samsung", "--b", "Bosch", "--c", "Miele", "--d", "LG",
            "--config", str(cfg_path),
        ],
    )
    assert result.exit_code == 1
    # rich wraps long lines at the terminal width (the tmp path is longer on
    # CI), so compare against whitespace-normalised output.
    flat = " ".join(result.output.split())
    assert "failed to load" in flat
    assert "Traceback" not in flat


def test_cli_index_lexical_only_dry_run_reports_and_writes_nothing(tmp_path: Path):
    cfg_path = _cli_cfg(tmp_path)
    result = CliRunner().invoke(
        app, ["index", "--lexical-only", "--dry-run", "--config", str(cfg_path)]
    )
    assert result.exit_code == 0, result.output
    assert "--lexical-only would rebuild" in result.output
    assert not (tmp_path / "embeddings" / "transcripts.pkl").exists()
    assert not (tmp_path / "embeddings" / VISUAL_TEXT_INDEX_NAME).exists()


def test_build_visual_bm25_index_write_is_atomic(tmp_path: Path, monkeypatch):
    """A failed write must leave the previous index intact and no temp file."""
    import castlerag.index.visual_lexical as vl_module

    out = tmp_path / VISUAL_TEXT_INDEX_NAME
    clips, events = _corpus()
    build_visual_bm25_index(clips, events, out)
    before = out.read_bytes()

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(vl_module.pickle, "dump", _boom)
    with pytest.raises(OSError):
        build_visual_bm25_index(clips[:1], [], out)
    assert out.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [VISUAL_TEXT_INDEX_NAME]


def test_score_visual_docs_answer_stopwords_do_not_count_as_evidence():
    from castlerag.retrieval.visual_lexical import score_visual_docs

    docs = [
        _vdoc("g1", "Allie", "a person walks in the garden"),
        _vdoc("k1", "Bjorn", "the fridge in the kitchen"),
    ]
    index = SimpleNamespace(bm25=_ScoresBM25([0.0, 0.0]), docs=docs)
    hits = score_visual_docs(
        visual_index=index,
        query="Where was the fridge?",
        choices={"a": "in the kitchen", "b": "in the hall", "c": "x", "d": "y"},
        day_hint="day1",
    )
    # g1 shares only "in" / "the" with the choices; that is not evidence.
    assert [h.record_id for h in hits] == ["k1"]


def test_score_visual_docs_yes_no_choices_are_not_evidence():
    from castlerag.retrieval.visual_lexical import score_visual_docs

    docs = [
        _vdoc("g1", "Allie", "no one is at the table"),
        _vdoc("k1", "Bjorn", "the fridge door is open"),
    ]
    index = SimpleNamespace(bm25=_ScoresBM25([0.0, 0.0]), docs=docs)
    hits = score_visual_docs(
        visual_index=index,
        query="Was the fridge open?",
        choices={"a": "Yes", "b": "No", "c": "Unclear", "d": "Sometimes"},
        day_hint="day1",
    )
    assert [h.record_id for h in hits] == ["k1"]


def test_build_visual_bm25_index_skips_tokenless_docs(tmp_path: Path):
    """Punctuation-only captions must not crash BM25Okapi (ZeroDivisionError)."""
    out = tmp_path / VISUAL_TEXT_INDEX_NAME
    only_punct = [_clip("c_punct", caption="..."), _clip("c_dots", caption="- - -")]
    bundle = build_visual_bm25_index(only_punct, [], out)
    assert bundle.docs == [] and bundle.bm25 is None
    loaded = load_visual_bm25_index(out)
    assert loaded.docs == [] and loaded.bm25 is None

    mixed = only_punct + [_clip("c_ok", caption="a red mug on the table")]
    bundle = build_visual_bm25_index(mixed, [], out)
    assert [d.record_id for d in bundle.docs] == ["c_ok"]
    assert len(bundle.tokenized_corpus) == 1
    assert bundle.bm25 is not None


def test_rrf_representative_is_first_seen_not_highest_raw_score():
    """A lexical hit must not replace the dense point for a shared record."""
    from castlerag.retrieval.search import reciprocal_rank_fusion

    dense = _vdoc("shared", "Allie", "SAMSUNG fridge")
    from castlerag.retrieval.visual_lexical import score_visual_docs

    lexical_hit = score_visual_docs(
        visual_index=SimpleNamespace(bm25=_ScoresBM25([9.0]), docs=[dense]),
        query="fridge brand",
        choices=_CHOICES,
    )[0]
    dense_hit = lexical_hit.model_copy(
        update={"point_id": "qdrant-uuid", "score": 0.42, "raw_score": 0.42}
    )
    fused = reciprocal_rank_fusion([[dense_hit], [lexical_hit]], k=60)
    assert len(fused) == 1
    assert fused[0].point_id == "qdrant-uuid"
    assert fused[0].raw_score == 0.42


def test_visual_text_route_weights_rejects_unknown_route_and_negative():
    from castlerag.config import CastleRAGConfig

    with pytest.raises(ValueError, match="unknown route"):
        CastleRAGConfig.model_validate(
            {"retrieval": {"visual_text_route_weights": {"visual": 2.0}}}
        )
    with pytest.raises(ValueError, match=">= 0"):
        CastleRAGConfig.model_validate(
            {"retrieval": {"visual_text_route_weights": {"mixed": -1.0}}}
        )
    cfg = CastleRAGConfig.model_validate(
        {"retrieval": {"visual_text_route_weights": {"static_visual": 3.0}}}
    )
    assert cfg.retrieval.visual_text_route_weights == {"static_visual": 3.0}
