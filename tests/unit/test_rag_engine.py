"""Tests for the real RagEngine, its adapters, and the gated engine factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pytest

from castlerag.schemas import RetrievalHit
from castlerag.ui.chat import PlaceholderEngine, SupportLevel
from castlerag.ui.rag_engine import RagEngine

# ---------------------------------------------------------------------------
# Adapters (deterministic, no pipeline)
# ---------------------------------------------------------------------------


def _engine(ego=("Allie", "Bjorn", "Cathal", "Florian", "Klaus")) -> RagEngine:
    return RagEngine(cfg=None, pipeline=None, ego_cameras=tuple(ego))


def _clip_hit(
    camera: str, score: float, *, day="day1", hour=12, start=120.0
) -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        score=score,
        point_id=f"p_{camera}",
        record_id=f"{day}_{camera}_{hour}_0004",
        source_type="main_clip",
        modality="video",
        day=day,
        camera_id=camera,
        hour=hour,
        start_seconds=start,
        end_seconds=start + 30.0,
        absolute_start=1_700_000_000_000,
        absolute_end=1_700_000_030_000,
    )


def test_hits_to_moments_groups_three_synchronized_cameras():
    hits = [
        _clip_hit("Bjorn", 0.61),
        _clip_hit("Luca", 0.74),
        _clip_hit("Klaus", 0.55),
    ]
    moments = _engine()._hits_to_moments(hits, SupportLevel.PARTIAL)
    assert len(moments) == 1
    moment = moments[0]
    assert moment.camera_count == 3
    assert len(moment.cameras) == 3
    # All three real cameras, all synchronized to the same (day, hour, start).
    assert {c.camera_id for c in moment.cameras} == {"Bjorn", "Luca", "Klaus"}
    assert {(c.day, c.hour, c.start_seconds) for c in moment.cameras} == {
        ("day1", 12, 120.0)
    }
    # Exactly one best, and it is the highest-scoring camera (Luca).
    best = [c for c in moment.cameras if c.is_best]
    assert len(best) == 1 and best[0].camera_id == "Luca"
    assert moment.clock_label == "12:02"


def test_hits_to_moments_pads_to_three_when_fewer_real_cameras():
    hits = [_clip_hit("Bjorn", 0.7)]
    moments = _engine()._hits_to_moments(hits, SupportLevel.PARTIAL)
    cams = moments[0].cameras
    assert len(cams) == 3
    # One real camera (score > 0) and two deterministic padded ones (score 0.0).
    real = [c for c in cams if c.match_score > 0]
    padded = [c for c in cams if c.match_score == 0.0]
    assert len(real) == 1 and real[0].camera_id == "Bjorn"
    assert len(padded) == 2
    assert all(c.camera_id != "Bjorn" for c in padded)


def test_pad_cameras_is_deterministic():
    eng = _engine()
    first = eng._hits_to_moments([_clip_hit("Klaus", 0.8)], SupportLevel.PARTIAL)
    second = eng._hits_to_moments([_clip_hit("Klaus", 0.8)], SupportLevel.PARTIAL)
    assert [c.camera_id for c in first[0].cameras] == [
        c.camera_id for c in second[0].cameras
    ]


@pytest.mark.parametrize(
    "max_prior,expected",
    [
        (0.85, SupportLevel.SUPPORTED),
        (0.5, SupportLevel.PARTIAL),
        (0.2, SupportLevel.UNSUPPORTED),
    ],
)
def test_synthesize_claim_support_thresholds(max_prior, expected):
    from castlerag.schemas import Prediction

    pred = Prediction(question_id="q", predicted_answer="b")
    priors = {"a": 0.1, "b": max_prior, "c": 0.0, "d": 0.0}
    claim = _engine()._synthesize_claim(pred, priors, {"b": "Luca"})
    assert claim.support is expected
    assert "Luca" in claim.text


def test_synthesize_claim_thresholds_on_predicted_choice_not_max():
    """Support must come from the predicted choice, never borrowed from another."""
    from castlerag.schemas import Prediction

    pred = Prediction(question_id="q", predicted_answer="b")
    # 'a' is strongly supported but 'b' (the prediction) is not — support must
    # follow 'b', so the claim is UNSUPPORTED rather than borrowing 'a''s prior.
    priors = {"a": 0.9, "b": 0.1, "c": 0.0, "d": 0.0}
    claim = _engine()._synthesize_claim(pred, priors, {"b": "Luca"})
    assert claim.support is SupportLevel.UNSUPPORTED


def test_synthesize_claim_free_form_uses_evidence_not_dummy_priors():
    """Free-form questions ignore MCQ priors and gauge support from evidence."""
    from castlerag.schemas import Prediction

    pred = Prediction(
        question_id="q", predicted_answer="a", raw_answer_text="Bjorn cooks porridge."
    )
    # Dummy priors are high, but is_mcq=False must disregard them; the strong
    # evidence rerank_score is what drives SUPPORTED, and the claim text is the
    # model's free-form answer (never "Option A").
    priors = {"a": 9.0, "b": 0.0, "c": 0.0, "d": 0.0}
    rows = [_clip_hit("Bjorn", 0.5)]
    rows[0] = rows[0].model_copy(update={"rerank_score": 0.85})
    claim = _engine()._synthesize_claim(
        pred,
        priors,
        {key: f"Option {key.upper()}" for key in "abcd"},
        is_mcq=False,
        question="What does Bjorn cook?",
        evidence_rows=rows,
    )
    assert claim.support is SupportLevel.SUPPORTED
    assert "Option" not in claim.text
    assert claim.text == "Bjorn cooks porridge."


def test_synthesize_claim_free_form_uses_explicit_support_score():
    """Free-form support comes from the reranker score, not the (None) rows."""
    from castlerag.schemas import Prediction

    pred = Prediction(
        question_id="q", predicted_answer="a", raw_answer_text="Cathal taught guitar."
    )
    # The displayed rows carry NO rerank_score (the real bug); a strong explicit
    # support_score must still drive SUPPORTED rather than collapsing to 0.0.
    rows = [_clip_hit("Cathal", 0.5)]  # rerank_score is None on these
    claim = _engine()._synthesize_claim(
        pred,
        {},
        {key: "" for key in "abcd"},
        is_mcq=False,
        question="What did Cathal teach Allie?",
        evidence_rows=rows,
        freeform_answer="Cathal taught Allie the guitar.",
        support_score=0.75,
    )
    assert claim.support is SupportLevel.SUPPORTED


def test_freeform_support_score_reads_reranked_not_displayed_rows():
    """The score comes off rerank_result.evidence_rows, which are stamped."""
    from types import SimpleNamespace

    # Rows shown to the UI have no rerank_score; the reranked rows do.
    displayed = [_clip_hit("Cathal", 0.5)]
    reranked = [_clip_hit("Cathal", 0.5).model_copy(update={"rerank_score": 0.75})]
    rr = SimpleNamespace(evidence_rows=reranked)
    assert _engine()._freeform_support_score(rr, displayed) == pytest.approx(0.75)


def test_freeform_support_score_falls_back_to_cosine_when_no_rerank():
    """With no kept reranked evidence, fall back to the best dense cosine."""
    from types import SimpleNamespace

    displayed = [_clip_hit("Cathal", 0.5).model_copy(update={"raw_score": 0.42})]
    rr = SimpleNamespace(evidence_rows=[])
    assert _engine()._freeform_support_score(rr, displayed) == pytest.approx(0.42)


def test_no_evidence_moment_keeps_focus_contract():
    """A no-hit query still yields a focusable, clearly-labelled moment."""
    moment = _engine()._no_evidence_moment(SupportLevel.UNSUPPORTED)
    assert moment.moment_id == "m0"
    assert "No supporting footage" in moment.place_label
    assert moment.cameras  # padded to the fixed camera count, all score 0
    assert all(cam.match_score == 0.0 for cam in moment.cameras)


# ---------------------------------------------------------------------------
# RetrievalHit timing fields plumb through _dense_search
# ---------------------------------------------------------------------------


def test_dense_search_plumbs_timing_fields():
    qm = pytest.importorskip("qdrant_client.http.models")
    from qdrant_client import QdrantClient

    from castlerag.index.qdrant import build_point_batches, upsert_batch
    from castlerag.retrieval.search import _dense_search
    from castlerag.schemas import ClipRecord

    clip = ClipRecord(
        clip_id="day1_Bjorn_12_0004",
        parent_source_id="vid",
        source_type="main_clip",
        modality="video",
        day="day1",
        hour=12,
        camera_id="Bjorn",
        camera_type="ego",
        room="Kitchen",
        start_seconds=120.0,
        end_seconds=150.0,
        absolute_start=1_700_000_000_000,
        absolute_end=1_700_000_030_000,
        source_video_path="/data/main/day1/Bjorn/video/12.mp4",
    )
    rows = build_point_batches([clip], model_version="t", model_name="t")
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="t",
        vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE),
    )
    upsert_batch(
        client=client,
        collection_name="t",
        point_ids=[r.point_id for r in rows],
        vectors=[[1.0, 0.0, 0.0, 0.0]],
        payloads=[r.model_dump(exclude_none=True) for r in rows],
    )
    hits = _dense_search(
        qdrant_client=client,
        collection_name="t",
        query_vector=[1.0, 0.0, 0.0, 0.0],
        limit=5,
        source_type="main_clip",
        modality="video",
    )
    assert hits and hits[0].hour == 12
    assert hits[0].start_seconds == 120.0
    assert hits[0].end_seconds == 150.0
    assert hits[0].room == "Kitchen"


# ---------------------------------------------------------------------------
# End-to-end answer() over a stub pipeline + in-memory Qdrant
# ---------------------------------------------------------------------------


class _StubEmbed:
    dim = 8

    def embed_texts(self, texts: List[str]):
        import numpy as np

        v = np.ones((len(texts), self.dim), dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class _StubResponse:
    def __init__(self, text: str) -> None:
        message = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": message})()]


class _StubLLM:
    class _Completions:
        def create(self, *, model: str, messages: Any, **_: Any) -> _StubResponse:
            last = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
            )
            if "Score this candidate" in last:
                text = json.dumps(
                    {
                        "relevance": 3,
                        "support": {"a": 3, "b": 1, "c": 1, "d": 1},
                        "keep": True,
                        "rationale": "Relevant.",
                    }
                )
            else:
                text = "Evidence supports option A.\nFINAL_ANSWER: a"
            return _StubResponse(text)

    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": self._Completions()})()


def _build_stub_engine(tmp_path: Path) -> RagEngine:
    qm = pytest.importorskip("qdrant_client.http.models")
    from qdrant_client import QdrantClient

    from castlerag.config import CastleRAGConfig
    from castlerag.eval.run_eval import EvalPipeline
    from castlerag.generation.answer import generate_answer
    from castlerag.index.qdrant import build_point_batches, upsert_batch
    from castlerag.index.transcript_lexical import build_bm25_index
    from castlerag.rerank.llm_reranker import rerank_candidates
    from castlerag.retrieval.search import retrieve as _retrieve
    from castlerag.routing.question_router import route_question
    from castlerag.schemas import ClipRecord, EvalQuestion, Prediction, TranscriptWindow

    base = 1_700_000_000_000
    # Three synchronized clips: same day/hour/window, three cameras.
    clips = [
        ClipRecord(
            clip_id=f"day1_{cam}_12_0004",
            parent_source_id=f"vid_{cam}",
            source_type="main_clip",
            modality="video",
            day="day1",
            hour=12,
            camera_id=cam,
            camera_type="ego",
            room="Kitchen",
            start_seconds=120.0,
            end_seconds=150.0,
            absolute_start=base,
            absolute_end=base + 30_000,
            source_video_path=f"/data/main/day1/{cam}/video/12.mp4",
            event_summary=f"{cam} is near the doorway with a cup.",
        )
        for cam in ("Allie", "Bjorn", "Cathal")
    ]
    windows = [
        TranscriptWindow(
            transcript_window_id="tw_0",
            source_type="transcript_window",
            modality="text",
            day="day1",
            camera_id="Allie",
            camera_type="ego",
            participant_id="Allie",
            room="Kitchen",
            hour=12,
            window_index=0,
            absolute_start=base,
            absolute_end=base + 15_000,
            transcript_text="Someone balances a cup of water on the door.",
        )
    ]
    bm25 = build_bm25_index(windows, tmp_path / "transcripts.pkl")
    embed = _StubEmbed()
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="t",
        vectors_config=qm.VectorParams(size=embed.dim, distance=qm.Distance.COSINE),
    )
    records = [*windows, *clips]
    rows = build_point_batches(records, model_version="t", model_name="t")
    upsert_batch(
        client=client,
        collection_name="t",
        point_ids=[r.point_id for r in rows],
        vectors=embed.embed_texts(["x"] * len(records)).tolist(),
        payloads=[r.model_dump(exclude_none=True) for r in rows],
    )
    cfg = CastleRAGConfig()
    llm = _StubLLM()

    def retrieve(question: EvalQuestion, hints: Any) -> List[RetrievalHit]:
        return _retrieve(
            question=question,
            hints=hints,
            qdrant_client=client,
            collection_name="t",
            bm25_index=bm25,
            embed_client=embed,
            retrieval_cfg=cfg.retrieval,
        )

    def rerank(question: EvalQuestion, hints: Any, packs: List[dict]) -> Any:
        return rerank_candidates(
            question=question,
            hints=hints,
            candidate_packs=packs,
            llm_client=llm,
            top_k=cfg.reranking.top_k,
            min_relevance=0,
            model="stub",
        )

    def generate(question, hints, evidence, support) -> Prediction:
        return generate_answer(
            question=question,
            hints=hints,
            evidence_rows=evidence,
            support_priors=support,
            llm_client=llm,
            model="stub",
            max_evidence_rows=cfg.retrieval.max_evidence_rows,
        )

    pipeline = EvalPipeline(
        route=route_question, retrieve=retrieve, rerank=rerank, generate=generate
    )
    return RagEngine(
        cfg=cfg, pipeline=pipeline, ego_cameras=tuple(cfg.dataset.ego_cameras)
    )


def test_run_question_skips_generation_for_free_form(tmp_path):
    """Free-form questions skip the MCQ generator but still retrieve+rerank."""
    from castlerag.eval.run_eval import run_question
    from castlerag.schemas import EvalQuestion

    engine = _build_stub_engine(tmp_path)
    free = EvalQuestion(
        question_id="q_ff",
        query="Who balances a cup of water on the door?",
        answers={"a": "", "b": "", "c": "", "d": ""},
    )
    assert free.is_free_form()
    result = run_question(
        engine.pipeline, engine.cfg, free, generate_prediction=False
    )
    # The MCQ generator was not called, so there is no raw FINAL_ANSWER text...
    assert result.prediction.raw_answer_text == ""
    # ...but retrieval and reranking still ran and produced evidence.
    assert result.evidence_rows


def test_answer_end_to_end_real_pipeline(tmp_path):
    engine = _build_stub_engine(tmp_path)
    result = engine.answer(
        "Who balances a cup of water on the door?",
        {"a": "Allie", "b": "Bjorn", "c": "Cathal", "d": "Luca"},
    )
    assert result.is_placeholder is False
    assert result.claim is not None
    assert result.predicted_choice in {"a", "b", "c", "d"}
    assert result.moments
    for moment in result.moments:
        assert len(moment.cameras) == 3
        assert sum(1 for c in moment.cameras if c.is_best) == 1
        for cam in moment.cameras:
            assert cam.day and cam.hour is not None and cam.start_seconds is not None


# ---------------------------------------------------------------------------
# Gated engine selection
# ---------------------------------------------------------------------------


def test_build_engine_falls_back_without_vllm(monkeypatch):
    from castlerag.ui.engine_factory import build_engine, engine_mode
    from castlerag.ui.youtube import YouTubeMirror

    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    engine = build_engine(YouTubeMirror.from_csv())
    assert isinstance(engine, PlaceholderEngine)
    assert engine_mode(engine) == "offline"


def test_build_engine_falls_back_on_pipeline_error(monkeypatch):
    from castlerag.ui import engine_factory
    from castlerag.ui.youtube import YouTubeMirror

    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:1/v1")
    # Server is up (probe passes); the pipeline build itself is what fails here.
    monkeypatch.setattr(engine_factory, "_vllm_reachable", lambda *_a, **_k: True)

    def _boom(cls, **_kwargs):
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(RagEngine, "from_config", classmethod(_boom))
    engine = engine_factory.build_engine(YouTubeMirror.from_csv())
    assert isinstance(engine, PlaceholderEngine)


def test_from_config_honors_castlerag_config_env(monkeypatch):
    """With no cfg injected, from_config loads the CASTLERAG_CONFIG override."""
    import importlib

    # ``castlerag.eval.run_eval`` is shadowed by a re-exported function at the
    # package level, so reach the real modules through importlib to patch them.
    config_mod = importlib.import_module("castlerag.config")
    run_eval_mod = importlib.import_module("castlerag.eval.run_eval")

    seen = {}

    def _fake_load_config(override_path=None):
        seen["override_path"] = override_path
        return type("Cfg", (), {"dataset": type("D", (), {"ego_cameras": ()})()})()

    monkeypatch.setattr(config_mod, "load_config", _fake_load_config)
    monkeypatch.setattr(run_eval_mod, "_build_default_pipeline", lambda cfg: object())
    monkeypatch.setenv("CASTLERAG_CONFIG", "configs/snellius_me.yaml")

    RagEngine.from_config()
    assert seen["override_path"] == "configs/snellius_me.yaml"


def test_build_engine_falls_back_when_vllm_unreachable(monkeypatch):
    """VLLM_BASE_URL is set but no server answers -> offline, without building."""
    from castlerag.ui import engine_factory
    from castlerag.ui.youtube import YouTubeMirror

    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setattr(engine_factory, "_vllm_reachable", lambda *_a, **_k: False)

    def _should_not_run(cls, **_kwargs):
        raise AssertionError("from_config must not be called when the probe fails")

    monkeypatch.setattr(RagEngine, "from_config", classmethod(_should_not_run))
    engine = engine_factory.build_engine(YouTubeMirror.from_csv())
    assert isinstance(engine, PlaceholderEngine)
    assert engine_factory.engine_mode(engine) == "offline"


def test_pad_cameras_emits_distinct_ids_with_empty_roster():
    """With no roster to pad from, fallback cameras must not duplicate an id."""
    engine = RagEngine(cfg=None, pipeline=None, ego_cameras=())
    moments = engine._hits_to_moments([_clip_hit("Bjorn", 0.7)], SupportLevel.PARTIAL)
    cams = moments[0].cameras
    assert len(cams) == 3
    assert len({c.camera_id for c in cams}) == 3  # all distinct, no duplicate "Bjorn"
    assert cams[0].camera_id == "Bjorn"
