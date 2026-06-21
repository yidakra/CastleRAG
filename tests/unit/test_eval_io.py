"""Tests for src/castlerag/eval/io.py"""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from castlerag.config import CastleRAGConfig
from castlerag.eval.io import (
    compute_accuracy,
    compute_diversity_metrics,
    export_submission,
    load_predictions,
    load_questions,
    load_questions_csv,
    select_questions,
    write_evidence_traces,
    write_predictions,
)
from castlerag.eval.run_eval import (
    EvalPipeline,
    PipelineDependencyError,
    run_eval,
)
from castlerag.retrieval.candidate_expand import expand_candidates
from castlerag.routing.question_router import RouteHints
from castlerag.schemas import (
    EvalQuestion,
    EvidencePack,
    Prediction,
    RerankResult,
    RetrievalHit,
)

run_eval_module = importlib.import_module("castlerag.eval.run_eval")


def _write_questions(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "questions.json"
    p.write_text(json.dumps(data))
    return p


def _write_predictions(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps(data))
    return p


def _write_answers(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "answers.json"
    p.write_text(json.dumps(data))
    return p


_QUESTIONS_RAW = {
    "q1": {
        "query": "What did Allie do?",
        "answers": {"a": "A", "b": "B", "c": "C", "d": "D"},
    },
    "q2": {
        "query": "Where was Bjorn?",
        "answers": {"a": "Kitchen", "b": "Office", "c": "Garden", "d": "Hall"},
    },
}


def test_load_questions(tmp_path: Path):
    p = _write_questions(tmp_path, _QUESTIONS_RAW)
    qs = load_questions(p)
    assert len(qs) == 2
    assert "q1" in qs
    assert isinstance(qs["q1"], EvalQuestion)
    assert qs["q1"].query == "What did Allie do?"


def test_load_predictions_compact(tmp_path: Path):
    p = _write_predictions(tmp_path, {"q1": "a", "q2": "c"})
    preds = load_predictions(p)
    assert preds["q1"].predicted_answer == "a"
    assert preds["q2"].predicted_answer == "c"


def test_load_predictions_rich_format(tmp_path: Path):
    p = _write_predictions(
        tmp_path,
        {
            "q1": {"predicted_answer": "b", "route": "speech_text"},
        },
    )
    preds = load_predictions(p)
    assert preds["q1"].predicted_answer == "b"
    assert preds["q1"].route == "speech_text"


def test_compute_accuracy_perfect(tmp_path: Path):
    q_path = _write_questions(tmp_path, _QUESTIONS_RAW)
    qs = load_questions(q_path)
    preds = {
        "q1": Prediction(question_id="q1", predicted_answer="a"),
        "q2": Prediction(question_id="q2", predicted_answer="b"),
    }
    ans_path = _write_answers(tmp_path, {"q1": "a", "q2": "b"})
    acc = compute_accuracy(qs, preds, ans_path)
    assert acc == 1.0


def test_compute_accuracy_partial(tmp_path: Path):
    q_path = _write_questions(tmp_path, _QUESTIONS_RAW)
    qs = load_questions(q_path)
    preds = {
        "q1": Prediction(question_id="q1", predicted_answer="a"),
        "q2": Prediction(question_id="q2", predicted_answer="a"),
    }
    ans_path = _write_answers(tmp_path, {"q1": "a", "q2": "b"})  # q2 wrong
    acc = compute_accuracy(qs, preds, ans_path)
    assert acc == 0.5


def test_compute_accuracy_empty(tmp_path: Path):
    q_path = _write_questions(tmp_path, {})
    qs = load_questions(q_path)
    ans_path = _write_answers(tmp_path, {})
    acc = compute_accuracy(qs, {}, ans_path)
    assert acc == 0.0


def test_compute_accuracy_partial_answer_key(tmp_path: Path):
    """Denominator must be graded questions, not total questions."""
    q_path = _write_questions(tmp_path, _QUESTIONS_RAW)
    qs = load_questions(q_path)
    preds = {"q1": Prediction(question_id="q1", predicted_answer="a")}
    # Answer key only has q1, not q2 — accuracy should be 1/1 not 1/2
    ans_path = _write_answers(tmp_path, {"q1": "a"})
    acc = compute_accuracy(qs, preds, ans_path)
    assert acc == 1.0


def test_load_predictions_rejects_bad_format(tmp_path: Path):
    p = _write_predictions(tmp_path, {"q1": 42})  # int, not str or dict
    with pytest.raises(ValueError, match="Unsupported"):
        load_predictions(p)


def test_export_submission(tmp_path: Path):
    preds = {
        "q2": Prediction(question_id="q2", predicted_answer="c"),
        "q1": Prediction(question_id="q1", predicted_answer="a"),
    }
    out = tmp_path / "submissions.json"
    export_submission(preds, out)
    data = json.loads(out.read_text())
    assert data == {"q1": "a", "q2": "c"}
    # Keys should be sorted
    assert list(data.keys()) == sorted(data.keys())


def test_select_questions_by_ids_and_limit(tmp_path: Path):
    q_path = _write_questions(tmp_path, _QUESTIONS_RAW)
    qs = load_questions(q_path)
    selected = select_questions(qs, question_ids=["q2", "q1"], limit=1)
    assert list(selected) == ["q2"]


def test_select_questions_rejects_unknown_id(tmp_path: Path):
    q_path = _write_questions(tmp_path, _QUESTIONS_RAW)
    qs = load_questions(q_path)
    with pytest.raises(KeyError, match="Unknown question ids"):
        select_questions(qs, question_ids=["q3"])


def test_write_predictions_and_traces(tmp_path: Path):
    predictions = {
        "q1": Prediction(
            question_id="q1",
            predicted_answer="b",
            route="speech_text",
            top_evidence_ids=["tw_1"],
        )
    }
    pred_path = tmp_path / "predictions.json"
    trace_path = tmp_path / "evidence_traces.jsonl"
    write_predictions(predictions, pred_path)
    write_evidence_traces(
        [{"question_id": "q1", "route": "speech_text", "predicted_answer": "b"}],
        trace_path,
    )

    pred_data = json.loads(pred_path.read_text())
    trace_lines = trace_path.read_text().splitlines()
    assert pred_data["q1"]["predicted_answer"] == "b"
    assert json.loads(trace_lines[0])["question_id"] == "q1"


def _make_hit(record_id: str) -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        score=0.9,
        point_id=f"pt_{record_id}",
        record_id=record_id,
        source_type="transcript_window",
        modality="text",
        day="day1",
        camera_id="Allie",
        participant_id="Allie",
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_215_000,
        transcript_text="Allie said hello in the kitchen.",
    )


def _fake_pipeline() -> EvalPipeline:
    def _route(question: str, choices: dict[str, str]) -> RouteHints:
        return RouteHints(route="speech_text", day="day1", participant="Allie")

    def _retrieve(question: EvalQuestion, hints: RouteHints) -> list[RetrievalHit]:
        return [
            _make_hit(f"{question.question_id}_1"),
            _make_hit(f"{question.question_id}_2"),
        ]

    def _rerank(
        question: EvalQuestion,
        hints: RouteHints,
        candidate_packs: list[EvidencePack],
    ) -> RerankResult:
        return RerankResult(
            route=hints.route,
            support_priors={"a": 1.0, "b": 0.25, "c": 0.0, "d": 0.0},
            evidence_rows=[candidate_packs[0].primary_hit],
        )

    def _generate(
        question: EvalQuestion,
        hints: RouteHints,
        evidence_rows: list[RetrievalHit],
        support_priors: dict[str, float],
    ) -> Prediction:
        return Prediction(
            question_id=question.question_id,
            predicted_answer="a",
            route=hints.route,
            support_priors=support_priors,
            top_evidence_ids=[hit.record_id for hit in evidence_rows],
            raw_answer_text="FINAL_ANSWER: a",
            confidence=0.9,
        )

    return EvalPipeline(
        route=_route,
        retrieve=_retrieve,
        rerank=_rerank,
        generate=_generate,
    )


def test_compute_diversity_metrics_empty():
    result = compute_diversity_metrics([])
    assert result["mean_cameras_per_question"] == 0.0
    assert result["pct_multi_camera"] == 0.0
    assert result["camera_count_distribution"] == {}


def test_compute_diversity_metrics_single_camera():
    traces = [
        {"top_evidence_cameras": ["Allie"]},
        {"top_evidence_cameras": ["Allie"]},
    ]
    result = compute_diversity_metrics(traces)
    assert result["mean_cameras_per_question"] == 1.0
    assert result["pct_multi_camera"] == 0.0
    assert result["camera_count_distribution"] == {1: 2}


def test_compute_diversity_metrics_multi_camera():
    traces = [
        {"top_evidence_cameras": ["Allie", "Bjorn", "Cathal"]},
        {"top_evidence_cameras": ["Allie", "Bjorn"]},
        {"top_evidence_cameras": ["Allie"]},
    ]
    result = compute_diversity_metrics(traces)
    assert result["mean_cameras_per_question"] == pytest.approx(2.0)
    assert result["pct_multi_camera"] == pytest.approx(2 / 3)
    assert result["camera_count_distribution"] == {1: 1, 2: 1, 3: 1}


def test_compute_diversity_metrics_ignores_none_cameras():
    traces = [{"top_evidence_cameras": [None, "Allie", None, "Bjorn"]}]
    result = compute_diversity_metrics(traces)
    assert result["mean_cameras_per_question"] == 2.0
    assert result["pct_multi_camera"] == 1.0


def test_compute_diversity_metrics_missing_key():
    traces = [{"top_evidence_ids": ["id1"]}, {"top_evidence_cameras": ["Allie"]}]
    result = compute_diversity_metrics(traces)
    assert result["camera_count_distribution"] == {0: 1, 1: 1}


def test_run_eval_traces_include_cameras_and_metrics_include_diversity(tmp_path: Path):
    q_path = _write_questions(tmp_path, _QUESTIONS_RAW)
    qs = load_questions(q_path)
    answers_path = _write_answers(tmp_path, {"q1": "a", "q2": "a"})
    result = run_eval(
        qs,
        answers_path=answers_path,
        out_dir=tmp_path / "outputs",
        pipeline=_fake_pipeline(),
    )

    assert all("top_evidence_cameras" in t for t in result.traces)
    assert result.diversity is not None
    assert "pct_multi_camera" in result.diversity
    metrics = json.loads(result.output_paths.metrics.read_text())
    assert "diversity" in metrics


def test_expand_candidates_builds_contextual_evidence_packs():
    transcript = _make_hit("tx_1")
    clip = RetrievalHit(
        rank=2,
        score=0.8,
        point_id="pt_clip",
        record_id="clip_1",
        source_type="main_clip",
        modality="video",
        day="day1",
        camera_id="Allie",
        participant_id="Allie",
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_230_000,
        event_summary="Allie speaks in the kitchen.",
        asset_path="/tmp/clip.mp4",
    )
    aux = RetrievalHit(
        rank=3,
        score=0.7,
        point_id="pt_aux",
        record_id="aux_1",
        source_type="aux_photo",
        modality="image",
        day="day1",
        camera_id=None,
        participant_id="Allie",
        absolute_start=1_672_531_205_000,
        absolute_end=1_672_531_205_001,
        ocr_text="Receipt on counter",
        asset_path="/tmp/photo.jpg",
    )
    packs = expand_candidates(
        [transcript, clip, aux],
        route="speech_text",
        max_candidate_videos=4,
        frames_per_candidate=32,
    )
    assert packs
    assert any(pack.transcript_evidence for pack in packs)
    assert any(pack.event_summaries for pack in packs)
    assert any(pack.auxiliary_notes for pack in packs)


def test_run_eval_writes_predictions_submission_and_metrics(tmp_path: Path):
    q_path = _write_questions(tmp_path, _QUESTIONS_RAW)
    qs = load_questions(q_path)
    answers_path = _write_answers(tmp_path, {"q1": "a", "q2": "a"})
    result = run_eval(
        qs,
        answers_path=answers_path,
        out_dir=tmp_path / "outputs",
        pipeline=_fake_pipeline(),
    )

    assert len(result.predictions) == 2
    assert result.accuracy == 1.0
    assert result.output_paths.predictions.exists()
    assert result.output_paths.submissions.exists()
    assert result.output_paths.evidence_traces.exists()
    assert result.output_paths.metrics.exists()
    assert json.loads(result.output_paths.submissions.read_text()) == {
        "q1": "a",
        "q2": "a",
    }


def test_run_eval_smoke_limit_respects_five_question_contract(tmp_path: Path):
    questions = {
        f"q{i}": {
            "query": f"Question {i}?",
            "answers": {"a": "A", "b": "B", "c": "C", "d": "D"},
        }
        for i in range(1, 8)
    }
    qs = load_questions(_write_questions(tmp_path, questions))
    result = run_eval(
        qs,
        out_dir=tmp_path / "smoke",
        max_questions=5,
        pipeline=_fake_pipeline(),
    )
    assert len(result.predictions) == 5
    assert list(result.predictions) == ["q1", "q2", "q3", "q4", "q5"]


def test_run_eval_wraps_missing_reranker_as_dependency_error(tmp_path: Path):
    q_path = _write_questions(tmp_path, {"q1": _QUESTIONS_RAW["q1"]})
    qs = load_questions(q_path)

    def _route(question: str, choices: dict[str, str]) -> RouteHints:
        return RouteHints(route="speech_text")

    def _retrieve(question: EvalQuestion, hints: RouteHints) -> list[RetrievalHit]:
        return [_make_hit("q1_1")]

    def _rerank(
        question: EvalQuestion,
        hints: RouteHints,
        candidate_packs: list[dict],
    ) -> list[dict]:
        raise NotImplementedError("reranker pending")

    def _generate(
        question: EvalQuestion,
        hints: RouteHints,
        evidence_rows: list[RetrievalHit],
        support_priors: dict[str, float],
    ) -> Prediction:
        raise AssertionError("should not reach generation")

    pipeline = EvalPipeline(
        route=_route,
        retrieve=_retrieve,
        rerank=_rerank,
        generate=_generate,
    )
    with pytest.raises(PipelineDependencyError, match="reranking.*q1"):
        run_eval(qs, out_dir=tmp_path / "outputs", pipeline=pipeline)


def test_run_eval_wraps_retrieve_dependency_error_with_question_id(tmp_path: Path):
    q_path = _write_questions(tmp_path, {"q1": _QUESTIONS_RAW["q1"]})
    qs = load_questions(q_path)

    def _route(question: str, choices: dict[str, str]) -> RouteHints:
        return RouteHints(route="speech_text")

    def _retrieve(question: EvalQuestion, hints: RouteHints) -> list[RetrievalHit]:
        raise PipelineDependencyError("Qdrant collection is empty")

    def _rerank(
        question: EvalQuestion,
        hints: RouteHints,
        candidate_packs: list[dict],
    ) -> list[dict]:
        raise AssertionError("should not reach reranking")

    def _generate(
        question: EvalQuestion,
        hints: RouteHints,
        evidence_rows: list[RetrievalHit],
        support_priors: dict[str, float],
    ) -> Prediction:
        raise AssertionError("should not reach generation")

    pipeline = EvalPipeline(
        route=_route,
        retrieve=_retrieve,
        rerank=_rerank,
        generate=_generate,
    )
    with pytest.raises(
        PipelineDependencyError,
        match="retrieval dependency failed for question q1: Qdrant collection is empty",
    ):
        run_eval(qs, out_dir=tmp_path / "outputs", pipeline=pipeline)


def test_run_eval_default_pipeline_reports_missing_local_index_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    q_path = _write_questions(tmp_path, {"q1": _QUESTIONS_RAW["q1"]})
    qs = load_questions(q_path)
    cfg = CastleRAGConfig()
    cfg.embedding.cache_dir = str(tmp_path / "embeddings")
    cfg.preprocessing.chunks_dir = str(tmp_path / "chunks")

    monkeypatch.setattr(run_eval_module, "load_config", lambda **_: cfg)

    with pytest.raises(
        PipelineDependencyError,
        match="BM25 transcript index not found",
    ):
        run_eval(qs, out_dir=tmp_path / "outputs")


def test_run_eval_default_pipeline_reports_empty_qdrant_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    q_path = _write_questions(tmp_path, {"q1": _QUESTIONS_RAW["q1"]})
    qs = load_questions(q_path)
    cfg = CastleRAGConfig()
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    (emb_dir / "transcripts.pkl").write_bytes(b"placeholder")
    cfg.embedding.cache_dir = str(emb_dir)
    cfg.preprocessing.chunks_dir = str(tmp_path / "chunks")
    monkeypatch.setattr(run_eval_module, "load_config", lambda **_: cfg)
    monkeypatch.setattr(
        run_eval_module,
        "load_bm25_index",
        lambda path: SimpleNamespace(windows=[object()]),
    )

    class _EmptyClient:
        def collection_exists(self, name: str) -> bool:
            return True

        def count(self, collection_name: str, exact: bool = False) -> int:
            return 0

    monkeypatch.setattr(
        run_eval_module,
        "get_client",
        lambda host, port: _EmptyClient(),
    )

    with pytest.raises(PipelineDependencyError, match="Qdrant collection .* is empty"):
        run_eval(qs, out_dir=tmp_path / "outputs")


def test_run_eval_default_pipeline_reports_missing_vllm_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    q_path = _write_questions(tmp_path, {"q1": _QUESTIONS_RAW["q1"]})
    qs = load_questions(q_path)
    cfg = CastleRAGConfig()
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    (emb_dir / "transcripts.pkl").write_bytes(b"placeholder")
    cfg.embedding.cache_dir = str(emb_dir)
    cfg.preprocessing.chunks_dir = str(tmp_path / "chunks")
    monkeypatch.setattr(run_eval_module, "load_config", lambda **_: cfg)
    monkeypatch.setattr(
        run_eval_module,
        "load_bm25_index",
        lambda path: SimpleNamespace(windows=[object()]),
    )

    class _ReadyClient:
        def collection_exists(self, name: str) -> bool:
            return True

        def count(self, collection_name: str, exact: bool = False) -> int:
            return 12

    monkeypatch.setattr(
        run_eval_module,
        "get_client",
        lambda host, port: _ReadyClient(),
    )
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    with pytest.raises(PipelineDependencyError, match="VLLM_BASE_URL is not set"):
        run_eval(qs, out_dir=tmp_path / "outputs")


def test_run_eval_default_pipeline_wraps_corrupt_bm25_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    q_path = _write_questions(tmp_path, {"q1": _QUESTIONS_RAW["q1"]})
    qs = load_questions(q_path)
    cfg = CastleRAGConfig()
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    (emb_dir / "transcripts.pkl").write_bytes(b"placeholder")
    cfg.embedding.cache_dir = str(emb_dir)
    cfg.preprocessing.chunks_dir = str(tmp_path / "chunks")
    monkeypatch.setattr(run_eval_module, "load_config", lambda **_: cfg)
    monkeypatch.setattr(
        run_eval_module,
        "load_bm25_index",
        lambda path: (_ for _ in ()).throw(ValueError("corrupt bundle")),
    )

    with pytest.raises(
        PipelineDependencyError,
        match="failed to load BM25 transcript index",
    ):
        run_eval(qs, out_dir=tmp_path / "outputs")


def test_run_eval_unexpected_retrieval_error_is_not_reclassified(
    tmp_path: Path,
):
    q_path = _write_questions(tmp_path, {"q1": _QUESTIONS_RAW["q1"]})
    qs = load_questions(q_path)

    def _route(question: str, choices: dict[str, str]) -> RouteHints:
        return RouteHints(route="speech_text")

    def _retrieve(question: EvalQuestion, hints: RouteHints) -> list[RetrievalHit]:
        raise ValueError("logic bug")

    def _rerank(
        question: EvalQuestion,
        hints: RouteHints,
        candidate_packs: list[dict],
    ) -> list[dict]:
        raise AssertionError("should not reach reranking")

    def _generate(
        question: EvalQuestion,
        hints: RouteHints,
        evidence_rows: list[RetrievalHit],
        support_priors: dict[str, float],
    ) -> Prediction:
        raise AssertionError("should not reach generation")

    pipeline = EvalPipeline(
        route=_route,
        retrieve=_retrieve,
        rerank=_rerank,
        generate=_generate,
    )
    with pytest.raises(
        PipelineDependencyError,
        match="retrieval failed for question q1",
    ):
        run_eval(qs, out_dir=tmp_path / "outputs", pipeline=pipeline)


# ---------------------------------------------------------------------------
# CSV loader tests
# ---------------------------------------------------------------------------

_CSV_HEADER = (
    "Question,Answer,Anchor,Distractor 1,Distractor 2,Distractor 3,"
    "Type,Day,Authored by,Verified By,Issue,Column 1\n"
)
_CSV_ROW1 = (
    "What did Allie eat?,Sausages,Day1 14:00,Pizza,Pasta,Soup,what,1,Luca,Werner,,\n"
)
_CSV_ROW2 = (
    "Where was Bjorn?,Kitchen,Day2 10:00,Meeting room,Living room,Garden,"
    "where,2,Klaus,Allie,,\n"
)


def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "questions.csv"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_questions_csv_parses_rows(tmp_path: Path):
    p = _write_csv(tmp_path, _CSV_HEADER + _CSV_ROW1 + _CSV_ROW2)
    qs = load_questions_csv(p)
    assert len(qs) == 2


def test_load_questions_csv_query_text(tmp_path: Path):
    p = _write_csv(tmp_path, _CSV_HEADER + _CSV_ROW1)
    qs = load_questions_csv(p)
    q = next(iter(qs.values()))
    assert q.query == "What did Allie eat?"


def test_load_questions_csv_four_choices(tmp_path: Path):
    p = _write_csv(tmp_path, _CSV_HEADER + _CSV_ROW1)
    qs = load_questions_csv(p)
    q = next(iter(qs.values()))
    assert set(q.answers.keys()) == {"a", "b", "c", "d"}
    assert set(q.answers.values()) == {"Sausages", "Pizza", "Pasta", "Soup"}


def test_load_questions_csv_ground_truth_is_correct_answer(tmp_path: Path):
    p = _write_csv(tmp_path, _CSV_HEADER + _CSV_ROW1)
    qs = load_questions_csv(p)
    q = next(iter(qs.values()))
    assert q.ground_truth is not None
    assert q.answers[q.ground_truth] == "Sausages"


def test_load_questions_csv_shuffle_is_stable(tmp_path: Path):
    p = _write_csv(tmp_path, _CSV_HEADER + _CSV_ROW1)
    qs1 = load_questions_csv(p)
    qs2 = load_questions_csv(p)
    q1 = next(iter(qs1.values()))
    q2 = next(iter(qs2.values()))
    assert q1.answers == q2.answers
    assert q1.ground_truth == q2.ground_truth


def test_load_questions_csv_skips_blank_rows(tmp_path: Path):
    content = _CSV_HEADER + _CSV_ROW1 + ",,,,,,,,,,,\n" + _CSV_ROW2
    p = _write_csv(tmp_path, content)
    qs = load_questions_csv(p)
    assert len(qs) == 2


def test_load_questions_dispatches_csv(tmp_path: Path):
    p = _write_csv(tmp_path, _CSV_HEADER + _CSV_ROW1)
    qs = load_questions(p)
    assert len(qs) == 1
    q = next(iter(qs.values()))
    assert q.answers[q.ground_truth] == "Sausages"


def test_load_questions_csv_accuracy_with_perfect_predictions(tmp_path: Path):
    p = _write_csv(tmp_path, _CSV_HEADER + _CSV_ROW1)
    qs = load_questions_csv(p)
    # Build a perfect predictions dict (predicted = ground_truth for every question)
    perfect = {
        qid: Prediction(question_id=qid, predicted_answer=q.ground_truth)  # type: ignore[arg-type]
        for qid, q in qs.items()
    }
    acc = compute_accuracy(qs, perfect, answers_path=None)
    assert acc == 1.0


def test_load_questions_csv_accuracy_with_wrong_predictions(tmp_path: Path):
    p = _write_csv(tmp_path, _CSV_HEADER + _CSV_ROW1)
    qs = load_questions_csv(p)
    q = next(iter(qs.values()))
    wrong_letter = next(
        letter for letter in ["a", "b", "c", "d"] if letter != q.ground_truth
    )
    wrong = {
        qid: Prediction(question_id=qid, predicted_answer=wrong_letter)  # type: ignore[arg-type]
        for qid in qs
    }
    acc = compute_accuracy(qs, wrong, answers_path=None)
    assert acc == 0.0


# ---------------------------------------------------------------------------
# WandbLogger no-op tests (wandb not installed in CI)
# ---------------------------------------------------------------------------

def test_wandb_logger_noop_when_not_installed(monkeypatch):
    """WandbLogger must be a silent no-op when wandb is absent."""
    import castlerag.eval.wandb_logger as wl
    monkeypatch.setattr(wl, "_wandb", lambda: None)
    from castlerag.config import CastleRAGConfig
    cfg = CastleRAGConfig()
    logger = wl.WandbLogger(cfg, n_questions=5)
    assert not logger._active
    # All methods must be callable without error when inactive
    logger.log_summary(0.5, {"mean_cameras": 2.0}, 5)
    logger.log_artifacts([Path("/nonexistent/file.json")])
    logger.finish()
