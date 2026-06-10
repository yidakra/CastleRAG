"""Tests for src/castlerag/eval/io.py"""

import json
from pathlib import Path

import pytest

from castlerag.eval.io import (
    compute_accuracy,
    export_submission,
    load_predictions,
    load_questions,
    select_questions,
    write_evidence_traces,
    write_predictions,
)
from castlerag.eval.run_eval import EvalPipeline, PipelineDependencyError, run_eval
from castlerag.routing.question_router import RouteHints
from castlerag.schemas import (
    EvalQuestion,
    Prediction,
    RerankResult,
    RetrievalHit,
)


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
        candidate_packs: list[dict],
    ) -> RerankResult:
        return RerankResult(
            route=hints.route,
            support_priors={"a": 1.0, "b": 0.25, "c": 0.0, "d": 0.0},
            evidence_rows=[candidate_packs[0]["primary_hit"]],
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
