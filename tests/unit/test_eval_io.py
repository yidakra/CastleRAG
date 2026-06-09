"""Tests for src/castlerag/eval/io.py"""
import json
from pathlib import Path

import pytest

from castlerag.eval.io import (
    compute_accuracy,
    export_submission,
    load_predictions,
    load_questions,
)
from castlerag.schemas import EvalQuestion, Prediction


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
    p = _write_predictions(tmp_path, {
        "q1": {"predicted_answer": "b", "route": "speech_text"},
    })
    preds = load_predictions(p)
    assert preds["q1"].predicted_answer == "b"
    assert preds["q1"].route == "speech_text"


def test_compute_accuracy_perfect(tmp_path: Path):
    q_path = _write_questions(tmp_path, _QUESTIONS_RAW)
    qs = load_questions(q_path)
    preds = {"q1": Prediction(question_id="q1", predicted_answer="a"),
             "q2": Prediction(question_id="q2", predicted_answer="b")}
    ans_path = _write_answers(tmp_path, {"q1": "a", "q2": "b"})
    acc = compute_accuracy(qs, preds, ans_path)
    assert acc == 1.0


def test_compute_accuracy_partial(tmp_path: Path):
    q_path = _write_questions(tmp_path, _QUESTIONS_RAW)
    qs = load_questions(q_path)
    preds = {"q1": Prediction(question_id="q1", predicted_answer="a"),
             "q2": Prediction(question_id="q2", predicted_answer="a")}
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
