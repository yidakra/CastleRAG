"""Loaders for official CASTLE questions, local answer keys, and submission export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from castlerag.schemas import EvalQuestion, Prediction


def load_questions(path: Path) -> Dict[str, EvalQuestion]:
    """Load the official CASTLE question JSON.

    Expected format:
      {
        "2026_q1": {"query": "...", "answers": {"a": ..., "b": ..., "c": ..., "d": ...}},
        ...
      }
    """
    raw: Dict[str, dict] = json.loads(path.read_text())
    return {
        qid: EvalQuestion(
            question_id=qid,
            query=item["query"],
            answers=item["answers"],
        )
        for qid, item in raw.items()
    }


def load_predictions(path: Path) -> Dict[str, Prediction]:
    """Load castlerag predictions.json.

    Accepts both the compact submission format {"qid": "a"} and the
    richer format {"qid": {predicted_answer: "a", ...}}.
    """
    raw: dict = json.loads(path.read_text())
    result: Dict[str, Prediction] = {}
    for qid, val in raw.items():
        if isinstance(val, str):
            result[qid] = Prediction(question_id=qid, predicted_answer=val)  # type: ignore[arg-type]
        elif isinstance(val, dict):
            result[qid] = Prediction.model_validate({"question_id": qid, **val})
        else:
            raise ValueError(
                f"Unsupported prediction format for question {qid!r}: "
                f"expected str or dict, got {type(val).__name__!r}"
            )
    return result


def compute_accuracy(
    questions: Dict[str, EvalQuestion],
    predictions: Dict[str, Prediction],
    answers_path: Path,
) -> float:
    """Exact-match accuracy over questions that have a ground-truth entry.

    The denominator is the number of questions present in the answer key,
    not the total number of questions, so partial answer keys produce a
    meaningful score rather than artificially deflating accuracy.
    """
    answers: Dict[str, str] = json.loads(answers_path.read_text())
    correct = 0
    graded = 0
    for qid in questions:
        truth = answers.get(qid)
        if truth is None:
            continue
        graded += 1
        pred = predictions.get(qid)
        if pred is not None and pred.predicted_answer == truth:
            correct += 1
    return correct / graded if graded > 0 else 0.0


def export_submission(predictions: Dict[str, Prediction], out_path: Path) -> None:
    """Write submission JSON in the official format: {question_id: answer}."""
    submission = {qid: pred.predicted_answer for qid, pred in sorted(predictions.items())}
    out_path.write_text(json.dumps(submission, indent=2))
