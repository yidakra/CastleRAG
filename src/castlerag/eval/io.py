"""Loaders for official CASTLE questions, eval artifacts, and submission export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from castlerag.schemas import EvalQuestion, Prediction


def load_questions(path: Path) -> Dict[str, EvalQuestion]:
    """Load the official CASTLE question JSON.

    Expected format:
      {
        "2026_q1": {
          "query": "...",
          "answers": {"a": ..., "b": ..., "c": ..., "d": ...},
        },
        ...
      }
    """
    raw: Dict[str, dict] = json.loads(path.read_text())
    return {
        qid: EvalQuestion(
            question_id=qid,
            query=item["query"],
            answers=item["answers"],
            ground_truth=item.get("ground_truth"),
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


def select_questions(
    questions: Dict[str, EvalQuestion],
    *,
    question_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> Dict[str, EvalQuestion]:
    """Select a deterministic subset of questions.

    The source JSON order is preserved. If ``question_ids`` is provided, the
    subset follows that order and rejects unknown ids immediately.
    """
    if question_ids is not None:
        selected: Dict[str, EvalQuestion] = {}
        missing: List[str] = []
        for qid in question_ids:
            question = questions.get(qid)
            if question is None:
                missing.append(qid)
                continue
            selected[qid] = question
        if missing:
            missing_str = ", ".join(missing)
            raise KeyError(f"Unknown question ids: {missing_str}")
    else:
        selected = dict(questions)

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be > 0")
        items = list(selected.items())[:limit]
        selected = dict(items)

    return selected


def write_predictions(predictions: Dict[str, Prediction], out_path: Path) -> None:
    """Write rich prediction artifacts for local evaluation/debugging."""
    payload = {
        qid: pred.model_dump(mode="json")
        for qid, pred in sorted(predictions.items())
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def write_evidence_traces(traces: List[dict], out_path: Path) -> None:
    """Write one JSON object per line for downstream trace inspection."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for trace in traces:
            f.write(json.dumps(trace))
            f.write("\n")


def export_submission(predictions: Dict[str, Prediction], out_path: Path) -> None:
    """Write submission JSON in the official format: {question_id: answer}."""
    submission = {
        qid: pred.predicted_answer for qid, pred in sorted(predictions.items())
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(submission, indent=2))
