"""Loaders for official CASTLE questions, eval artifacts, and submission export."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from castlerag.schemas import EvalQuestion, Prediction

log = logging.getLogger("castlerag.eval")

_LETTERS = ("a", "b", "c", "d")


def _question_id_from_text(text: str) -> str:
    """Stable short id derived from question text."""
    return "castle_" + hashlib.sha1(text.strip().lower().encode()).hexdigest()[:8]


def _shuffle_choices(
    question: str, correct: str, distractors: List[str]
) -> tuple[Dict[str, str], str]:
    """Deterministically assign correct + 3 distractors to letters a–d.

    Uses the question text hash to pick the position of the correct answer so
    the assignment is stable across reruns but looks random across questions.
    Returns (answers dict, correct_letter).
    """
    h = int(hashlib.sha1(question.strip().lower().encode()).hexdigest(), 16)
    correct_idx = h % 4
    others = list(distractors[:3])
    answers: Dict[str, str] = {}
    other_pos = 0
    for i, letter in enumerate(_LETTERS):
        if i == correct_idx:
            answers[letter] = correct
        else:
            answers[letter] = others[other_pos]
            other_pos += 1
    return answers, _LETTERS[correct_idx]


def load_questions_csv(path: Path) -> Dict[str, EvalQuestion]:
    """Load questions from the CASTLE question-bank CSV.

    Expected columns (from the CVPR challenge spreadsheet export):
      Question, Answer, Anchor, Distractor 1, Distractor 2, Distractor 3,
      Type, Day, Authored by, Verified By, Issue, Column 1

    Each row is converted to an EvalQuestion with four shuffled answer choices
    (a–d) and ground_truth set to the letter that holds the correct answer.
    The shuffle is deterministic per question text so reruns are stable.
    """
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        questions: Dict[str, EvalQuestion] = {}
        for row in reader:
            question = (row.get("Question") or "").strip()
            if not question:
                continue
            correct = (row.get("Answer") or "").strip()
            distractors = [
                (row.get("Distractor 1") or "").strip(),
                (row.get("Distractor 2") or "").strip(),
                (row.get("Distractor 3") or "").strip(),
            ]
            answers, gt = _shuffle_choices(question, correct, distractors)
            qid = _question_id_from_text(question)
            existing = questions.get(qid)
            if existing is not None:
                if existing.query == question:
                    # Same question text appears twice; the second row carries no
                    # new information, so skip it rather than overwrite.
                    continue
                # Different text hashed to the same id (8-hex collision). Append a
                # deterministic suffix so neither question is silently dropped.
                base, n = qid, 2
                while qid in questions:
                    qid = f"{base}_{n}"
                    n += 1
                log.warning(
                    "question_id collision on %s; disambiguated to %s", base, qid
                )
            questions[qid] = EvalQuestion(
                question_id=qid,
                query=question,
                answers=answers,
                ground_truth=gt,  # type: ignore[arg-type]
            )
    return questions


def load_questions(path: Path) -> Dict[str, EvalQuestion]:
    """Load CASTLE questions from JSON or CSV (auto-detected by extension).

    JSON format (official submission format):
      {
        "2026_q1": {
          "query": "...",
          "answers": {"a": ..., "b": ..., "c": ..., "d": ...},
        },
        ...
      }

    CSV format: CASTLE CVPR challenge Question Bank spreadsheet export.
    """
    if Path(path).suffix.lower() == ".csv":
        return load_questions_csv(Path(path))
    raw: Dict[str, dict] = json.loads(Path(path).read_text())
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


def accuracy_breakdown(
    questions: Dict[str, EvalQuestion],
    predictions: Dict[str, Prediction],
    answers_path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Return ``(correct, graded)`` over questions with resolvable ground truth.

    ``graded`` counts only questions that have a ground-truth entry, so callers
    can report ``correct/graded`` instead of fabricating a denominator from the
    total number of predictions or questions (which over-counts when the answer
    key is partial).

    Ground truth is resolved in order:
    1. ``answers_path`` JSON (``{qid: letter}``) — official external key.
    2. ``questions[qid].ground_truth`` — embedded by the CSV loader.
    """
    ext_answers: Dict[str, str] = (
        json.loads(answers_path.read_text()) if answers_path is not None else {}
    )
    correct = 0
    graded = 0
    for qid, q in questions.items():
        truth = ext_answers.get(qid) or q.ground_truth
        if truth is None:
            continue
        graded += 1
        pred = predictions.get(qid)
        if pred is not None and pred.predicted_answer == truth:
            correct += 1
    return correct, graded


def support_breakdown(
    questions: Dict[str, EvalQuestion],
    predictions: Dict[str, Prediction],
    answers_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Split predictions into evidence-backed vs unsupported guesses.

    ``is_supported`` is False when an answer has no retrieval/reranker backing
    (no evidence retrieved, or the reranker credited no choice). Splitting
    accuracy by that flag separates retrieval failures (unsupported guesses)
    from reasoning failures (supported but wrong), and the unsupported rate is a
    cheap health signal on its own. Per-subset accuracy is None without ground
    truth (``unsupported_rate`` is always available).
    """
    ext: Dict[str, str] = (
        json.loads(answers_path.read_text()) if answers_path is not None else {}
    )
    total = len(predictions)
    num_unsupported = sum(1 for p in predictions.values() if not p.is_supported)
    sup_correct = sup_graded = guess_correct = guess_graded = 0
    for qid, q in questions.items():
        truth = ext.get(qid) or q.ground_truth
        if truth is None:
            continue
        pred = predictions.get(qid)
        if pred is None:
            continue
        ok = pred.predicted_answer == truth
        if pred.is_supported:
            sup_graded += 1
            sup_correct += int(ok)
        else:
            guess_graded += 1
            guess_correct += int(ok)

    def _acc(correct: int, graded: int) -> Optional[float]:
        return (correct / graded) if graded > 0 else None

    return {
        "num_predictions": total,
        "num_unsupported": num_unsupported,
        "unsupported_rate": (num_unsupported / total) if total > 0 else None,
        "supported": {
            "graded": sup_graded,
            "correct": sup_correct,
            "accuracy": _acc(sup_correct, sup_graded),
        },
        "unsupported": {
            "graded": guess_graded,
            "correct": guess_correct,
            "accuracy": _acc(guess_correct, guess_graded),
        },
    }


def compute_accuracy(
    questions: Dict[str, EvalQuestion],
    predictions: Dict[str, Prediction],
    answers_path: Optional[Path] = None,
) -> float:
    """Exact-match accuracy over questions that have a ground-truth entry.

    The denominator is the number of graded questions, so partial answer keys
    produce a meaningful score rather than artificially deflating accuracy.
    """
    correct, graded = accuracy_breakdown(questions, predictions, answers_path)
    return correct / graded if graded > 0 else 0.0


def compute_diversity_metrics(traces: List[dict]) -> Dict[str, Any]:
    """Camera diversity across evidence traces.

    For each trace, counts the unique camera IDs in ``top_evidence_cameras``.
    Returns mean cameras per question, fraction of questions with ≥2 cameras,
    and the full count distribution.
    """
    if not traces:
        return {
            "mean_cameras_per_question": 0.0,
            "pct_multi_camera": 0.0,
            "camera_count_distribution": {},
        }

    counts: List[int] = []
    for trace in traces:
        cameras = trace.get("top_evidence_cameras") or []
        counts.append(len({c for c in cameras if c}))

    total = len(counts)
    dist: Dict[int, int] = {}
    for c in counts:
        dist[c] = dist.get(c, 0) + 1

    return {
        "mean_cameras_per_question": sum(counts) / total,
        "pct_multi_camera": sum(1 for c in counts if c >= 2) / total,
        "camera_count_distribution": dist,
    }


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
