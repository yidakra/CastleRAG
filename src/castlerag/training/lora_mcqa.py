"""LoRA fine-tuning scaffold for CASTLE multiple-choice answer alignment.

The live blocker is still supervision: this repo does not include a checked-in
CASTLE QA train/validation split. The code below therefore focuses on:

- precise prerequisite validation
- loading question + answer-key supervision when explicitly provided
- formatting answer-only MCQA examples for a future PEFT/LoRA job

It intentionally does not implement the actual training loop yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from castlerag.eval.io import load_questions
from castlerag.schemas import AnswerChoice, EvalQuestion

_MCQA_SYSTEM_PROMPT = (
    "You answer CASTLE multiple-choice questions. "
    "Output only one lowercase letter: a, b, c, or d."
)


class LoRABlockedError(RuntimeError):
    """Raised when LoRA training cannot proceed from local data/contracts."""


@dataclass(frozen=True)
class LoRASplitPaths:
    """Local labeled split contract for answer-format LoRA."""

    split_name: str
    questions_path: Path
    answers_path: Path


@dataclass(frozen=True)
class LoRASupervisionPaths:
    """Train/validation supervision files required for LoRA."""

    train: LoRASplitPaths
    val: LoRASplitPaths


@dataclass(frozen=True)
class LoRATrainingExample:
    """Formatted MCQA example for answer-only alignment."""

    question_id: str
    prompt_messages: List[Dict[str, str]]
    target_answer: AnswerChoice


def _format_question_block(question: EvalQuestion) -> str:
    return (
        f"Question: {question.query}\n"
        f"A. {question.answers['a']}\n"
        f"B. {question.answers['b']}\n"
        f"C. {question.answers['c']}\n"
        f"D. {question.answers['d']}\n\n"
        "Respond with exactly one lowercase letter: a, b, c, or d."
    )


def _validate_answer_key(
    questions: Dict[str, EvalQuestion],
    answers: Dict[str, Any],
    *,
    split_name: str,
) -> Dict[str, AnswerChoice]:
    normalized: Dict[str, AnswerChoice] = {}
    missing_labels = [qid for qid in questions if qid not in answers]
    if missing_labels:
        missing = ", ".join(missing_labels[:5])
        raise LoRABlockedError(
            f"LoRA training is blocked: {split_name} answer key is missing labels "
            f"for question ids: {missing}"
        )

    for qid in questions:
        raw = answers[qid]
        if not isinstance(raw, str):
            raise LoRABlockedError(
                f"LoRA training is blocked: {split_name} answer key for {qid!r} "
                "must be a string choice in {'a','b','c','d'}."
            )
        choice = raw.strip().lower()
        if choice not in {"a", "b", "c", "d"}:
            raise LoRABlockedError(
                f"LoRA training is blocked: {split_name} answer key for {qid!r} "
                f"contains invalid choice {raw!r}; expected one of a, b, c, d."
            )
        normalized[qid] = cast(AnswerChoice, choice)
    return normalized


def validate_lora_prerequisites(
    *,
    train_questions_path: Optional[Path],
    train_answers_path: Optional[Path],
    val_questions_path: Optional[Path],
    val_answers_path: Optional[Path],
) -> LoRASupervisionPaths:
    """Return validated supervision paths or raise a precise blocker error."""

    required = {
        "train_questions_path": train_questions_path,
        "train_answers_path": train_answers_path,
        "val_questions_path": val_questions_path,
        "val_answers_path": val_answers_path,
    }
    missing_args = [name for name, path in required.items() if path is None]
    if missing_args:
        missing = ", ".join(missing_args)
        raise LoRABlockedError(
            "LoRA training is blocked: missing labeled CASTLE QA supervision "
            f"inputs: {missing}. Supply real train/validation question and "
            "answer-key files before enabling LoRA."
        )

    missing_files = [name for name, path in required.items() if not path.is_file()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise LoRABlockedError(
            "LoRA training is blocked: expected supervision files do not exist: "
            f"{missing}."
        )

    return LoRASupervisionPaths(
        train=LoRASplitPaths(
            split_name="train",
            questions_path=train_questions_path,
            answers_path=train_answers_path,
        ),
        val=LoRASplitPaths(
            split_name="val",
            questions_path=val_questions_path,
            answers_path=val_answers_path,
        ),
    )


def check_lora_prerequisites(
    *,
    train_questions_path: Optional[Path],
    train_answers_path: Optional[Path],
    val_questions_path: Optional[Path],
    val_answers_path: Optional[Path],
) -> bool:
    """Return True only if labeled train/val supervision files exist."""

    try:
        validate_lora_prerequisites(
            train_questions_path=train_questions_path,
            train_answers_path=train_answers_path,
            val_questions_path=val_questions_path,
            val_answers_path=val_answers_path,
        )
    except LoRABlockedError:
        return False
    return True


def load_supervised_split(split: LoRASplitPaths) -> List[LoRATrainingExample]:
    """Load question JSON plus answer-key JSON into answer-only MCQA examples."""

    questions = load_questions(split.questions_path)
    try:
        answers = json.loads(split.answers_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LoRABlockedError(
            f"LoRA training is blocked: failed to read/parse "
            f"{split.split_name} answer key JSON at {split.answers_path}."
        ) from exc
    if not isinstance(answers, dict):
        raise LoRABlockedError(
            f"LoRA training is blocked: {split.split_name} answer key must be a "
            "JSON object mapping question id to a|b|c|d."
        )

    normalized_answers = _validate_answer_key(
        questions,
        answers,
        split_name=split.split_name,
    )

    examples: List[LoRATrainingExample] = []
    for qid, question in questions.items():
        examples.append(
            LoRATrainingExample(
                question_id=qid,
                prompt_messages=[
                    {"role": "system", "content": _MCQA_SYSTEM_PROMPT},
                    {"role": "user", "content": _format_question_block(question)},
                ],
                target_answer=normalized_answers[qid],
            )
        )
    return examples


def prepare_lora_datasets(
    *,
    train_questions_path: Optional[Path],
    train_answers_path: Optional[Path],
    val_questions_path: Optional[Path],
    val_answers_path: Optional[Path],
) -> Dict[str, List[LoRATrainingExample]]:
    """Load and validate local LoRA supervision into formatted train/val splits."""

    supervision = validate_lora_prerequisites(
        train_questions_path=train_questions_path,
        train_answers_path=train_answers_path,
        val_questions_path=val_questions_path,
        val_answers_path=val_answers_path,
    )
    return {
        "train": load_supervised_split(supervision.train),
        "val": load_supervised_split(supervision.val),
    }


def train_lora(
    *,
    base_model: str,
    train_questions_path: Optional[Path],
    train_answers_path: Optional[Path],
    val_questions_path: Optional[Path],
    val_answers_path: Optional[Path],
    output_dir: Path,
    rank: int = 16,
    alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2.0e-4,
) -> None:
    """Validate supervision and stop before the not-yet-implemented PEFT job."""

    prepare_lora_datasets(
        train_questions_path=train_questions_path,
        train_answers_path=train_answers_path,
        val_questions_path=val_questions_path,
        val_answers_path=val_answers_path,
    )

    raise NotImplementedError(
        "LoRA training scaffold is ready, but the actual PEFT training job is "
        "not implemented yet. Current local blocker for real runs remains the "
        "absence of a checked-in CASTLE QA train/validation split."
    )
