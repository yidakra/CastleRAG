"""Tests for CASTLE LoRA supervision guardrails and formatting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from castlerag.training.lora_mcqa import (
    LoRABlockedError,
    LoRASplitPaths,
    check_lora_prerequisites,
    load_supervised_split,
    prepare_lora_datasets,
    train_lora,
    validate_lora_prerequisites,
)


def _write_questions(path: Path) -> Path:
    questions = {
        "q1": {
            "query": "What did Alex pick up from the kitchen table?",
            "answers": {
                "a": "A mug",
                "b": "A laptop",
                "c": "A notebook",
                "d": "A jacket",
            },
        },
        "q2": {
            "query": "Where did Jordan leave the bag?",
            "answers": {
                "a": "Bedroom",
                "b": "Office",
                "c": "Kitchen",
                "d": "Hallway",
            },
        },
    }
    path.write_text(json.dumps(questions))
    return path


def _write_answers(path: Path, mapping: dict[str, str]) -> Path:
    path.write_text(json.dumps(mapping))
    return path


def test_validate_lora_prerequisites_lists_missing_inputs():
    with pytest.raises(LoRABlockedError, match="train_questions_path"):
        validate_lora_prerequisites(
            train_questions_path=None,
            train_answers_path=None,
            val_questions_path=None,
            val_answers_path=None,
        )


def test_check_lora_prerequisites_false_when_files_absent(tmp_path: Path):
    assert (
        check_lora_prerequisites(
            train_questions_path=tmp_path / "train_questions.json",
            train_answers_path=tmp_path / "train_answers.json",
            val_questions_path=tmp_path / "val_questions.json",
            val_answers_path=tmp_path / "val_answers.json",
        )
        is False
    )


def test_prepare_lora_datasets_formats_answer_only_examples(tmp_path: Path):
    train_q = _write_questions(tmp_path / "train_questions.json")
    train_a = _write_answers(tmp_path / "train_answers.json", {"q1": "b", "q2": "d"})
    val_q = _write_questions(tmp_path / "val_questions.json")
    val_a = _write_answers(tmp_path / "val_answers.json", {"q1": "a", "q2": "c"})

    datasets = prepare_lora_datasets(
        train_questions_path=train_q,
        train_answers_path=train_a,
        val_questions_path=val_q,
        val_answers_path=val_a,
    )

    assert len(datasets["train"]) == 2
    first = datasets["train"][0]
    assert first.question_id == "q1"
    assert first.target_answer == "b"
    assert first.prompt_messages[0]["role"] == "system"
    assert "Output only one lowercase letter" in first.prompt_messages[0]["content"]
    assert "Question: What did Alex pick up" in first.prompt_messages[1]["content"]
    assert "A. A mug" in first.prompt_messages[1]["content"]
    assert "Respond with exactly one lowercase letter" in first.prompt_messages[1][
        "content"
    ]


def test_load_supervised_split_rejects_missing_labels(tmp_path: Path):
    questions_path = _write_questions(tmp_path / "train_questions.json")
    answers_path = _write_answers(tmp_path / "train_answers.json", {"q1": "a"})

    with pytest.raises(LoRABlockedError, match="missing labels"):
        load_supervised_split(
            LoRASplitPaths(
                split_name="train",
                questions_path=questions_path,
                answers_path=answers_path,
            )
        )


def test_load_supervised_split_rejects_invalid_choice(tmp_path: Path):
    questions_path = _write_questions(tmp_path / "train_questions.json")
    answers_path = _write_answers(
        tmp_path / "train_answers.json",
        {"q1": "a", "q2": "e"},
    )

    with pytest.raises(LoRABlockedError, match="invalid choice"):
        load_supervised_split(
            LoRASplitPaths(
                split_name="train",
                questions_path=questions_path,
                answers_path=answers_path,
            )
        )


def test_train_lora_fails_fast_without_supervision(tmp_path: Path):
    with pytest.raises(LoRABlockedError, match="missing labeled CASTLE QA supervision"):
        train_lora(
            base_model="Qwen/Qwen3-VL-8B-Instruct",
            train_questions_path=None,
            train_answers_path=None,
            val_questions_path=None,
            val_answers_path=None,
            output_dir=tmp_path / "out",
        )


def test_train_lora_stops_at_scaffold_after_validation(tmp_path: Path):
    train_q = _write_questions(tmp_path / "train_questions.json")
    train_a = _write_answers(tmp_path / "train_answers.json", {"q1": "a", "q2": "b"})
    val_q = _write_questions(tmp_path / "val_questions.json")
    val_a = _write_answers(tmp_path / "val_answers.json", {"q1": "c", "q2": "d"})

    with pytest.raises(
        NotImplementedError,
        match="PEFT training job is not implemented",
    ):
        train_lora(
            base_model="Qwen/Qwen3-VL-8B-Instruct",
            train_questions_path=train_q,
            train_answers_path=train_a,
            val_questions_path=val_q,
            val_answers_path=val_a,
            output_dir=tmp_path / "out",
        )
