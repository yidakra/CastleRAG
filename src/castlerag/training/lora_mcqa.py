"""LoRA fine-tuning and held-out evaluation for CASTLE multiple-choice answering.

BLOCKED until a CASTLE QA train/validation split is explicitly confirmed.
See SPEC §9.1 and config.lora.enabled.

Target: answer-format alignment only (emit 'a'|'b'|'c'|'d', not long explanations).
Eval metric: exact-match accuracy on a held-out validation split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def check_lora_prerequisites(
    train_path: Optional[Path],
    val_path: Optional[Path],
) -> bool:
    """Return True only if real QA train and val splits exist on disk.

    This check is the gate that must pass before any LoRA training starts.
    Do not add training logic until this returns True in a real run.
    """
    if train_path is None or val_path is None:
        return False
    return train_path.is_file() and val_path.is_file()


def train_lora(
    base_model: str,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    rank: int = 16,
    alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2.0e-4,
) -> None:
    """Fine-tune base_model with LoRA on CASTLE QA data."""
    if not check_lora_prerequisites(train_path, val_path):
        raise RuntimeError(
            "LoRA training is blocked: no CASTLE QA train/val split found. "
            "Locate the official split before enabling lora.enabled in config."
        )
    raise NotImplementedError(
        "Implemented only after QA split is confirmed (SPEC §9.1)"
    )
