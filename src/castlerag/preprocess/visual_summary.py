"""Offline visual summaries for chunks using the selected open-weight VL model."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def generate_visual_summary(
    frame_paths: List[Path],
    transcript_text: Optional[str],
    model_name: str,
    vllm_base_url: Optional[str] = None,
) -> str:
    """Return a compact visual summary string for a clip's sampled frames."""
    raise NotImplementedError("Implemented in issue #4")
