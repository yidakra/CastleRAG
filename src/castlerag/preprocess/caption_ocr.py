"""Offline frame captioning and OCR over representative clip frames.

Per-clip inputs: 30 sampled JPEG frames at 1 fps + optional transcript text.
Outputs per clip: clip_caption, ocr_text, caption_confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ClipAnnotation:
    clip_id: str
    clip_caption: Optional[str]
    ocr_text: Optional[str]
    caption_confidence: float


def annotate_clip(
    clip_id: str,
    frame_paths: List[Path],
    transcript_text: Optional[str],
    model_name: str,
    vllm_base_url: Optional[str] = None,
) -> ClipAnnotation:
    """Generate clip_caption and ocr_text for a 30-second clip.

    Captions should emphasise people, objects, actions, room cues, and
    visible text/screens.  OCR is run on frames where text is likely.
    """
    raise NotImplementedError("Implemented in issue #4")
