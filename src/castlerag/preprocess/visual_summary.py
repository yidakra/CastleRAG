"""Offline visual summaries for chunks using the selected open-weight VL model."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from castlerag.preprocess.caption_ocr import _vllm_chat
import base64


def generate_visual_summary(
    frame_paths: List[Path],
    transcript_text: Optional[str],
    model_name: str,
    vllm_base_url: Optional[str] = None,
) -> str:
    """Return a compact visual summary string for a clip's sampled frames.

    Used as input to the event-compression step.  Up to 8 evenly-spaced frames
    are included.  Returns an empty string when frame_paths is empty.
    """
    if not vllm_base_url:
        raise ValueError("vllm_base_url is required for generate_visual_summary")
    if not frame_paths:
        return ""

    step = max(1, len(frame_paths) // 8)
    sample = frame_paths[::step][:8]

    content: list = []
    for fp in sample:
        img_b64 = base64.b64encode(fp.read_bytes()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    prompt = "Summarise what is happening in this video clip in one concise sentence."
    if transcript_text:
        prompt += f"\n\nTranscript: {transcript_text}"
    content.append({"type": "text", "text": prompt})

    return _vllm_chat(vllm_base_url, model_name, [{"role": "user", "content": content}])
