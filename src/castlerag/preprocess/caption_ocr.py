"""Offline frame captioning and OCR over representative clip frames.

Per-clip inputs: 30 sampled JPEG frames at 1 fps + optional transcript text.
Outputs per clip: clip_caption, ocr_text, caption_confidence.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ClipAnnotation:
    clip_id: str
    clip_caption: Optional[str]
    ocr_text: Optional[str]
    caption_confidence: float


def _vllm_chat(
    vllm_base_url: str,
    model: str,
    messages: list,
    max_tokens: int = 256,
    timeout: float = 120.0,
) -> str:
    """Send a chat completion request to a vLLM OpenAI-compatible endpoint."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package required for caption/OCR; pip install castlerag[inference]"
        ) from exc
    client = OpenAI(base_url=vllm_base_url, api_key="not-needed", timeout=timeout)
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens
    )
    if not resp.choices:
        return ""
    content = resp.choices[0].message.content
    return content.strip() if content else ""


def annotate_clip(
    clip_id: str,
    frame_paths: List[Path],
    transcript_text: Optional[str],
    model_name: str,
    vllm_base_url: Optional[str] = None,
) -> ClipAnnotation:
    """Generate clip_caption and ocr_text for a 30-second clip.

    Up to 8 evenly-spaced frames are included in the captioning prompt to stay
    within model context limits.  OCR is run on the first frame as a quick pass;
    a more thorough pass can be added per-frame once model throughput is known.

    Captions emphasise people, objects, actions, room cues, and visible text/screens.
    """
    if not vllm_base_url:
        raise ValueError("vllm_base_url is required for annotate_clip")
    if not frame_paths:
        return ClipAnnotation(
            clip_id=clip_id, clip_caption=None, ocr_text=None, caption_confidence=0.0
        )

    # Sample up to 8 evenly-spaced frames
    step = max(1, len(frame_paths) // 8)
    sample = frame_paths[::step][:8]

    content: list = []
    for fp in sample:
        img_b64 = base64.b64encode(fp.read_bytes()).decode()
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )

    caption_prompt = (
        "Describe this video clip in 1-2 sentences, emphasising people, objects, "
        "actions, room cues, and any visible text or screens."
    )
    if transcript_text:
        caption_prompt += f"\n\nTranscript: {transcript_text}"
    content.append({"type": "text", "text": caption_prompt})

    caption = _vllm_chat(
        vllm_base_url, model_name, [{"role": "user", "content": content}]
    )

    # OCR pass on the first frame
    ocr_text: Optional[str] = None
    first_frame = frame_paths[0]
    img_b64 = base64.b64encode(first_frame.read_bytes()).decode()
    ocr_response = _vllm_chat(
        vllm_base_url,
        model_name,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract any visible text from this frame exactly as it "
                            "appears. Return only the text, or 'NONE' if no "
                            "text is visible."
                        ),
                    },
                ],
            }
        ],
        max_tokens=128,
    )
    if ocr_response and ocr_response.upper() != "NONE":
        ocr_text = ocr_response

    return ClipAnnotation(
        clip_id=clip_id,
        clip_caption=caption or None,
        ocr_text=ocr_text,
        caption_confidence=1.0 if caption else 0.0,
    )
