"""Offline frame captioning and OCR over representative clip frames.

Per-clip inputs: 30 sampled JPEG frames at 1 fps + optional transcript text.
Outputs per clip: clip_caption, ocr_text, caption_confidence.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Cap the longer edge of each frame before sending to the VLM.  Qwen-VL counts
# image tokens proportionally to pixel area, so a 1080p source frame can cost
# 8k+ tokens on its own — blowing past typical context budgets in a multi-frame
# request.  448 px keeps captions readable while staying under ~700 tokens/img.
_MAX_FRAME_EDGE = 448
# Number of frames per multi-image caption request.  Lower = fewer tokens per
# call.  Eight 1080p frames was ~65k tokens; four 448px frames is ~3k.
_CAPTION_NUM_FRAMES = 4


def _frame_to_b64(path: Path, max_edge: int = _MAX_FRAME_EDGE) -> str:
    """Return a base64-encoded JPEG of the frame, downsized to max_edge."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_edge / max(w, h))
        if scale < 1.0:
            # Clamp to >=1 px so extremely thin frames don't crash resize().
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            im = im.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()


@dataclass
class ClipAnnotation:
    clip_id: str
    clip_caption: Optional[str]
    ocr_text: Optional[str]
    caption_confidence: float
    scene_graph_text: Optional[str] = None


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

    # Sample up to _CAPTION_NUM_FRAMES evenly-spaced frames
    step = max(1, len(frame_paths) // _CAPTION_NUM_FRAMES)
    sample = frame_paths[::step][:_CAPTION_NUM_FRAMES]

    content: list = []
    for fp in sample:
        img_b64 = _frame_to_b64(fp)
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
    img_b64 = _frame_to_b64(first_frame)
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

    # Scene graph pass: objects and spatial relationships for richer retrieval
    scene_graph_text: Optional[str] = None
    sg_content: list = []
    for fp in sample:
        img_b64 = _frame_to_b64(fp)
        sg_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )
    sg_content.append(
        {
            "type": "text",
            "text": (
                "List the main objects and people visible in this clip with their "
                "approximate positions (left, center, right, foreground, background). "
                "Use short phrases separated by semicolons, e.g. "
                "'person at counter (left); pan on stove (center)'. "
                "Return only the list, or 'NONE' if the scene is empty or unclear."
            ),
        }
    )
    sg_response = _vllm_chat(
        vllm_base_url,
        model_name,
        [{"role": "user", "content": sg_content}],
        max_tokens=128,
    )
    if sg_response and sg_response.upper() != "NONE":
        scene_graph_text = sg_response

    return ClipAnnotation(
        clip_id=clip_id,
        clip_caption=caption or None,
        ocr_text=ocr_text,
        scene_graph_text=scene_graph_text,
        caption_confidence=1.0 if caption else 0.0,
    )
