"""Token-bounded multimodal frame encoding.

Frames sampled from CASTLE videos are stored at full capture resolution. Qwen3-VL
turns each image into visual tokens roughly proportional to its pixel area (one
token per 28x28 patch), so a handful of full-resolution frames can push a prompt
past the model's context window — exactly the failure (66k-72k-token generation
prompts against a 49k limit) that silently dropped 21/40 questions in an earlier
eval.

This module downscales each frame to a bounded longest edge before base64
encoding, which caps per-image token cost to a small known value, and exposes
cheap token estimators so callers can pack frames (and trim evidence text) under
an explicit prompt-token budget instead of overflowing the server.
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

# Qwen-family vision transformers merge 14px patches 2x2, so one visual token
# corresponds to a ~28x28 pixel region. The estimate ceils on both axes, making
# it a deliberate over-estimate so token budgeting stays conservative.
_PATCH = 28


def estimate_text_tokens(text: str) -> int:
    """Rough upper-bound token count for a text string (~4 chars per token)."""
    return len(text) // 4 + 1


def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate Qwen-VL visual tokens for an image of the given pixel dimensions."""
    return math.ceil(width / _PATCH) * math.ceil(height / _PATCH)


def encode_frame(
    path: str, max_pixels: int = 768, quality: int = 85
) -> Optional[Tuple[str, int]]:
    """Downscale a frame to ``max_pixels`` on its longest edge and base64-encode it.

    Returns ``(base64_jpeg, estimated_visual_tokens)``, or ``None`` if the file is
    missing or unreadable. Frames already within the bound are re-encoded to JPEG
    for a predictable on-wire size.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            longest = max(w, h)
            if longest > max_pixels:
                scale = max_pixels / longest
                w, h = max(1, round(w * scale)), max(1, round(h * scale))
                im = im.resize((w, h), Image.BILINEAR)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
    except (OSError, ValueError):
        return None
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, estimate_image_tokens(w, h)
