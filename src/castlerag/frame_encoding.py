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
from typing import List, Optional, Sequence, Tuple

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


_FRAME_PATH_ALIASES: List[Tuple[str, str]] = []


def set_frame_path_aliases(aliases: "dict[str, str]") -> None:
    """Register old-prefix -> new-prefix rewrites for stored frame paths.

    Called by ``load_config`` from ``preprocessing.frame_path_aliases``. The
    longest matching prefix wins.
    """
    _FRAME_PATH_ALIASES[:] = sorted(
        ((old.rstrip("/"), new.rstrip("/")) for old, new in aliases.items()),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )


def relocate_frame_path(path: str) -> str:
    """Apply the first matching alias to ``path`` (no existence check)."""
    for old, new in _FRAME_PATH_ALIASES:
        if path == old or path.startswith(old + "/"):
            return new + path[len(old) :]
    return path


def resolve_frame_path(path: str) -> str:
    """Return where a stored frame path lives now.

    The stored path if it exists, else its aliased location if that exists,
    else the stored path unchanged (callers treat it as missing).
    """
    if Path(path).exists():
        return path
    moved = relocate_frame_path(path)
    if moved != path and Path(moved).exists():
        return moved
    return path


def available_frames(paths: Sequence[str]) -> List[str]:
    """Return the frames in ``paths`` that still exist on disk.

    Frames may be thinned after a day is embedded (see scripts/thin_frames.py)
    while older Qdrant payloads still list every sampled frame. Sampling from
    the full list would then pick positions whose files are gone, so readers
    sample from the surviving frames instead. If none of the paths exist (for
    example the day's frames were never kept, or the paths are synthetic in
    tests) the list is returned unchanged and callers skip unreadable files as
    before.
    """
    frames = [resolve_frame_path(p) for p in paths]
    existing = [p for p in frames if Path(p).exists()]
    return existing if existing else list(paths)


def sample_frames_evenly(paths: Sequence[str], max_frames: int) -> List[str]:
    """Pick up to ``max_frames`` paths spread evenly across ``paths``.

    Clips are sampled at 1 fps, so ``paths[:n]`` is the first ``n`` seconds of
    a 30 s clip: an object shown mid-clip (a t-shirt back, a fridge logo) is
    never seen. This picks the centre of ``max_frames`` equal-width buckets
    instead, so 4 of 30 frames become seconds 3, 11, 18, 26. Order is
    preserved; ``max_frames <= 0`` returns an empty list and a list already
    within the cap is returned unchanged.
    """
    if max_frames <= 0:
        return []
    frames = list(paths)
    if len(frames) <= max_frames:
        return frames
    total = len(frames)
    return [frames[int((i + 0.5) * total / max_frames)] for i in range(max_frames)]


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
