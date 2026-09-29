"""Shared text budgeting for evidence rendered into reranker/generator prompts.

Clip captions and scene-graph strings come straight from a VLM and are not
length-bounded at index time. Both prompt builders truncate them with the same
limits so one runaway annotation cannot blow the prompt budget.
"""

from __future__ import annotations

from typing import Optional

# Caption is 1-2 sentences by prompt design; scene graph is a semicolon list
# capped at 128 tokens. Limits are in characters and deliberately generous.
MAX_CAPTION_CHARS = 600
MAX_SCENE_GRAPH_CHARS = 400
_ELLIPSIS = "..."


def truncate_text(text: Optional[str], limit: int) -> Optional[str]:
    """Return ``text`` stripped and cut to ``limit`` chars with an ellipsis.

    ``None``/blank input returns ``None`` so callers can skip empty fields.
    """
    if text is None:
        return None
    cleaned = " ".join(text.split())
    if not cleaned:
        return None
    if len(cleaned) <= limit:
        return cleaned
    if limit <= len(_ELLIPSIS):
        return cleaned[:limit]
    return cleaned[: limit - len(_ELLIPSIS)].rstrip() + _ELLIPSIS
