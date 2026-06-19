"""CastleRAG Dash UI: chat + YouTube evidence embeds + Plotly analytics shell.

This is the placeholder backbone (meeting note item 1): the whole UI runs end to
end on a stub :class:`~castlerag.ui.chat.PlaceholderEngine`, with no RAG, models,
Qdrant, or vLLM wired in.  The real pipeline drops in later behind the
:class:`~castlerag.ui.chat.ChatEngine` protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from castlerag.ui.chat import (
    ChatEngine,
    ChatTurnResult,
    EvidenceRef,
    PlaceholderEngine,
)
from castlerag.ui.youtube import YouTubeMirror

if TYPE_CHECKING:
    from dash import Dash

__all__ = [
    "ChatEngine",
    "ChatTurnResult",
    "EvidenceRef",
    "PlaceholderEngine",
    "YouTubeMirror",
    "build_app",
]


def build_app(
    engine: Optional[ChatEngine] = None,
    mirror: Optional[YouTubeMirror] = None,
) -> "Dash":
    """Lazily import and call :func:`castlerag.ui.app.build_app`.

    Keeps ``import castlerag.ui`` free of the optional Dash dependency so the
    pure-Python pieces (mirror, engine) import without the ``ui`` extra.
    """
    from castlerag.ui.app import build_app as _build_app

    return _build_app(engine, mirror)
