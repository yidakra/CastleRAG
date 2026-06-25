"""CastleRAG Dash UI: chat + synchronized YouTube embeds + a Plotly score chart.

The whole UI runs end to end on a stub :class:`~castlerag.ui.chat.PlaceholderEngine`
(no RAG, models, Qdrant, or vLLM required); the real
:class:`~castlerag.ui.rag_engine.RagEngine` drops in behind the
:class:`~castlerag.ui.chat.ChatEngine` protocol when the backend is reachable.
The right-column viewer pairs the three synchronized camera embeds with a Plotly
per-camera match-score bar (:mod:`castlerag.ui.figures`).
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
