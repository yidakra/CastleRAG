"""Dash application factory and dev-server entrypoint for the CastleRAG UI.

``build_app`` assembles the layout, mirror, and engine and registers callbacks,
defaulting to the offline :class:`~castlerag.ui.chat.PlaceholderEngine` so the
dashboard runs with no models, Qdrant, or vLLM.  A future ``RagEngine`` is
injected via the ``engine`` argument without touching the layout or callbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from castlerag.ui.callbacks import register_callbacks
from castlerag.ui.chat import ChatEngine
from castlerag.ui.layout import build_layout
from castlerag.ui.youtube import YouTubeMirror

if TYPE_CHECKING:
    from dash import Dash


def build_app(
    engine: Optional[ChatEngine] = None,
    mirror: Optional[YouTubeMirror] = None,
) -> "Dash":
    """Build and return the configured Dash app (callbacks registered)."""
    from dash import Dash

    from castlerag.ui.engine_factory import build_engine, engine_mode

    mirror = mirror or YouTubeMirror.from_csv()
    engine = engine or build_engine(mirror)

    app = Dash(__name__, title="CastleRAG")
    app.layout = build_layout(mirror, mode=engine_mode(engine))
    register_callbacks(app, engine, mirror)
    return app


def run(
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Launch the Dash development server."""
    build_app().run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run(debug=True)
