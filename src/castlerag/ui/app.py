"""Dash application factory and dev-server entrypoint for the CastleRAG UI.

``build_app`` assembles the layout, mirror, and engine and registers callbacks,
defaulting to the offline :class:`~castlerag.ui.chat.PlaceholderEngine` so the
dashboard runs with no models, Qdrant, or vLLM.  A future ``RagEngine`` is
injected via the ``engine`` argument without touching the layout or callbacks.

The whole layout is wrapped in a :class:`dmc.MantineProvider`; every visual
surface is a Dash Mantine Component themed from :data:`THEME` (Dash Mantine
Components ships its CSS in its JS bundle, so no external stylesheet is needed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from castlerag.ui.callbacks import register_callbacks
from castlerag.ui.chat import ChatEngine
from castlerag.ui.layout import build_layout
from castlerag.ui.youtube import YouTubeMirror

if TYPE_CHECKING:
    from dash import Dash

# Single source of truth for the Mantine theme (indigo accent, soft radius),
# matching the dashboard's original light/indigo look.
THEME = {
    "primaryColor": "indigo",
    "primaryShade": 6,
    "defaultRadius": "md",
    "fontFamily": "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
    "fontFamilyMonospace": "ui-monospace, SFMono-Regular, Menlo, monospace",
}


def build_app(
    engine: Optional[ChatEngine] = None,
    mirror: Optional[YouTubeMirror] = None,
) -> "Dash":
    """Build and return the configured Dash app (callbacks registered)."""
    import dash_mantine_components as dmc
    from dash import Dash

    from castlerag.ui.engine_factory import build_engine, engine_mode

    mirror = mirror or YouTubeMirror.from_csv()
    engine = engine or build_engine(mirror)

    # Read score_mode from engine config when live, else load it from base.yaml so
    # offline mode still honours whatever is set in the config file.
    import os

    from castlerag.config import load_config
    cfg = getattr(engine, "cfg", None)
    if cfg is None:
        cfg = load_config(override_path=os.getenv("CASTLERAG_CONFIG"))
    score_mode: str = getattr(getattr(cfg, "ui", None), "score_mode", "rrf_normalized")

    app = Dash(__name__, title="CastleRAG")
    app.layout = dmc.MantineProvider(
        build_layout(mirror, mode=engine_mode(engine), score_mode=score_mode),
        theme=THEME,
        forceColorScheme="light",
    )
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
