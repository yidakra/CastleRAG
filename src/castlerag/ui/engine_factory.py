"""Select the dashboard's chat engine: real RAG when reachable, else offline.

``build_engine`` tries to build the real :class:`~castlerag.ui.rag_engine.RagEngine`
(which needs ``VLLM_BASE_URL`` set plus a reachable Qdrant + built index). On any
failure it falls back to the offline :class:`~castlerag.ui.chat.PlaceholderEngine`
so the dashboard always boots — important on hosts where the backend isn't up.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger("castlerag.ui")


def build_engine(mirror: object, cfg: Optional[Any] = None) -> Any:
    """Return the real RagEngine when infra is reachable, else PlaceholderEngine."""
    from castlerag.ui.chat import PlaceholderEngine

    if not os.getenv("VLLM_BASE_URL"):
        log.info("VLLM_BASE_URL unset; using offline PlaceholderEngine.")
        return PlaceholderEngine.from_mirror(mirror)
    try:
        from castlerag.ui.rag_engine import RagEngine

        engine = RagEngine.from_config(cfg=cfg, mirror=mirror)
        log.info("RagEngine active (real route->retrieve->rerank->generate).")
        return engine
    except Exception as exc:  # noqa: BLE001 - any infra failure must not crash the UI
        log.warning(
            "RagEngine unavailable (%s: %s); using offline PlaceholderEngine.",
            type(exc).__name__,
            exc,
        )
        return PlaceholderEngine.from_mirror(mirror)


def engine_mode(engine: object) -> str:
    """Return ``"live"`` for the real engine, ``"offline"`` otherwise."""
    return "live" if getattr(engine, "is_live", False) else "offline"
