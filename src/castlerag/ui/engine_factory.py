"""Select the dashboard's chat engine: real RAG when reachable, else offline.

``build_engine`` tries to build the real :class:`~castlerag.ui.rag_engine.RagEngine`
(which needs ``VLLM_BASE_URL`` set plus a reachable Qdrant + built index). On any
failure it falls back to the offline :class:`~castlerag.ui.chat.PlaceholderEngine`
so the dashboard always boots — important on hosts where the backend isn't up.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from typing import Any, Optional

log = logging.getLogger("castlerag.ui")

# Short budget for the liveness probe; a real local vLLM answers in well under this.
_PROBE_TIMEOUT_SECONDS = 3.0


def _vllm_reachable(base_url: str, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
    """Return True if the vLLM/OpenAI server at ``base_url`` answers ``GET /models``.

    A connection-level failure (refused/DNS/timeout) means the server is down, so we
    fall back. Any HTTP response — including a non-2xx status — means a server *is*
    listening, which is all this gate needs to confirm.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout):  # noqa: S310 - local infra URL
            return True
    except urllib.error.HTTPError:
        return True  # server responded (any status) -> it is up
    except Exception:  # noqa: BLE001 - any connection failure means "unreachable"
        return False


def build_engine(mirror: object, cfg: Optional[Any] = None) -> Any:
    """Return the real RagEngine when infra is reachable, else PlaceholderEngine."""
    from castlerag.ui.chat import PlaceholderEngine

    base_url = os.getenv("VLLM_BASE_URL")
    if not base_url:
        log.info("VLLM_BASE_URL unset; using offline PlaceholderEngine.")
        return PlaceholderEngine.from_mirror(mirror)
    if not _vllm_reachable(base_url):
        log.warning(
            "VLLM_BASE_URL=%s not reachable; using offline PlaceholderEngine.",
            base_url,
        )
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
