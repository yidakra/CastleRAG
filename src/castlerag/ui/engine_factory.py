"""Select the dashboard's chat engine: real RAG when reachable, else offline.

``build_engine`` tries to build the real :class:`~castlerag.ui.rag_engine.RagEngine`
(which needs ``VLLM_BASE_URL`` set plus a reachable Qdrant + built index). On any
failure it falls back to the offline :class:`~castlerag.ui.chat.PlaceholderEngine`
so the dashboard always boots — important on hosts where the backend isn't up.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from typing import Any, Optional

log = logging.getLogger("castlerag.ui")

# Short budget for the liveness probe; a real local vLLM answers in well under this.
_PROBE_TIMEOUT_SECONDS = 3.0

# Budget for the 1-token generation completion probe. A served-model mismatch or a
# stalled endpoint must not hang startup, so the request itself is time-bounded.
_GEN_PROBE_TIMEOUT_SECONDS = 15.0


class EngineUnavailable(RuntimeError):
    """Raised when ``require_live`` is set but the real backend cannot be built.

    Carries a precise, actionable reason (which dependency is missing / wrong) so
    the CLI can print it instead of silently serving the offline demo.
    """


def _vllm_reachable(base_url: str, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
    """Return True if the vLLM/OpenAI server at ``base_url`` answers ``GET /models``.

    A connection-level failure (refused/DNS/timeout) means the server is down. A
    non-2xx HTTP status (404/401/500) means *something* is listening but the model
    API is not actually usable, so we also treat that as unreachable and fall back
    to offline mode rather than deferring the failure to query time. Only a 2xx
    response counts as live.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout):  # noqa: S310 - local infra URL
            return True
    except Exception:  # noqa: BLE001 - HTTPError or connection failure -> unreachable
        return False


def _fallback_or_raise(reason: str, mirror: object, require_live: bool) -> Any:
    """Either raise ``EngineUnavailable`` (strict) or return the offline engine."""
    if require_live:
        raise EngineUnavailable(reason)
    from castlerag.ui.chat import PlaceholderEngine

    log.warning("%s Using offline PlaceholderEngine.", reason)
    return PlaceholderEngine.from_mirror(mirror)


def build_engine(
    mirror: object, cfg: Optional[Any] = None, require_live: bool = False
) -> Any:
    """Return the real RagEngine when infra is reachable, else PlaceholderEngine.

    With ``require_live=True`` the function never falls back: if the real backend
    cannot be built it raises :class:`EngineUnavailable` with the precise reason
    (so the caller can surface it), instead of quietly serving the demo engine.
    """
    base_url = os.getenv("VLLM_BASE_URL")
    if not base_url:
        return _fallback_or_raise(
            "VLLM_BASE_URL is not set — point it at the Qwen3-VL vLLM endpoint "
            "(e.g. export VLLM_BASE_URL=http://<host>:8201/v1).",
            mirror,
            require_live,
        )
    if not _vllm_reachable(base_url):
        return _fallback_or_raise(
            f"vLLM endpoint {base_url} is not reachable (GET /models failed) — "
            "is the Qwen3-VL server up and is the host reachable from here?",
            mirror,
            require_live,
        )
    try:
        from castlerag.ui.rag_engine import RagEngine

        engine = RagEngine.from_config(cfg=cfg, mirror=mirror)
        # Catch a served-model-name / config mismatch (and other call-time faults)
        # up front rather than on the first question. This runs in both modes: in
        # strict mode it raises; otherwise EngineUnavailable is routed to fallback
        # below so a mismatched endpoint yields the offline engine, not a live one
        # that fails on the first query.
        _verify_generation_model(engine, base_url)
        log.info("RagEngine active (real route->retrieve->rerank->generate).")
        return engine
    except EngineUnavailable as exc:
        # Served-model verification failed. In strict mode re-raise; otherwise fall
        # back to offline rather than returning a live engine that would fail on the
        # first query.
        return _fallback_or_raise(str(exc), mirror, require_live)
    except Exception as exc:  # noqa: BLE001 - any infra failure must not crash the UI
        return _fallback_or_raise(
            f"RagEngine unavailable ({type(exc).__name__}: {exc}).",
            mirror,
            require_live,
        )


def _verify_generation_model(engine: Any, base_url: str) -> None:
    """Confirm the configured generation model actually resolves on the endpoint.

    Does a 1-token completion with ``cfg.generation.model``; on failure raises
    :class:`EngineUnavailable` listing the names the server *does* serve, so a
    ``--served-model-name`` mismatch is caught at startup.
    """
    model = engine._gen_model()
    try:
        client = engine._chat_client()
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
            timeout=_GEN_PROBE_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001
        served = ""
        try:
            import json
            import urllib.request

            with urllib.request.urlopen(
                base_url.rstrip("/") + "/models", timeout=5
            ) as r:
                names = [m.get("id") for m in json.load(r).get("data", [])]
            served = f" Served models: {names}."
        except Exception:  # noqa: BLE001
            pass
        raise EngineUnavailable(
            f"vLLM endpoint did not accept generation model {model!r} "
            f"({type(exc).__name__}: {exc}).{served} Serve the model under a name "
            "matching cfg.generation.model (e.g. add it to --served-model-name)."
        ) from exc


def engine_mode(engine: object) -> str:
    """Return ``"live"`` for the real engine, ``"offline"`` otherwise."""
    return "live" if getattr(engine, "is_live", False) else "offline"
