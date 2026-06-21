"""Tests for live/offline engine selection and the --require-live contract."""

from __future__ import annotations

import pytest

from castlerag.ui import engine_factory as ef
from castlerag.ui.chat import PlaceholderEngine


class _Mirror:
    mapping: dict = {}


def test_default_falls_back_to_offline_when_no_vllm(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    engine = ef.build_engine(_Mirror())
    assert isinstance(engine, PlaceholderEngine)
    assert ef.engine_mode(engine) == "offline"


def test_require_live_raises_when_vllm_unset(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    with pytest.raises(ef.EngineUnavailable) as exc:
        ef.build_engine(_Mirror(), require_live=True)
    assert "VLLM_BASE_URL" in str(exc.value)


def test_require_live_raises_when_unreachable(monkeypatch):
    # A port nothing listens on -> probe fails -> strict mode raises.
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:1/v1")
    with pytest.raises(ef.EngineUnavailable) as exc:
        ef.build_engine(_Mirror(), require_live=True)
    assert "not reachable" in str(exc.value)


def test_unreachable_without_require_live_still_offline(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:1/v1")
    engine = ef.build_engine(_Mirror(), require_live=False)
    assert isinstance(engine, PlaceholderEngine)


def test_probe_treats_non_2xx_as_unreachable(monkeypatch):
    # A server that responds but with a non-2xx status (e.g. 404) is listening
    # yet not actually usable; the probe must report it as unreachable.
    import urllib.error

    def _raise_http_error(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(ef.urllib.request, "urlopen", _raise_http_error)
    assert ef._vllm_reachable("http://127.0.0.1:9/v1") is False


def test_probe_treats_2xx_as_reachable(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(ef.urllib.request, "urlopen", lambda url, timeout: _Resp())
    assert ef._vllm_reachable("http://127.0.0.1:9/v1") is True


def test_engine_mode_reads_is_live_flag():
    assert ef.engine_mode(PlaceholderEngine()) == "offline"

    class _Live:
        is_live = True

    assert ef.engine_mode(_Live()) == "live"
