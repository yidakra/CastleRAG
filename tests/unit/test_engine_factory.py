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


class _FakeCompletions:
    def __init__(self, exc=None, recorder=None):
        self._exc = exc
        self._recorder = recorder

    def create(self, **kwargs):
        if self._recorder is not None:
            self._recorder.update(kwargs)
        if self._exc is not None:
            raise self._exc
        return object()


class _FakeChatClient:
    def __init__(self, exc=None, recorder=None):
        self.chat = type("_Chat", (), {})()
        self.chat.completions = _FakeCompletions(exc=exc, recorder=recorder)


class _FakeRagEngine:
    """Stands in for the real RagEngine; ``probe_exc`` simulates a bad served model."""

    is_live = True
    probe_exc = None
    probe_recorder = None

    @classmethod
    def from_config(cls, cfg=None, mirror=None):
        return cls()

    def _gen_model(self):
        return "configured-model"

    def _chat_client(self):
        return _FakeChatClient(exc=self.probe_exc, recorder=self.probe_recorder)


def _install_fake_engine(monkeypatch, probe_exc=None, recorder=None):
    """Make build_engine see a reachable endpoint and the fake RagEngine."""
    import sys
    import types

    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(ef, "_vllm_reachable", lambda base_url: True)

    engine_cls = type("_FakeRagEngine", (_FakeRagEngine,), {})
    engine_cls.probe_exc = probe_exc
    engine_cls.probe_recorder = recorder
    fake_mod = types.ModuleType("castlerag.ui.rag_engine")
    fake_mod.RagEngine = engine_cls
    monkeypatch.setitem(sys.modules, "castlerag.ui.rag_engine", fake_mod)
    # /models lookup inside the failure path should not hit the network.
    monkeypatch.setattr(
        ef.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no network")),
    )
    return engine_cls


def test_served_model_mismatch_without_require_live_falls_back_offline(monkeypatch):
    # /models is reachable but the served model name differs, so the generation
    # probe fails. In non-strict mode we must fall back to offline, not return live.
    _install_fake_engine(monkeypatch, probe_exc=RuntimeError("model not found"))
    engine = ef.build_engine(_Mirror(), require_live=False)
    assert isinstance(engine, PlaceholderEngine)
    assert ef.engine_mode(engine) == "offline"


def test_served_model_mismatch_with_require_live_raises(monkeypatch):
    _install_fake_engine(monkeypatch, probe_exc=RuntimeError("model not found"))
    with pytest.raises(ef.EngineUnavailable):
        ef.build_engine(_Mirror(), require_live=True)


def test_generation_probe_is_time_bounded(monkeypatch):
    # The 1-token completion probe must pass a per-request timeout so a stalled
    # endpoint cannot hang startup.
    recorder: dict = {}
    _install_fake_engine(monkeypatch, recorder=recorder)
    ef.build_engine(_Mirror(), require_live=True)
    assert recorder.get("timeout") == ef._GEN_PROBE_TIMEOUT_SECONDS


def test_engine_mode_reads_is_live_flag():
    assert ef.engine_mode(PlaceholderEngine()) == "offline"

    class _Live:
        is_live = True

    assert ef.engine_mode(_Live()) == "live"
