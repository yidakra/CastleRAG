"""Tests for LLM review suggestions (justification + refined query).

Covers the production code path with a fake OpenAI-compatible client (no live
endpoint), the deterministic offline drafts on PlaceholderEngine, and the
RagEngine template fallback when the client raises.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from castlerag.generation.suggestions import (
    suggest_justification_text,
    suggest_refined_query_text,
)
from castlerag.ui.chat import PlaceholderEngine, compose_justification

# --- fake OpenAI-compatible client ----------------------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.message = type("M", (), {"content": content})()


class _FakeCompletions:
    def __init__(self, content: str, sink: List[Dict[str, Any]]) -> None:
        self._content = content
        self._sink = sink

    def create(self, **kwargs: Any) -> Any:
        self._sink.append(kwargs)
        return type("R", (), {"choices": [_FakeMessage(self._content)]})()


class FakeClient:
    """Mimics the OpenAI client surface used by the suggestions module."""

    def __init__(self, content: str = "  drafted text  ") -> None:
        self.calls: List[Dict[str, Any]] = []
        self.chat = type(
            "C", (), {"completions": _FakeCompletions(content, self.calls)}
        )()


class BoomClient:
    """Client whose completion call always raises (to test fallbacks)."""

    def __init__(self) -> None:
        def _raise(**_kwargs: Any) -> Any:
            raise RuntimeError("vLLM down")

        self.chat = type(
            "C", (), {"completions": type("X", (), {"create": staticmethod(_raise)})()}
        )()


# --- production code path (fake client) ------------------------------------


def test_suggest_justification_text_calls_client_and_strips():
    client = FakeClient("  Bjorn's doorway view clearly shows the handoff.  ")
    out = suggest_justification_text(
        claim="X did Y",
        camera_id="Bjorn",
        verdict="confirmed",
        evidence_text="doorway handoff at 12:29",
        meta={"clock_label": "12:29", "place_label": "Doorway", "match_score": 0.73},
        llm_client=client,
        model="test-model",
    )
    assert out == "Bjorn's doorway view clearly shows the handoff."
    # One chat completion with our model and a system+user message pair.
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "test-model"
    roles = [m["role"] for m in call["messages"]]
    assert roles == ["system", "user"]
    # The grounding evidence and verdict are present in the prompt.
    user = call["messages"][1]["content"]
    assert "doorway handoff" in user
    assert "Bjorn" in user


def test_suggest_refined_query_text_includes_weak_cameras():
    client = FakeClient("Re-examine the doorway handoff with a clearer Luca angle.")
    reviews = {
        "Bjorn": {"state": "confirmed", "justification": "clear"},
        "Luca": {"state": "rejected", "justification": "occluded"},
        "Klaus": {"state": "flagged", "justification": "blurry"},
    }
    out = suggest_refined_query_text("X did Y", reviews, llm_client=client, model="m")
    assert out.startswith("Re-examine")
    user = client.calls[0]["messages"][1]["content"]
    assert "Luca" in user and "Klaus" in user  # weak angles surfaced


def test_suggest_refined_query_text_separates_rejected_from_flagged():
    # The LLM-path query must steer toward flagged angles but AWAY from rejected
    # ones (not lump both into "needs a clearer view").
    client = FakeClient("refined")
    reviews = {
        "Luca": {"state": "rejected", "justification": "occluded"},
        "Klaus": {"state": "flagged", "justification": "blurry"},
    }
    suggest_refined_query_text("X did Y", reviews, llm_client=client, model="m")
    user = client.calls[0]["messages"][1]["content"]
    clearer_line = next(ln for ln in user.splitlines() if "clearer view" in ln)
    rejected_line = next(ln for ln in user.splitlines() if "rejected" in ln.lower())
    assert "Klaus" in clearer_line and "Luca" not in clearer_line
    assert "Luca" in rejected_line and "Klaus" not in rejected_line


def test_suggestions_reject_non_openai_client():
    with pytest.raises(TypeError):
        suggest_justification_text(
            "c", "Cam", "confirmed", None, None, llm_client=object()
        )


# --- offline / deterministic drafts ---------------------------------------


@pytest.mark.parametrize("verdict", ["confirmed", "flagged", "rejected"])
def test_compose_justification_is_nonempty_per_verdict(verdict: str):
    text = compose_justification("the claim", "Bjorn", verdict, evidence_text="x" * 200)
    assert text and text.endswith(".")
    assert "Bjorn" in text


def test_placeholder_engine_suggestions_are_deterministic():
    eng = PlaceholderEngine()
    a = eng.suggest_justification("c", "Bjorn", "flagged", "snippet", None)
    b = eng.suggest_justification("c", "Bjorn", "flagged", "snippet", None)
    assert a == b and a  # deterministic, non-empty
    reviews = {"Bjorn": {"state": "rejected", "justification": "occluded"}}
    q = eng.suggest_refined_query("the claim", reviews)
    assert "Bjorn" in q


# --- RagEngine fallback (no live infra) ------------------------------------


def test_rag_engine_falls_back_to_template_on_client_error():
    from castlerag.ui.rag_engine import RagEngine

    eng = RagEngine(cfg=None, pipeline=None)
    eng._client = BoomClient()  # force the LLM path to raise
    out = eng.suggest_justification("the claim", "Bjorn", "confirmed", "ev", {})
    # Falls back to the deterministic template (never raises, never empty).
    assert out == compose_justification("the claim", "Bjorn", "confirmed", "ev")

    q = eng.suggest_refined_query("the claim", {"Bjorn": {"state": "flagged"}})
    assert q and "Bjorn" in q
