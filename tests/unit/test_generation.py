"""Tests for src/castlerag/generation/answer.py (pure logic, no LLM calls)."""
import pytest

from castlerag.generation.answer import (
    _format_evidence_row,
    build_prompt,
    extract_answer,
)
from castlerag.rerank.llm_reranker import format_candidate_pack
from castlerag.routing.question_router import RouteHints
from castlerag.schemas import EvalQuestion, RetrievalHit


def _make_question() -> EvalQuestion:
    return EvalQuestion(
        question_id="q1",
        query="What did Allie do after breakfast?",
        answers={"a": "Worked", "b": "Slept", "c": "Cooked", "d": "Walked"},
    )


def _make_hit() -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        score=0.9,
        point_id="pt1",
        record_id="clip_0",
        source_type="main_clip",
        modality="video",
        day="day1",
        camera_id="Allie",
        participant_id="Allie",
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_230_000,
        transcript_text="Allie walked to the office.",
    )


def test_extract_answer_regex():
    assert extract_answer("some text\nFINAL_ANSWER: b\n", {}) == "b"
    assert extract_answer("FINAL_ANSWER: c", {}) == "c"
    assert extract_answer("FINAL_ANSWER: D", {}) == "d"  # case-insensitive


def test_extract_answer_fallback_to_priors():
    # No FINAL_ANSWER line → use highest prior
    result = extract_answer("no answer here", {"a": 0.1, "b": 0.8, "c": 0.3, "d": 0.2})
    assert result == "b"


def test_extract_answer_fallback_no_priors():
    result = extract_answer("no answer", {})
    assert result == "a"  # default when no priors


def test_format_evidence_row():
    hit = _make_hit()
    text = _format_evidence_row(hit)
    assert "Allie" in text
    assert "day1" in text
    assert "Allie walked" in text


def test_build_prompt_contains_question():
    q = _make_question()
    hints = RouteHints(route="speech_text")
    hits = [_make_hit()]
    priors = {"a": 0.5, "b": 0.2, "c": 0.2, "d": 0.1}
    prompt = build_prompt(q, hints, hits, priors)
    assert "What did Allie do after breakfast?" in prompt
    assert "FINAL_ANSWER" in prompt


def test_build_prompt_anti_confabulation_rules():
    q = _make_question()
    hints = RouteHints(route="static_visual")
    prompt = build_prompt(q, hints, [], {})
    assert "no_echo" in prompt
    assert "abstain" in prompt
    assert "localise" in prompt
    assert "ground" in prompt


def test_build_prompt_truncates_evidence():
    q = _make_question()
    hints = RouteHints(route="mixed")
    hits = [_make_hit() for _ in range(100)]
    priors: dict = {}
    prompt = build_prompt(q, hints, hits, priors, max_evidence_rows=10)
    # Only 10 evidence rows should appear ([1] through [10])
    assert "[10]" in prompt
    assert "[11]" not in prompt


def test_parse_reranker_response_with_preamble():
    """parse_reranker_response must skip extra text before the JSON object."""
    from castlerag.rerank.llm_reranker import parse_reranker_response
    raw = (
        'Sure, here is my assessment:\n'
        '{"relevance": 3, "support": {"a": 2, "b": 1, "c": 0, "d": 3}, '
        '"keep": true, "rationale": "Strong match"}\nDone.'
    )
    out = parse_reranker_response(raw)
    assert out.relevance == 3
    assert out.keep is True


def test_parse_reranker_response_with_extra_braces():
    """Should not over-capture when the model emits extra {} in preamble."""
    from castlerag.rerank.llm_reranker import parse_reranker_response
    raw = (
        'Context: {} not relevant.\n'
        '{"relevance": 2, "support": {"a": 1, "b": 2, "c": 0, "d": 0}, '
        '"keep": true, "rationale": "Moderate"}'
    )
    out = parse_reranker_response(raw)
    assert out.relevance == 2


def test_parse_reranker_response_no_json_raises():
    from castlerag.rerank.llm_reranker import parse_reranker_response
    with pytest.raises(ValueError, match="No valid"):
        parse_reranker_response("No JSON here at all.")


def test_format_candidate_pack_structure():
    hit = _make_hit()
    pack = format_candidate_pack(
        rank=1,
        route="speech_text",
        hit=hit,
        transcript_chunks="Allie walked to the office.",
    )
    assert "Candidate pack 1" in pack
    assert "speech_text" in pack
    assert "Allie walked" in pack
