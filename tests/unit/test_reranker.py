"""Tests for route-aware reranking logic."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from castlerag.rerank.llm_reranker import (
    _aggregate_support_priors,
    _coerce_pack,
    _flatten_evidence_rows,
    _format_time_range,
    _invoke_reranker,
    _normalize_content,
    build_reranker_prompt,
    compute_rerank_score,
    format_candidate_pack,
    parse_reranker_response,
    rerank_candidates,
)
from castlerag.routing.question_router import RouteHints
from castlerag.schemas import (
    EvalQuestion,
    EvidencePack,
    RerankedEvidencePack,
    RerankerOutput,
    RetrievalHit,
)


def _question() -> EvalQuestion:
    return EvalQuestion(
        question_id="q1",
        query="What did Allie say after breakfast in the kitchen?",
        answers={
            "a": "She went to work",
            "b": "She cooked soup",
            "c": "She called Bjorn",
            "d": "She left the house",
        },
    )


def _hit(
    record_id: str,
    source_type: str = "main_clip",
    modality: str = "video",
    score: float = 0.9,
    start: int = 1_672_531_200_000,
) -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        score=score,
        point_id=f"pt_{record_id}",
        record_id=record_id,
        source_type=source_type,
        modality=modality,
        day="day1",
        camera_id="Allie",
        participant_id="Allie",
        absolute_start=start,
        absolute_end=start + 30_000,
        transcript_text="Allie said she would call Bjorn.",
        event_summary="Allie speaks in the kitchen.",
        ocr_text="Bjorn",
        asset_path="/tmp/clip.mp4",
    )


def _pack(pack_id: str, score: float, *, choice_hint: str = "Bjorn") -> EvidencePack:
    hit = _hit(pack_id, score=score)
    aux_hit = _hit(
        f"{pack_id}_tx",
        source_type="transcript_window",
        modality="text",
        score=score - 0.1,
        start=hit.absolute_start or 0,
    )
    return EvidencePack(
        pack_id=pack_id,
        route="speech_text",
        primary_hit=hit,
        retrieval_score=score,
        evidence_rows=[hit, aux_hit],
        transcript_evidence=[f"Transcript mentions {choice_hint} after breakfast."],
        event_summaries=["Kitchen conversation after breakfast."],
        ocr_spans=[choice_hint],
        frame_descriptions=["Allie standing at the kitchen counter."],
        auxiliary_notes=["Heartrate stable during the exchange."],
    )


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def test_format_candidate_pack_includes_route_and_modalities():
    pack = _pack("pack1", 0.9)
    text = format_candidate_pack(pack, rank=1)
    assert "Candidate pack 1" in text
    assert "Route: speech_text" in text
    assert "Primary source: main_clip" in text
    assert "Transcript evidence" in text
    assert "Sampled-frame descriptions" in text
    assert "Auxiliary notes" in text


def test_build_reranker_prompt_contains_json_contract():
    pack = _pack("pack1", 0.9)
    prompt = build_reranker_prompt(_question(), pack, rank=2)
    assert "What did Allie say after breakfast in the kitchen?" in prompt
    assert '"support": {"a": 0-4, "b": 0-4, "c": 0-4, "d": 0-4}' in prompt
    assert "Candidate pack 2" in prompt


def test_parse_reranker_response_handles_extra_text():
    raw = (
        "Assessment follows.\n"
        '{"relevance": 4, "support": {"a": 0, "b": 1, "c": 4, "d": 0}, '
        '"keep": true, "rationale": "Direct transcript support"}\nDone.'
    )
    out = parse_reranker_response(raw)
    assert out.relevance == 4
    assert out.support["c"] == 4
    assert out.final_rerank_score == pytest.approx(4.0)


def test_compute_rerank_score_respects_weights():
    raw = parse_reranker_response(
        '{"relevance": 2, "support": {"a": 0, "b": 4, "c": 1, "d": 1}, '
        '"keep": true, "rationale": "Moderate"}',
        relevance_weight=0.6,
        support_weight=0.4,
    )
    score = compute_rerank_score(raw, relevance_weight=0.6, support_weight=0.4)
    assert score == pytest.approx(2.8)


def test_rerank_candidates_orders_top_packs_and_aggregates_priors():
    packs = [_pack("pack_low", 0.5), _pack("pack_high", 0.9), _pack("pack_mid", 0.7)]
    client = _FakeLLMClient(
        [
            (
                '{"relevance": 2, "support": {"a": 0, "b": 1, "c": 1, "d": 0}, '
                '"keep": true, "rationale": "Weak"}'
            ),
            (
                '{"relevance": 4, "support": {"a": 0, "b": 0, "c": 4, "d": 1}, '
                '"keep": true, "rationale": "Strong"}'
            ),
            (
                '{"relevance": 3, "support": {"a": 0, "b": 0, "c": 2, "d": 0}, '
                '"keep": true, "rationale": "Solid"}'
            ),
        ]
    )
    result = rerank_candidates(
        question=_question(),
        hints=RouteHints(route="speech_text"),
        candidate_packs=packs,
        llm_client=client,
        top_k=2,
        max_evidence_rows=3,
    )
    assert [item.pack.pack_id for item in result.kept_packs] == [
        "pack_high",
        "pack_mid",
    ]
    assert result.support_priors == {"a": 0.0, "b": 0.0, "c": 6.0, "d": 1.0}
    assert len(result.evidence_rows) == 3
    assert client.chat.completions.calls[0]["model"] == "Qwen/Qwen3-VL-8B-Instruct"


def test_rerank_candidates_skips_parse_failures_and_low_relevance(caplog):
    pack_ok = _pack("pack_ok", 0.8)
    pack_bad = _pack("pack_bad", 0.7)
    pack_low = _pack("pack_low", 0.6)
    client = _FakeLLMClient(
        [
            "not json at all",
            (
                '{"relevance": 1, "support": {"a": 1, "b": 0, "c": 0, "d": 0}, '
                '"keep": true, "rationale": "Too weak"}'
            ),
            (
                '{"relevance": 3, "support": {"a": 0, "b": 0, "c": 3, "d": 0}, '
                '"keep": true, "rationale": "Usable"}'
            ),
        ]
    )
    with caplog.at_level(logging.WARNING):
        result = rerank_candidates(
            question=_question(),
            hints=RouteHints(route="speech_text"),
            candidate_packs=[pack_bad, pack_low, pack_ok],
            llm_client=client,
            min_relevance=1,
        )
    assert [item.pack.pack_id for item in result.kept_packs] == ["pack_ok"]
    assert "parse failure" in caplog.text


# ---------------------------------------------------------------------------
# format_candidate_pack — ValueError when neither pack nor route+hit provided
# ---------------------------------------------------------------------------


def test_format_candidate_pack_raises_without_pack_or_route_hit():
    with pytest.raises(ValueError, match="Either pack or both route and hit"):
        format_candidate_pack(pack=None, route=None, hit=None)


def test_format_candidate_pack_raises_with_route_only():
    with pytest.raises(ValueError, match="Either pack or both route and hit"):
        format_candidate_pack(pack=None, route="speech_text", hit=None)


def test_format_candidate_pack_raises_with_hit_only():
    hit = _hit("rec1")
    with pytest.raises(ValueError, match="Either pack or both route and hit"):
        format_candidate_pack(pack=None, route=None, hit=hit)


# ---------------------------------------------------------------------------
# _format_time_range — None absolute_start/end
# ---------------------------------------------------------------------------


def test_format_time_range_returns_na_when_no_timestamps():
    hit = RetrievalHit(
        rank=1,
        score=0.5,
        point_id="pt_none",
        record_id="rec_none",
        source_type="main_clip",
        modality="video",
        absolute_start=None,
        absolute_end=None,
    )
    assert _format_time_range(hit) == "N/A"


def test_format_time_range_returns_na_when_only_start():
    hit = RetrievalHit(
        rank=1,
        score=0.5,
        point_id="pt_none",
        record_id="rec_none",
        source_type="main_clip",
        modality="video",
        absolute_start=1_000,
        absolute_end=None,
    )
    assert _format_time_range(hit) == "N/A"


# ---------------------------------------------------------------------------
# parse_reranker_response — no valid JSON found
# ---------------------------------------------------------------------------


def test_parse_reranker_response_raises_on_no_valid_json():
    with pytest.raises(ValueError, match="No valid reranker JSON found"):
        parse_reranker_response("this has no json at all")


def test_parse_reranker_response_raises_on_invalid_schema():
    # JSON but missing required fields
    with pytest.raises(ValueError, match="No valid reranker JSON found"):
        parse_reranker_response('{"foo": "bar"}')


# ---------------------------------------------------------------------------
# _coerce_pack — dict input with missing route / retrieval_score
# ---------------------------------------------------------------------------


def test_coerce_pack_dict_with_missing_route_uses_hints_route():
    hit = _hit("rec_x")
    data = {
        "pack_id": "pack_x",
        "primary_hit": hit,
        "evidence_rows": [hit],
    }
    hints = RouteHints(route="temporal")
    pack = _coerce_pack(data, hints)
    assert pack.route == "temporal"


def test_coerce_pack_dict_with_missing_retrieval_score_uses_primary_hit_score():
    hit = _hit("rec_y", score=0.75)
    data = {
        "pack_id": "pack_y",
        "route": "mixed",
        "primary_hit": hit,
        "evidence_rows": [hit],
    }
    hints = RouteHints(route="mixed")
    pack = _coerce_pack(data, hints)
    assert pack.retrieval_score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# _invoke_reranker — fallback client.create(...) path
# ---------------------------------------------------------------------------


def test_invoke_reranker_uses_create_fallback():
    class _CreateClient:
        def create(self, **kwargs):
            return (
                '{"relevance": 3, "support": {"a": 1, "b": 2, "c": 3, "d": 0},'
                ' "keep": true, "rationale": "ok"}'
            )

    result = _invoke_reranker(
        llm_client=_CreateClient(),
        model="test-model",
        prompt="Test prompt",
    )
    assert "relevance" in result


def test_invoke_reranker_raises_on_incompatible_client():
    class _BadClient:
        pass

    with pytest.raises(TypeError, match="llm_client must expose"):
        _invoke_reranker(
            llm_client=_BadClient(),
            model="test-model",
            prompt="Test prompt",
        )


# ---------------------------------------------------------------------------
# _normalize_content — None, list-of-strings, list-of-dicts
# ---------------------------------------------------------------------------


def test_normalize_content_none_returns_empty_string():
    assert _normalize_content(None) == ""


def test_normalize_content_list_of_strings():
    result = _normalize_content(["Hello", "World"])
    assert "Hello" in result
    assert "World" in result


def test_normalize_content_list_of_dicts_with_text_key():
    result = _normalize_content([{"text": "Part one"}, {"text": "Part two"}])
    assert "Part one" in result
    assert "Part two" in result


def test_normalize_content_list_of_dicts_without_text_key_skipped():
    result = _normalize_content([{"other": "no text"}, {"text": "kept"}])
    assert "kept" in result
    assert "no text" not in result


def test_normalize_content_non_string_coerced():
    assert _normalize_content(42) == "42"


# ---------------------------------------------------------------------------
# _aggregate_support_priors — non-float support values
# ---------------------------------------------------------------------------


def test_aggregate_support_priors_with_non_float_values():
    output = RerankerOutput(
        relevance=3,
        support={"a": 1, "b": 2, "c": 3, "d": 0},
        keep=True,
        rationale="ok",
    )
    pack = _pack("p1", 0.9)
    reranked = RerankedEvidencePack(
        pack=pack,
        reranker_output=output,
        final_rerank_score=3.0,
    )
    priors = _aggregate_support_priors([reranked])
    # Values should be floats
    assert isinstance(priors["a"], float)
    assert priors["c"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _flatten_evidence_rows — deduplication and max_rows cap
# ---------------------------------------------------------------------------


def test_flatten_evidence_rows_deduplicates_same_record_id():
    shared_hit = _hit("shared_rec")
    output = RerankerOutput(
        relevance=3,
        support={"a": 1, "b": 1, "c": 1, "d": 1},
        keep=True,
        rationale="ok",
    )
    pack1 = EvidencePack(
        pack_id="p1",
        route="speech_text",
        primary_hit=shared_hit,
        retrieval_score=0.9,
        evidence_rows=[shared_hit],
    )
    pack2 = EvidencePack(
        pack_id="p2",
        route="speech_text",
        primary_hit=shared_hit,
        retrieval_score=0.8,
        evidence_rows=[shared_hit],
    )
    reranked1 = RerankedEvidencePack(
        pack=pack1, reranker_output=output, final_rerank_score=3.0
    )
    reranked2 = RerankedEvidencePack(
        pack=pack2, reranker_output=output, final_rerank_score=2.5
    )
    rows = _flatten_evidence_rows([reranked1, reranked2], max_rows=10)
    # shared_rec should appear only once
    assert len([r for r in rows if r.record_id == "shared_rec"]) == 1


def test_flatten_evidence_rows_respects_max_rows_cap():
    output = RerankerOutput(
        relevance=3,
        support={"a": 1, "b": 1, "c": 1, "d": 1},
        keep=True,
        rationale="ok",
    )
    hits = [_hit(f"rec_{i}") for i in range(10)]
    pack = EvidencePack(
        pack_id="big_pack",
        route="speech_text",
        primary_hit=hits[0],
        retrieval_score=0.9,
        evidence_rows=hits,
    )
    reranked = RerankedEvidencePack(
        pack=pack, reranker_output=output, final_rerank_score=3.0
    )
    rows = _flatten_evidence_rows([reranked], max_rows=3)
    assert len(rows) == 3
