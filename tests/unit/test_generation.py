"""Tests for src/castlerag/generation/answer.py."""

from __future__ import annotations

from castlerag.generation.answer import (
    _call_generation_llm,
    _fallback_answer,
    _format_citation,
    _format_evidence_row,
    _format_timestamp,
    build_messages,
    build_prompt,
    extract_answer,
    generate_answer,
)
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


def _make_aux_hit() -> RetrievalHit:
    return RetrievalHit(
        rank=2,
        score=0.7,
        point_id="pt2",
        record_id="photo_day1_allie_00034",
        source_type="aux_photo",
        modality="image",
        day="day1",
        camera_id=None,
        participant_id="Allie",
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_200_001,
        ocr_text="Receipt on the kitchen counter.",
        asset_path="aux/day1/allie/photo_00034.jpg",
    )


class _FakeChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None

    def chat(self, messages):  # noqa: ANN001
        self.messages = messages
        return {"choices": [{"message": {"content": self.response}}]}


class _FakeOpenAIChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        return type(
            "Resp",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Msg",
                                (),
                                {"content": self.response},
                            )()
                        },
                    )()
                ]
            },
        )()


def test_extract_answer_regex():
    assert extract_answer("some text\nFINAL_ANSWER: b\n", {}) == "b"
    assert extract_answer("FINAL_ANSWER: c", {}) == "c"
    assert extract_answer("FINAL_ANSWER: D", {}) == "d"


def test_extract_answer_ignores_non_final_answer_letters():
    raw = "I think option b is plausible, but I will not follow the format."
    assert extract_answer(raw, {"a": 0.1, "b": 0.2, "c": 0.7, "d": 0.0}) == "c"


def test_extract_answer_fallback_to_priors_on_invalid_final_line():
    raw = "FINAL_ANSWER: e\nConfidence is low."
    assert extract_answer(raw, {"a": 0.6, "b": 0.1, "c": 0.2, "d": 0.1}) == "a"


def test_extract_answer_fallback_on_conflicting_final_answer_lines():
    raw = "FINAL_ANSWER: a\n...\nFINAL_ANSWER: d"
    assert extract_answer(raw, {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.9}) == "d"


def test_extract_answer_fallback_no_priors():
    assert extract_answer("no answer", {}) == "a"


def test_format_main_video_citation():
    citation = _format_citation(_make_hit())
    assert citation == "[camera=Allie time=day1 00:00:00-00:00:30]"


def test_format_aux_citation():
    citation = _format_citation(_make_aux_hit())
    assert citation == "[aux=aux_photo id=photo_day1_allie_00034]"


def test_format_evidence_row_contains_citation_and_asset():
    row = _format_evidence_row(_make_aux_hit())
    assert "citation=[aux=aux_photo id=photo_day1_allie_00034]" in row
    assert "asset: aux/day1/allie/photo_00034.jpg" in row
    assert "ocr: Receipt on the kitchen counter." in row


def test_build_prompt_contains_question_and_route_block():
    q = _make_question()
    hints = RouteHints(route="speech_text")
    hits = [_make_hit()]
    priors = {"a": 0.5, "b": 0.2, "c": 0.2, "d": 0.1}
    prompt = build_prompt(q, hints, hits, priors)
    assert "What did Allie do after breakfast?" in prompt
    assert "Prioritise transcript windows and exact spoken content." in prompt
    assert "Choices:" in prompt


def test_build_messages_puts_anti_confabulation_rules_in_system_prompt():
    messages = build_messages(
        question=_make_question(),
        hints=RouteHints(route="static_visual"),
        evidence_rows=[],
        support_priors={},
    )
    assert len(messages) == 2
    system_prompt = messages[0]["content"]
    assert "Qwen/Qwen3-VL-8B-Instruct" in system_prompt
    assert "no_echo" in system_prompt
    assert "abstain" in system_prompt
    assert "localise" in system_prompt
    assert "ground" in system_prompt
    assert "top-50 evidence budget" in system_prompt


def test_build_prompt_truncates_evidence():
    q = _make_question()
    hints = RouteHints(route="mixed")
    hits = [_make_hit() for _ in range(100)]
    prompt = build_prompt(q, hints, hits, {}, max_evidence_rows=10)
    assert "[10]" in prompt
    assert "[11]" not in prompt


def test_build_prompt_uses_placeholder_when_no_evidence():
    prompt = build_prompt(
        question=_make_question(),
        hints=RouteHints(route="mixed"),
        evidence_rows=[],
        support_priors={},
    )
    assert "[0] source=none" in prompt
    assert "No evidence rows were retrieved." in prompt


def test_generate_answer_returns_prediction_with_top_50_cap():
    client = _FakeChatClient(
        "Evidence is limited, but the cited clip supports walking.\nFINAL_ANSWER: d"
    )
    hits = [_make_hit() for _ in range(60)]
    prediction = generate_answer(
        question=_make_question(),
        hints=RouteHints(route="temporal"),
        evidence_rows=hits,
        support_priors={"a": 0.1, "b": 0.1, "c": 0.0, "d": 0.9},
        llm_client=client,
    )
    assert prediction.predicted_answer == "d"
    assert prediction.route == "temporal"
    assert len(prediction.top_evidence_ids) == 50
    assert prediction.top_evidence_ids[0] == "clip_0"
    assert prediction.raw_answer_text is not None
    assert client.messages is not None
    assert len(client.messages) == 2


def test_generate_answer_falls_back_to_support_priors():
    client = _FakeChatClient("The evidence is insufficient.\nNo valid final line.")
    prediction = generate_answer(
        question=_make_question(),
        hints=RouteHints(route="mixed"),
        evidence_rows=[_make_hit()],
        support_priors={"a": 0.2, "b": 0.1, "c": 0.6, "d": 0.1},
        llm_client=client,
    )
    assert prediction.predicted_answer == "c"
    assert prediction.confidence is not None
    assert 0.0 <= prediction.confidence <= 1.0


def test_generate_answer_supports_openai_compatible_vllm_client():
    client = _FakeOpenAIChatClient("Cited evidence supports walking.\nFINAL_ANSWER: d")
    prediction = generate_answer(
        question=_make_question(),
        hints=RouteHints(route="temporal"),
        evidence_rows=[_make_hit()],
        support_priors={"a": 0.1, "b": 0.1, "c": 0.0, "d": 0.9},
        llm_client=client,
        model="OpenGVLab/InternVL3-8B",
    )
    assert prediction.predicted_answer == "d"
    assert client.calls
    assert client.calls[0]["model"] == "OpenGVLab/InternVL3-8B"


def test_generate_answer_drops_confidence_on_low_confidence_language():
    client = _FakeChatClient(
        "Evidence is insufficient and this is low-confidence.\nFINAL_ANSWER: a"
    )
    prediction = generate_answer(
        question=_make_question(),
        hints=RouteHints(route="static_visual"),
        evidence_rows=[_make_hit()],
        support_priors={"a": 0.8, "b": 0.2, "c": 0.1, "d": 0.1},
        llm_client=client,
    )
    assert prediction.predicted_answer == "a"
    assert prediction.confidence == 0.35


# ---------------------------------------------------------------------------
# _format_timestamp — None input
# ---------------------------------------------------------------------------


def test_format_timestamp_none_returns_none():
    assert _format_timestamp(None) is None


# ---------------------------------------------------------------------------
# _format_citation — aux source_type path, camera_id without day, fallback
# ---------------------------------------------------------------------------


def test_format_citation_aux_source_types():
    for st in ("aux_heartrate", "aux_gaze", "aux_photo", "aux_thermal", "aux_video"):
        hit = RetrievalHit(
            rank=1,
            score=0.5,
            point_id="pt_aux",
            record_id=f"rec_{st}",
            source_type=st,
            modality="text",
        )
        citation = _format_citation(hit)
        assert citation.startswith(f"[aux={st} id=rec_{st}]")


def test_format_citation_camera_id_without_day():
    hit = RetrievalHit(
        rank=1,
        score=0.5,
        point_id="pt_cam",
        record_id="rec_cam",
        source_type="main_clip",
        modality="video",
        camera_id="Allie",
        day=None,
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_230_000,
    )
    citation = _format_citation(hit)
    assert citation == "[camera=Allie time=unknown]"


def test_format_citation_fallback_no_camera_no_day():
    hit = RetrievalHit(
        rank=1,
        score=0.5,
        point_id="pt_fb",
        record_id="rec_fb",
        source_type="main_clip",
        modality="video",
        camera_id=None,
        day=None,
    )
    citation = _format_citation(hit)
    assert citation == "[aux=main_clip id=rec_fb]"


# ---------------------------------------------------------------------------
# extract_answer — conflicting FINAL_ANSWER lines, free-floating letter
# ---------------------------------------------------------------------------


def test_extract_answer_multiple_conflicting_final_answer_lines_uses_priors():
    raw = "FINAL_ANSWER: a\nSome analysis.\nFINAL_ANSWER: b"
    # Two different letters → fall back to priors
    result = extract_answer(raw, {"a": 0.1, "b": 0.2, "c": 0.6, "d": 0.1})
    assert result == "c"


def test_extract_answer_free_floating_letter_falls_back_to_priors():
    raw = "Looking at the evidence, option c seems most supported."
    result = extract_answer(raw, {"a": 0.1, "b": 0.2, "c": 0.6, "d": 0.1})
    assert result == "c"


# ---------------------------------------------------------------------------
# _fallback_answer — empty support_priors returns "a"
# ---------------------------------------------------------------------------


def test_fallback_answer_empty_priors_returns_a():
    assert _fallback_answer({}) == "a"


# ---------------------------------------------------------------------------
# _call_generation_llm — various client interface variants
# ---------------------------------------------------------------------------


def test_call_generation_llm_chat_returns_string():
    class _StringChatClient:
        def chat(self, messages):
            return "FINAL_ANSWER: b"

    result = _call_generation_llm(_StringChatClient(), [])
    assert result == "FINAL_ANSWER: b"


def test_call_generation_llm_chat_returns_dict_with_content():
    class _DictContentChatClient:
        def chat(self, messages):
            return {"content": "FINAL_ANSWER: c"}

    result = _call_generation_llm(_DictContentChatClient(), [])
    assert result == "FINAL_ANSWER: c"


def test_call_generation_llm_chat_returns_dict_with_choices():
    class _DictChoicesChatClient:
        def chat(self, messages):
            return {"choices": [{"message": {"content": "FINAL_ANSWER: d"}}]}

    result = _call_generation_llm(_DictChoicesChatClient(), [])
    assert result == "FINAL_ANSWER: d"


def test_call_generation_llm_generate_returns_string():
    class _GenerateStringClient:
        def generate(self, messages):
            return "FINAL_ANSWER: a"

    result = _call_generation_llm(_GenerateStringClient(), [])
    assert result == "FINAL_ANSWER: a"


def test_call_generation_llm_generate_returns_dict_with_outputs():
    class _GenerateOutputsClient:
        def generate(self, messages):
            return {"outputs": [{"text": "FINAL_ANSWER: b"}]}

    result = _call_generation_llm(_GenerateOutputsClient(), [])
    assert result == "FINAL_ANSWER: b"


def test_call_generation_llm_generate_returns_dict_with_text():
    class _GenerateTextClient:
        def generate(self, messages):
            return {"text": "FINAL_ANSWER: c"}

    result = _call_generation_llm(_GenerateTextClient(), [])
    assert result == "FINAL_ANSWER: c"


def test_call_generation_llm_callable_client():
    result = _call_generation_llm(lambda messages: "FINAL_ANSWER: d", [])
    assert result == "FINAL_ANSWER: d"


def test_call_generation_llm_raises_on_incompatible_client():
    class _BadClient:
        pass

    import pytest

    with pytest.raises(TypeError, match="llm_client must expose"):
        _call_generation_llm(_BadClient(), [])


# ---------------------------------------------------------------------------
# _call_generation_llm_with_model — fallback to _call_generation_llm
# ---------------------------------------------------------------------------


def test_generate_answer_with_model_fallback_via_generate():
    """_call_generation_llm_with_model falls back to _call_generation_llm."""

    class _GenerateClient:
        def generate(self, messages):
            return "FINAL_ANSWER: a"

    prediction = generate_answer(
        question=_make_question(),
        hints=RouteHints(route="speech_text"),
        evidence_rows=[_make_hit()],
        support_priors={"a": 0.8, "b": 0.1, "c": 0.05, "d": 0.05},
        llm_client=_GenerateClient(),
        model="test-model",
    )
    assert prediction.predicted_answer == "a"


def test_generate_answer_with_model_fallback_via_callable():
    """_call_generation_llm_with_model falls back for callable clients."""

    def _fake_llm(messages):
        return "FINAL_ANSWER: b"

    prediction = generate_answer(
        question=_make_question(),
        hints=RouteHints(route="mixed"),
        evidence_rows=[_make_hit()],
        support_priors={"a": 0.1, "b": 0.8, "c": 0.05, "d": 0.05},
        llm_client=_fake_llm,
        model="test-model",
    )
    assert prediction.predicted_answer == "b"
