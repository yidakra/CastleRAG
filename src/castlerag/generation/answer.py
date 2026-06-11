"""Answer generation prompt, citation formatting, and answer extraction.

Model: Qwen/Qwen3-VL-8B-Instruct via vLLM.
Ablation: OpenGVLab/InternVL3-8B.

Anti-confabulation rules (SPEC §6.1.1 — mandatory, not optional style):
  no_echo    — do not repeat prompt text as evidence
  abstain    — say evidence is insufficient instead of inventing; still choose
               the least-unsupported option with a low-confidence note
  localise   — every count/location claim needs camera+timestamp citation
  ground     — confidence from retrieved evidence, not world knowledge
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Dict, List, Sequence

from castlerag.routing.question_router import RouteHints
from castlerag.schemas import AnswerChoice, EvalQuestion, Prediction, RetrievalHit

_FINAL_ANSWER_RE = re.compile(
    r"(?mi)^\s*FINAL_ANSWER:\s*([abcd])\s*$",
)
_ANSWER_LETTER_RE = re.compile(r"(?i)\b([abcd])\b")
_AUX_CITATION_PREFIXES = frozenset(
    {
        "aux_heartrate",
        "aux_gaze",
        "aux_photo",
        "aux_thermal",
        "aux_video",
    }
)

_ROUTE_PROMPT_BLOCKS: Dict[str, str] = {
    "static_visual": (
        "Prioritise frames, OCR text, object counts, colours, brands, and room layout."
    ),
    "speech_text": (
        "Prioritise transcript windows and exact spoken content. "
        "Use video evidence only to disambiguate speakers, locations, or "
        "visible objects."
    ),
    "temporal": (
        "Reconstruct order using timestamps and neighbouring evidence. "
        "Sample frames from candidate videos to verify before/after/while relations."
    ),
    "mixed": (
        "Require agreement between transcript and visual evidence before "
        "preferring an option."
    ),
}

_SYSTEM_PROMPT = """\
You are CastleRAG's final answer generator for CASTLE multiple-choice questions.
Target model contract: Qwen/Qwen3-VL-8B-Instruct served through vLLM.

Rules:
- Use only the provided evidence.
- Prefer direct evidence over speculation.
- If evidence is weak, say so briefly but still choose the most supported option.
- Every factual claim used in the decision must cite at least one evidence item.
- Citations must use the format [camera={{camera_id}} time={{day}} {{start}}-{{end}}] \
or [aux={{source_type}} id={{record_id}}].
- Follow the route-specific instruction block exactly.
- Respect the top-50 evidence budget. Ignore any evidence not included in the prompt.
- Anti-confabulation rules are mandatory:
  - no_echo: do not quote the question, answer options, route hints,
    or prompt instructions as evidence.
  - abstain: when no clip supports a claim, explicitly say the evidence
    is insufficient instead of guessing; still select the
    least-unsupported answer and mark the rationale low-confidence.
  - localise: every count, object-location, or spatial claim must cite
    a specific camera and timestamp.
  - ground: confidence must come from cited evidence,
    not from option plausibility or outside knowledge.
- End with exactly one line: FINAL_ANSWER: a|b|c|d
"""

_USER_PROMPT_TEMPLATE = """\
Answer the CASTLE multiple-choice question using only the supplied evidence.

Question route:
{route}

Route-specific instructions:
{route_block}

Question:
{question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Choice support priors:
{support_summary}

Evidence:
{evidence}
"""

_MISSING_EVIDENCE_ROW = (
    "[0] source=none citation=[aux=none id=no_evidence]\n"
    "summary: No evidence rows were retrieved."
)


def build_prompt(
    question: EvalQuestion,
    hints: RouteHints,
    evidence_rows: List[RetrievalHit],
    support_priors: Dict[str, float],
    max_evidence_rows: int = 50,
) -> str:
    """Return the formatted user prompt string for the generation LLM."""
    rows = evidence_rows[:max_evidence_rows]
    evidence_text = "\n\n".join(_enumerate_evidence_rows(rows))
    support_summary = "  ".join(
        f"{k.upper()}: {v:.2f}" for k, v in sorted(support_priors.items())
    )
    return _USER_PROMPT_TEMPLATE.format(
        route=hints.route,
        route_block=_ROUTE_PROMPT_BLOCKS.get(hints.route, ""),
        question=question.query,
        choice_a=question.answers["a"],
        choice_b=question.answers["b"],
        choice_c=question.answers["c"],
        choice_d=question.answers["d"],
        support_summary=support_summary or "N/A",
        evidence=evidence_text or _MISSING_EVIDENCE_ROW,
    )


def build_messages(
    question: EvalQuestion,
    hints: RouteHints,
    evidence_rows: List[RetrievalHit],
    support_priors: Dict[str, float],
    max_evidence_rows: int = 50,
) -> List[Dict[str, str]]:
    """Return the system+user message list for the generation LLM."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_prompt(
                question=question,
                hints=hints,
                evidence_rows=evidence_rows,
                support_priors=support_priors,
                max_evidence_rows=max_evidence_rows,
            ),
        },
    ]


def _enumerate_evidence_rows(evidence_rows: Sequence[RetrievalHit]) -> List[str]:
    """Return evidence rows as numbered strings prefixed with their 1-based index."""
    return [
        f"[{i + 1}] {_format_evidence_row(hit)}"
        for i, hit in enumerate(evidence_rows)
    ]


def _format_timestamp(ms: int | None) -> str | None:
    """Return a UTC HH:MM:SS string for the given millisecond epoch value, or None."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%H:%M:%S")


def _format_citation(hit: RetrievalHit) -> str:
    """Return the bracketed citation string for a retrieval hit."""
    if hit.source_type in _AUX_CITATION_PREFIXES:
        return f"[aux={hit.source_type} id={hit.record_id}]"
    if (
        hit.camera_id
        and hit.day
        and hit.absolute_start is not None
        and hit.absolute_end is not None
    ):
        start = _format_timestamp(hit.absolute_start)
        end = _format_timestamp(hit.absolute_end)
        if start and end:
            return f"[camera={hit.camera_id} time={hit.day} {start}-{end}]"
    if hit.camera_id:
        return f"[camera={hit.camera_id} time=unknown]"
    return f"[aux={hit.source_type} id={hit.record_id}]"


def _format_evidence_row(hit: RetrievalHit) -> str:
    """Return a header-plus-body string representing a single evidence hit."""
    parts = [f"source={hit.source_type}"]
    if hit.camera_id:
        parts.append(f"camera={hit.camera_id}")
    if hit.day:
        parts.append(f"day={hit.day}")
    parts.append(f"citation={_format_citation(hit)}")
    header = " ".join(parts)
    body_parts = []
    if hit.transcript_text:
        body_parts.append(f"transcript: {hit.transcript_text}")
    if hit.event_summary:
        body_parts.append(f"event: {hit.event_summary}")
    if hit.ocr_text:
        body_parts.append(f"ocr: {hit.ocr_text}")
    if hit.asset_path:
        body_parts.append(f"asset: {hit.asset_path}")
    body = "\n".join(body_parts) if body_parts else "[no text]"
    return f"{header}\n{body}"


def extract_answer(raw_text: str, support_priors: Dict[str, float]) -> AnswerChoice:
    """Parse a strict FINAL_ANSWER line; fall back to highest support prior."""
    matches = [match.group(1).lower() for match in _FINAL_ANSWER_RE.finditer(raw_text)]
    if len(matches) == 1:
        return matches[0]  # type: ignore[return-value]
    if len(matches) > 1:
        unique = set(matches)
        if len(unique) == 1:
            return matches[0]  # type: ignore[return-value]
    # Reject free-floating choice letters; generation must use FINAL_ANSWER.
    if _ANSWER_LETTER_RE.search(raw_text):
        return _fallback_answer(support_priors)
    if support_priors:
        return _fallback_answer(support_priors)
    return "a"


def _fallback_answer(support_priors: Dict[str, float]) -> AnswerChoice:
    """Return the answer choice with the highest support prior, defaulting to 'a'."""
    ordered = sorted(
        support_priors.items(),
        key=lambda item: (-item[1], item[0]),
    )
    if ordered:
        return ordered[0][0]  # type: ignore[return-value]
    return "a"


def _estimate_confidence(
    answer: AnswerChoice,
    support_priors: Dict[str, float],
    evidence_rows: Sequence[RetrievalHit],
    raw_text: str,
) -> float:
    """Return a confidence score in [0, 1] from evidence count and support priors."""
    if not evidence_rows:
        return 0.0
    best_prior = max(support_priors.values()) if support_priors else 0.0
    selected_prior = support_priors.get(answer, 0.0)
    prior_ratio = 0.0 if best_prior <= 0 else min(selected_prior / best_prior, 1.0)
    confidence = 0.25 + 0.5 * prior_ratio + 0.25 * min(len(evidence_rows), 50) / 50
    if "low-confidence" in raw_text.lower() or "insufficient" in raw_text.lower():
        confidence = min(confidence, 0.35)
    return round(max(0.0, min(confidence, 1.0)), 4)


def _call_generation_llm(llm_client: Any, messages: List[Dict[str, str]]) -> str:
    """Dispatch a chat-completion request via the LLM client's available interface."""
    if hasattr(llm_client, "generate_from_messages"):
        return str(llm_client.generate_from_messages(messages))
    if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
        response = llm_client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct",
            messages=messages,
            max_tokens=512,
            temperature=0.0,
        )
        if not response.choices:
            return ""
        return str(response.choices[0].message.content or "")
    if hasattr(llm_client, "chat"):
        response = llm_client.chat(messages)
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if isinstance(response.get("content"), str):
                return response["content"]
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {})
                if isinstance(message, dict) and isinstance(
                    message.get("content"), str
                ):
                    return message["content"]
        return str(response)
    if hasattr(llm_client, "generate"):
        response = llm_client.generate(messages=messages)
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            outputs = response.get("outputs")
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if isinstance(first, dict) and isinstance(first.get("text"), str):
                    return first["text"]
            if isinstance(response.get("text"), str):
                return response["text"]
        return str(response)
    if callable(llm_client):
        return str(llm_client(messages))
    raise TypeError(
        "llm_client must expose generate_from_messages(), chat(), "
        "generate(), or be callable"
    )


def generate_answer(
    question: EvalQuestion,
    hints: RouteHints,
    evidence_rows: List[RetrievalHit],
    support_priors: Dict[str, float],
    llm_client: Any,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    max_evidence_rows: int = 50,
) -> Prediction:
    """Run grounded answer generation and return a normalized Prediction."""
    rows = evidence_rows[:max_evidence_rows]
    messages = build_messages(
        question=question,
        hints=hints,
        evidence_rows=rows,
        support_priors=support_priors,
        max_evidence_rows=max_evidence_rows,
    )
    raw_answer_text = _call_generation_llm_with_model(llm_client, messages, model=model)
    predicted_answer = extract_answer(raw_answer_text, support_priors)
    return Prediction(
        question_id=question.question_id,
        predicted_answer=predicted_answer,
        route=hints.route,
        support_priors=support_priors or None,
        top_evidence_ids=[hit.record_id for hit in rows],
        raw_answer_text=raw_answer_text,
        confidence=_estimate_confidence(
            answer=predicted_answer,
            support_priors=support_priors,
            evidence_rows=rows,
            raw_text=raw_answer_text,
        ),
    )


def _call_generation_llm_with_model(
    llm_client: Any,
    messages: List[Dict[str, str]],
    *,
    model: str,
) -> str:
    """Dispatch a chat-completion for a specific model, falling back as needed."""
    if hasattr(llm_client, "generate_from_messages"):
        return str(llm_client.generate_from_messages(messages))
    if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.0,
        )
        if not response.choices:
            return ""
        return str(response.choices[0].message.content or "")
    return _call_generation_llm(llm_client, messages)
