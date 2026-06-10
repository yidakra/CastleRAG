"""Answer generation prompt, citation formatting, and answer extraction.

Model: Qwen/Qwen3-VL-8B-Instruct via vLLM.
Ablation: OpenGVLab/InternVL3-8B.

Anti-confabulation rules (SPEC §6.1.1 — mandatory, not optional style):
  no_echo    — do not repeat prompt text as evidence
  abstain    — say evidence is insufficient instead of inventing; still choose
               the least-unsupported option with a low-confidence note
  localise   — every count/location claim needs camera+timestamp citation
  ground     — confidence from retrieved evidence, not world knowledge

Citation format:
  main video  → [camera=Allie time=day2 14:03:00-14:03:30]
  auxiliary   → [aux=photo id=photo_day2_allie_00034]

Final answer extraction regex: FINAL_ANSWER:\\s*([abcd])
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from castlerag.routing.question_router import RouteHints
from castlerag.schemas import AnswerChoice, EvalQuestion, Prediction, RetrievalHit

_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER:\s*([abcd])", re.IGNORECASE)

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

_GENERATION_PROMPT_TEMPLATE = """\
You answer multiple-choice questions about the CASTLE dataset.

Rules:
- Use only the provided evidence.
- Prefer direct evidence over speculation.
- If evidence is weak, say so briefly but still choose the most supported option.
- Every factual claim used in the decision must cite at least one evidence item.
- Citations must use the format [camera={{camera_id}} time={{day}} {{start}}-{{end}}] \
or [aux={{source_type}} id={{record_id}}].
- Follow the route-specific instruction block exactly.
- Anti-confabulation rules (mandatory):
    no_echo: do not repeat prompt text, answer options, or route hints as evidence.
    abstain: if no evidence supports a claim, say so; still choose the least \
unsupported option and mark low-confidence.
    localise: every count, object-location, or spatial claim must cite camera
    + timestamp.
    ground: confidence must come from retrieved evidence, not world knowledge.
- End with exactly one line: FINAL_ANSWER: a|b|c|d

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


def build_prompt(
    question: EvalQuestion,
    hints: RouteHints,
    evidence_rows: List[RetrievalHit],
    support_priors: Dict[str, float],
    max_evidence_rows: int = 50,
) -> str:
    rows = evidence_rows[:max_evidence_rows]
    evidence_text = "\n\n".join(
        f"[{i + 1}] {_format_evidence_row(r)}" for i, r in enumerate(rows)
    )
    support_summary = "  ".join(
        f"{k.upper()}: {v:.2f}" for k, v in sorted(support_priors.items())
    )
    return _GENERATION_PROMPT_TEMPLATE.format(
        route=hints.route,
        route_block=_ROUTE_PROMPT_BLOCKS.get(hints.route, ""),
        question=question.query,
        choice_a=question.answers["a"],
        choice_b=question.answers["b"],
        choice_c=question.answers["c"],
        choice_d=question.answers["d"],
        support_summary=support_summary or "N/A",
        evidence=evidence_text or "[no evidence retrieved]",
    )


def _format_evidence_row(hit: RetrievalHit) -> str:
    parts = [f"source={hit.source_type}"]
    if hit.camera_id:
        parts.append(f"camera={hit.camera_id}")
    if hit.day:
        parts.append(f"day={hit.day}")
    if hit.absolute_start is not None and hit.absolute_end is not None:
        parts.append(f"time={hit.absolute_start}-{hit.absolute_end}")
    header = " ".join(parts)
    body_parts = []
    if hit.transcript_text:
        body_parts.append(f"transcript: {hit.transcript_text}")
    if hit.event_summary:
        body_parts.append(f"event: {hit.event_summary}")
    if hit.ocr_text:
        body_parts.append(f"ocr: {hit.ocr_text}")
    body = "\n".join(body_parts) if body_parts else "[no text]"
    return f"{header}\n{body}"


def extract_answer(raw_text: str, support_priors: Dict[str, float]) -> AnswerChoice:
    """Parse FINAL_ANSWER line; fall back to highest support prior on failure."""
    match = _FINAL_ANSWER_RE.search(raw_text)
    if match:
        return match.group(1).lower()  # type: ignore[return-value]
    # Fall back to the answer choice with the highest cumulative support
    if support_priors:
        return max(support_priors, key=support_priors.get)  # type: ignore[return-value]
    return "a"


def generate_answer(
    question: EvalQuestion,
    hints: RouteHints,
    evidence_rows: List[RetrievalHit],
    support_priors: Dict[str, float],
    llm_client: Any,
    max_evidence_rows: int = 50,
) -> Prediction:
    """Run generation and return a Prediction with the extracted answer."""
    raise NotImplementedError("Implemented in issue #9")
