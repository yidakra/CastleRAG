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

import base64
import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from castlerag.routing.question_router import RouteHints
from castlerag.schemas import AnswerChoice, EvalQuestion, Prediction, RetrievalHit

_FINAL_ANSWER_RE = re.compile(
    r"(?mi)^\s*FINAL_ANSWER:\s*([abcd])\s*$",
)
_ABSTAIN_RE = re.compile(
    r"(?mi)^\s*FINAL_ANSWER:\s*(?:abstain|none|insufficient|n/?a)\b",
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
    is insufficient and mark the rationale low-confidence — but you must
    still commit to a letter; never write FINAL_ANSWER: abstain.
  - localise: every count, object-location, or spatial claim must cite
    a specific camera and timestamp.
  - ground: confidence must come from cited evidence,
    not from option plausibility or outside knowledge.
- You MUST end with exactly one line, and the value MUST be a single letter:
    FINAL_ANSWER: a
    FINAL_ANSWER: b
    FINAL_ANSWER: c
    FINAL_ANSWER: d
  Never substitute "abstain", "none", "insufficient", or any other token.
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


def _b64_frame(path: str) -> Optional[str]:
    """Read a frame JPEG and return its base64 string, or None if missing/unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return base64.b64encode(p.read_bytes()).decode()
    except OSError:
        return None


def _gather_frame_paths(
    evidence_rows: List[RetrievalHit], max_frames: int = 8
) -> List[str]:
    """Collect deduped frame paths from evidence rows, capped at max_frames."""
    paths: List[str] = []
    seen: set = set()
    for row in evidence_rows:
        for p in row.sampled_frame_paths:
            if p not in seen:
                seen.add(p)
                paths.append(p)
            if len(paths) >= max_frames:
                return paths
    return paths


def _build_user_content(
    prompt: str,
    evidence_rows: List[RetrievalHit],
    max_frames: int,
) -> Any:
    """Return the user message content for a chat call.

    When sampled frame JPEGs are available on disk they are base64-encoded and
    appended as image_url items alongside the text, making the call multimodal.
    Falls back to the plain prompt string when no frames are present or files
    are missing.
    """
    encoded = [
        b
        for b in (_b64_frame(p) for p in _gather_frame_paths(evidence_rows, max_frames))
        if b
    ]
    if not encoded:
        return prompt
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in encoded:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    return user_content


def build_messages(
    question: EvalQuestion,
    hints: RouteHints,
    evidence_rows: List[RetrievalHit],
    support_priors: Dict[str, float],
    max_evidence_rows: int = 50,
    max_frames: int = 8,
) -> List[Dict[str, Any]]:
    """Return the system+user message list for the generation LLM.

    When sampled frame JPEGs are available on disk they are base64-encoded and
    appended as image_url items in the user message, making the call multimodal.
    Falls back to plain text when no frames are present or files are missing.
    """
    prompt = build_prompt(
        question=question,
        hints=hints,
        evidence_rows=evidence_rows,
        support_priors=support_priors,
        max_evidence_rows=max_evidence_rows,
    )
    user_content = _build_user_content(prompt, evidence_rows, max_frames)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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


def extract_answer(
    raw_text: str,
    support_priors: Dict[str, float],
    question_id: Optional[str] = None,
) -> AnswerChoice:
    """Parse a strict FINAL_ANSWER line; fall back when the model abstains.

    When the model writes ``FINAL_ANSWER: abstain`` (or a similar non-letter
    sentinel) and the support priors are all zero, picking the alphabetic
    fallback ("a") is a strong bias: the indexed slice is partial, so this
    pathway covers most questions on Codabench.  Deriving the fallback from a
    SHA1 of ``question_id`` instead yields a deterministic uniform distribution
    across a/b/c/d while preserving reproducibility.
    """
    matches = [match.group(1).lower() for match in _FINAL_ANSWER_RE.finditer(raw_text)]
    if len(matches) == 1:
        return matches[0]  # type: ignore[return-value]
    if len(matches) > 1:
        unique = set(matches)
        if len(unique) == 1:
            return matches[0]  # type: ignore[return-value]
    if _ABSTAIN_RE.search(raw_text):
        return _fallback_answer(support_priors, question_id=question_id)
    # Reject free-floating choice letters; generation must use FINAL_ANSWER.
    if _ANSWER_LETTER_RE.search(raw_text):
        return _fallback_answer(support_priors, question_id=question_id)
    # No FINAL_ANSWER, no abstain sentinel, no stray letter, no priors: route
    # through the fallback so a supplied question_id yields a deterministic
    # uniform pick instead of a constant "a" bias.
    return _fallback_answer(support_priors, question_id=question_id)


def _fallback_answer(
    support_priors: Dict[str, float],
    question_id: Optional[str] = None,
) -> AnswerChoice:
    """Return the best fallback answer choice.

    Order of preference:
      1. The choice with the strictly-highest support prior.
      2. If priors are tied or empty and ``question_id`` is supplied,
         a deterministic uniform pick derived from sha1(question_id).
      3. Alphabetic fallback ('a') — kept for legacy callers.
    """
    if support_priors:
        ordered = sorted(
            support_priors.items(),
            key=lambda item: (-item[1], item[0]),
        )
        best_choice, best_score = ordered[0]
        runner_score = ordered[1][1] if len(ordered) > 1 else None
        if runner_score is None or best_score > runner_score:
            return best_choice  # type: ignore[return-value]
    if question_id is not None:
        digest = hashlib.sha1(question_id.encode("utf-8")).digest()
        return "abcd"[digest[0] % 4]  # type: ignore[return-value]
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


def choice_permutation(question_id: str) -> Dict[str, str]:
    """Return a deterministic shuffle map presented_letter -> original_letter.

    Uses sha1(question_id) so the same question always yields the same order,
    keeping the eval reproducible while breaking the late-position bias that
    small multiple-choice models exhibit on weak evidence.
    """
    digest = hashlib.sha1(question_id.encode("utf-8")).digest()
    originals = sorted("abcd", key=lambda letter: digest[ord(letter) - ord("a")])
    return dict(zip("abcd", originals))


def generate_answer(
    question: EvalQuestion,
    hints: RouteHints,
    evidence_rows: List[RetrievalHit],
    support_priors: Dict[str, float],
    llm_client: Any,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    max_evidence_rows: int = 50,
    shuffle_choices: bool = False,
) -> Prediction:
    """Run grounded answer generation and return a normalized Prediction."""
    rows = evidence_rows[:max_evidence_rows]

    if shuffle_choices:
        presented_to_original = choice_permutation(question.question_id)
        prompt_question = question.model_copy(
            update={
                "answers": {
                    presented: question.answers[original]
                    for presented, original in presented_to_original.items()
                }
            }
        )
        prompt_priors = {
            presented: support_priors.get(original, 0.0)
            for presented, original in presented_to_original.items()
        }
    else:
        presented_to_original = {letter: letter for letter in "abcd"}
        prompt_question = question
        prompt_priors = support_priors

    messages = build_messages(
        question=prompt_question,
        hints=hints,
        evidence_rows=rows,
        support_priors=prompt_priors,
        max_evidence_rows=max_evidence_rows,
    )
    raw_answer_text = _call_generation_llm_with_model(llm_client, messages, model=model)
    presented_answer = extract_answer(
        raw_answer_text, prompt_priors, question_id=question.question_id
    )
    predicted_answer: AnswerChoice = presented_to_original[presented_answer]  # type: ignore[assignment]

    # When nothing supports any choice — no evidence retrieved, or the reranker
    # credited no choice with support — a forced MCQ answer is just a guess, and
    # the LLM reliably defaults to "a". Override it with a deterministic uniform
    # pick so the guess is unbiased, and record that the answer is unsupported.
    is_supported = bool(rows) and max(support_priors.values(), default=0.0) > 0.0
    if not is_supported:
        predicted_answer = _fallback_answer({}, question_id=question.question_id)

    return Prediction(
        question_id=question.question_id,
        predicted_answer=predicted_answer,
        route=hints.route,
        support_priors=support_priors or None,
        top_evidence_ids=[hit.record_id for hit in rows],
        raw_answer_text=raw_answer_text,
        is_supported=is_supported,
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


# ---------------------------------------------------------------------------
# Free-form (open-question) generation — for the UI, which asks open questions
# rather than CASTLE MCQs. The MCQ generator above forces a `FINAL_ANSWER: x`
# sentinel that is meaningless without real choices and leaks into the answer
# the dashboard shows; this path answers the question directly instead.
# ---------------------------------------------------------------------------

# Matches the MCQ sentinel anywhere (the strict line-anchored _FINAL_ANSWER_RE
# misses it when the model writes it inline, e.g. "...answer is: FINAL_ANSWER: c").
_FINAL_ANSWER_ANYWHERE_RE = re.compile(r"(?is)FINAL_ANSWER:\s*[a-z/]+\b\.?")
# Inline evidence citations the MCQ/freeform prompt instructs the LLM to write.
# These are meaningless in the UI (the videos are already shown) and dcc.Markdown
# renders bare [text] as reference links that go nowhere.
_CITATION_RE = re.compile(
    r"\[(?:camera=[^\]]*|aux=[^\]]*)\](?:\([^)]*\))?"
)
# A trailing "Thus, the correct answer is:" clause left dangling once the
# sentinel it pointed at is removed.
_DANGLING_SCAFFOLD_RE = re.compile(
    r"(?is)(?:thus|therefore|hence|so)?,?\s*(?:the\s+)?"
    r"(?:correct\s+|final\s+)?answer\s+is\s*[:\-]?\s*$"
)

_FREEFORM_SYSTEM_PROMPT = """\
You are CastleRAG answering an OPEN question about the CASTLE recordings — not a
multiple-choice question. Use ONLY the supplied evidence.

Rules:
- Open with a direct, specific answer in one sentence, then at most one sentence
  of support.
- Ground every claim in the evidence, never in outside knowledge.
- Do NOT include citations, links, or bracket references (no [camera=...], no
  [aux=...], no markdown links). Write plain prose only.
- If the evidence does not actually answer the question, or is conflicting or
  insufficient, say so plainly — do NOT invent an answer.
- Never output a letter choice, an "Option A/B/C/D", or a "FINAL_ANSWER:" line.
"""

_FREEFORM_USER_TEMPLATE = """\
Answer this open question about the CASTLE recordings using only the evidence.

Question route:
{route}

Route-specific instructions:
{route_block}

Question:
{question}

Evidence:
{evidence}{context_block}
"""


def clean_answer_text(raw_text: str) -> str:
    """Strip the MCQ ``FINAL_ANSWER`` sentinel and any dangling scaffold clause.

    The MCQ generator must end with ``FINAL_ANSWER: <letter>``; when that output
    is shown verbatim for a free-form question it reads as nonsense (e.g.
    "...the correct answer is: FINAL_ANSWER: c"). This removes the sentinel
    wherever it appears and trims a now-orphaned "the answer is:" lead-in.
    """
    if not raw_text:
        return ""
    text = _FINAL_ANSWER_ANYWHERE_RE.sub("", raw_text).rstrip()
    text = _DANGLING_SCAFFOLD_RE.sub("", text).rstrip()
    # Strip evidence citations ([camera=...] / [aux=...]) — they render as broken
    # links in the UI and the videos are shown in the evidence viewer anyway.
    text = _CITATION_RE.sub("", text)
    # Clean up dangling conjunctions/prepositions left before punctuation.
    text = re.sub(r"\b(?:and|or|from|in|at|by|of)\s+([.,;])", r"\1", text)
    # Collapse multiple spaces and tidy " ." → ".".
    text = re.sub(r" +([.,;])", r"\1", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def generate_freeform_answer(
    question: EvalQuestion,
    hints: RouteHints,
    evidence_rows: List[RetrievalHit],
    llm_client: Any,
    *,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    max_evidence_rows: int = 50,
    max_frames: int = 8,
    refinement_context: Optional[str] = None,
) -> str:
    """Answer an open question directly from evidence (no MCQ sentinel).

    ``refinement_context`` — when set (refinement pass) — appends the
    human reviewer's camera verdicts and justifications after the evidence
    block, grounding the answer in human-validated signals.

    Sampled frame JPEGs on the evidence rows are base64-encoded and attached
    as image_url items (mirroring :func:`build_messages`), so static-visual
    and mixed questions reach the Qwen3-VL model with the actual frame
    evidence rather than text alone. Falls back to a plain-text message when
    no frames are available.
    """
    rows = evidence_rows[:max_evidence_rows]
    evidence_text = "\n\n".join(_enumerate_evidence_rows(rows)) or _MISSING_EVIDENCE_ROW
    context_block = (
        f"\n\nReviewer feedback on previous evidence (use as guidance only — "
        f"do NOT cite timestamps from this block; only cite timestamps from "
        f"the Evidence section above):\n{refinement_context}"
        if refinement_context
        else ""
    )
    user = _FREEFORM_USER_TEMPLATE.format(
        route=hints.route,
        route_block=_ROUTE_PROMPT_BLOCKS.get(hints.route, ""),
        question=question.query,
        evidence=evidence_text,
        context_block=context_block,
    )
    user_content = _build_user_content(user, rows, max_frames)
    messages = [
        {"role": "system", "content": _FREEFORM_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    raw = _call_generation_llm_with_model(llm_client, messages, model=model)
    return clean_answer_text(raw)
