"""LLM-as-reranker prompts and scoring (SPEC §5).

Model: Qwen/Qwen3-VL-8B-Instruct (same as generator — one model for both).
Ablation: OpenGVLab/InternVL3-8B.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Sequence

from castlerag.routing.question_router import RouteHints
from castlerag.schemas import (
    EvalQuestion,
    EvidencePack,
    RerankedEvidencePack,
    RerankerOutput,
    RerankResult,
    RetrievalHit,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_RERANK_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

_RERANKER_PROMPT_TEMPLATE = """\
You are ranking a route-specific evidence pack for a multiple-choice CASTLE question.

Question:
{question}

Question route:
{route}

Answer choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Evidence pack:
{candidate_text}

Score this candidate on two axes:
1. Evidence relevance from 0 to 4
2. Support for each answer choice from 0 to 4

Return strict JSON:
{{
  "relevance": 0-4,
  "support": {{"a": 0-4, "b": 0-4, "c": 0-4, "d": 0-4}},
  "keep": true|false,
  "rationale": "<<=40 words>"
}}"""


def format_candidate_pack(
    pack: EvidencePack | None = None,
    *,
    rank: int | None = None,
    route: str | None = None,
    hit: RetrievalHit | None = None,
    transcript_chunks: str = "",
    event_summary: str = "",
    ocr_text: str = "",
    aux_summary: str = "",
    frame_descriptions: str = "",
) -> str:
    """Render a route-aware evidence pack as reranker input text.

    Supports the typed EvidencePack path used by issue #7 and the legacy
    keyword-argument shape already referenced by existing unit tests.
    """
    if pack is None:
        if hit is None or route is None:
            raise ValueError("Either pack or both route and hit must be provided")
        pack = EvidencePack(
            pack_id=f"legacy_{hit.record_id}",
            route=route,  # type: ignore[arg-type]
            primary_hit=hit,
            retrieval_score=hit.score,
            evidence_rows=[hit],
            transcript_evidence=[transcript_chunks] if transcript_chunks else [],
            event_summaries=[event_summary] if event_summary else [],
            ocr_spans=[ocr_text] if ocr_text else [],
            frame_descriptions=[frame_descriptions] if frame_descriptions else [],
            auxiliary_notes=[aux_summary] if aux_summary else [],
        )

    primary = pack.primary_hit
    if rank is not None:
        header = f"Candidate pack {rank}"
    else:
        header = f"Candidate pack {pack.pack_id}"
    time_text = _format_time_range(primary)
    transcript_text = _format_section("Transcript evidence", pack.transcript_evidence)
    event_text = _format_section("Event summaries", pack.event_summaries)
    ocr_text = _format_section("OCR spans", pack.ocr_spans)
    frame_text = _format_section("Sampled-frame descriptions", pack.frame_descriptions)
    aux_text = _format_section("Auxiliary notes", pack.auxiliary_notes)

    return (
        f"{header}\n"
        f"Pack id: {pack.pack_id}\n"
        f"Route: {pack.route}\n"
        f"Primary source: {primary.source_type}\n"
        f"Primary modality: {primary.modality}\n"
        f"Retrieval score: {pack.retrieval_score:.4f}\n"
        f"Day: {primary.day or 'N/A'}\n"
        f"Camera: {primary.camera_id or 'N/A'}\n"
        f"Participant: {primary.participant_id or 'N/A'}\n"
        f"Time: {time_text}\n\n"
        f"{transcript_text}\n\n"
        f"{event_text}\n\n"
        f"{ocr_text}\n\n"
        f"{frame_text}\n\n"
        f"{aux_text}"
    )


def build_reranker_prompt(
    question: EvalQuestion,
    pack: EvidencePack,
    rank: int | None = None,
) -> str:
    """Build the strict-JSON reranker prompt for one candidate pack."""
    return _RERANKER_PROMPT_TEMPLATE.format(
        question=question.query,
        route=pack.route,
        choice_a=question.answers["a"],
        choice_b=question.answers["b"],
        choice_c=question.answers["c"],
        choice_d=question.answers["d"],
        candidate_text=format_candidate_pack(pack, rank=rank),
    )


def compute_rerank_score(
    output: RerankerOutput,
    relevance_weight: float = 0.7,
    support_weight: float = 0.3,
) -> float:
    """Compute the final rerank score from reranker JSON output."""
    max_support = max(output.support.values())
    return relevance_weight * output.relevance + support_weight * max_support


def parse_reranker_response(
    raw: str,
    relevance_weight: float = 0.7,
    support_weight: float = 0.3,
) -> RerankerOutput:
    """Extract and parse the JSON block from a reranker LLM response."""
    start = 0
    while True:
        brace = raw.find("{", start)
        if brace == -1:
            break
        depth = 0
        for i, ch in enumerate(raw[brace:], brace):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[brace : i + 1]
                    try:
                        data = json.loads(candidate)
                        out = RerankerOutput.model_validate(data)
                        out.final_rerank_score = compute_rerank_score(
                            out,
                            relevance_weight=relevance_weight,
                            support_weight=support_weight,
                        )
                        return out
                    except (json.JSONDecodeError, ValueError):
                        break
        start = brace + 1

    raise ValueError(f"No valid reranker JSON found in response: {raw[:200]!r}")


def rerank_candidates(
    question: EvalQuestion,
    hints: RouteHints,
    candidate_packs: Sequence[EvidencePack | Mapping[str, Any]],
    llm_client: Any,
    top_k: int = 4,
    min_relevance: int = 1,
    *,
    max_evidence_rows: int = 50,
    model: str = DEFAULT_RERANK_MODEL,
    relevance_weight: float = 0.7,
    support_weight: float = 0.3,
) -> RerankResult:
    """Rerank evidence packs with a local Qwen3-VL-compatible chat client."""
    ranked: list[RerankedEvidencePack] = []

    for rank, raw_pack in enumerate(candidate_packs, start=1):
        pack = _coerce_pack(raw_pack, hints)
        prompt = build_reranker_prompt(question, pack, rank=rank)
        try:
            raw_response = _invoke_reranker(
                llm_client=llm_client,
                model=model,
                prompt=prompt,
            )
            reranker_output = parse_reranker_response(
                raw_response,
                relevance_weight=relevance_weight,
                support_weight=support_weight,
            )
        except ValueError as exc:
            LOGGER.warning(
                "Skipping reranker candidate %s after parse failure: %s",
                pack.pack_id,
                exc,
            )
            continue

        if not reranker_output.keep or reranker_output.relevance <= min_relevance:
            continue

        final_score = compute_rerank_score(
            reranker_output,
            relevance_weight=relevance_weight,
            support_weight=support_weight,
        )
        ranked.append(
            RerankedEvidencePack(
                pack=pack,
                reranker_output=reranker_output,
                final_rerank_score=final_score,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.final_rerank_score,
            -item.pack.retrieval_score,
            item.pack.pack_id,
        )
    )
    kept = ranked[:top_k]
    return RerankResult(
        route=hints.route,
        kept_packs=kept,
        support_priors=_aggregate_support_priors(kept),
        evidence_rows=_flatten_evidence_rows(kept, max_rows=max_evidence_rows),
    )


def _coerce_pack(
    raw_pack: EvidencePack | Mapping[str, Any],
    hints: RouteHints,
) -> EvidencePack:
    if isinstance(raw_pack, EvidencePack):
        return raw_pack

    data = dict(raw_pack)
    if "route" not in data:
        data["route"] = hints.route
    if "retrieval_score" not in data and "primary_hit" in data:
        primary = data["primary_hit"]
        if isinstance(primary, RetrievalHit):
            data["retrieval_score"] = primary.score
    return EvidencePack.model_validate(data)


def _invoke_reranker(
    *,
    llm_client: Any,
    model: str,
    prompt: str,
    max_tokens: int = 256,
) -> str:
    """Call an OpenAI-compatible vLLM chat endpoint or a test double."""
    if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        if not response.choices:
            return ""
        content = response.choices[0].message.content
        return _normalize_content(content)
    if hasattr(llm_client, "create"):
        response = llm_client.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return _normalize_content(response)
    raise TypeError("llm_client must expose chat.completions.create or create")


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunk.strip() for chunk in chunks if chunk).strip()
    return str(content).strip()


def _aggregate_support_priors(
    kept_packs: Sequence[RerankedEvidencePack],
) -> dict[str, float]:
    priors = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    for item in kept_packs:
        for choice, score in item.reranker_output.support.items():
            priors[choice] += float(score)
    return priors


def _flatten_evidence_rows(
    kept_packs: Sequence[RerankedEvidencePack],
    *,
    max_rows: int,
) -> list[RetrievalHit]:
    rows: list[RetrievalHit] = []
    seen: set[str] = set()
    for item in kept_packs:
        for row in item.pack.evidence_rows:
            if row.record_id in seen:
                continue
            rows.append(row)
            seen.add(row.record_id)
            if len(rows) >= max_rows:
                return rows
    return rows


def _format_section(title: str, values: Sequence[str]) -> str:
    if not values:
        return f"{title}:\n[none]"
    lines = "\n".join(f"- {value}" for value in values)
    return f"{title}:\n{lines}"


def _format_time_range(hit: RetrievalHit) -> str:
    if hit.absolute_start is None or hit.absolute_end is None:
        return "N/A"
    return f"{hit.absolute_start} to {hit.absolute_end}"
