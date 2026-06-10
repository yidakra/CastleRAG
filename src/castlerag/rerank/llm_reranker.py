"""LLM-as-reranker prompts and scoring (SPEC §5).

Model: Qwen/Qwen3-VL-8B-Instruct (same as generator — one model for both).
Ablation: OpenGVLab/InternVL3-8B.

Scoring formula (SPEC §5.4):
  final_rerank_score = 0.7 * relevance + 0.3 * max(support.values())
  Discard if keep=False or relevance <= 1.
  Retain top 4 candidate packs globally.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from castlerag.routing.question_router import RouteHints
from castlerag.schemas import EvalQuestion, RerankerOutput, RetrievalHit

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
    rank: int,
    route: str,
    hit: RetrievalHit,
    transcript_chunks: str = "",
    event_summary: str = "",
    ocr_text: str = "",
    aux_summary: str = "",
) -> str:
    return (
        f"Candidate pack {rank}\n"
        f"Route: {route}\n"
        f"Primary source: {hit.source_type}\n"
        f"Day: {hit.day or 'N/A'}\n"
        f"Camera: {hit.camera_id or 'N/A'}\n"
        f"Participant: {hit.participant_id or 'N/A'}\n"
        f"Time: {hit.absolute_start} to {hit.absolute_end}\n"
        f"Transcript evidence:\n{transcript_chunks or '[none]'}\n\n"
        f"Event summary:\n{event_summary or '[not available]'}\n\n"
        f"OCR evidence:\n{ocr_text or '[none]'}\n\n"
        f"Auxiliary evidence:\n{aux_summary or '[none]'}"
    )


def compute_rerank_score(output: RerankerOutput) -> float:
    max_support = max(output.support.values())
    return 0.7 * output.relevance + 0.3 * max_support


def parse_reranker_response(raw: str) -> RerankerOutput:
    """Extract and parse the JSON block from a reranker LLM response.

    Tries each '{' position left-to-right and returns the first span that
    produces valid JSON, avoiding greedy-regex over-capture when the model
    emits extra brace-containing text before or after the target object.
    """
    start = 0
    while True:
        brace = raw.find("{", start)
        if brace == -1:
            break
        # find the matching closing brace by tracking depth
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
                        out.final_rerank_score = compute_rerank_score(out)
                        return out
                    except (json.JSONDecodeError, Exception):
                        pass
                    break
        start = brace + 1

    raise ValueError(f"No valid reranker JSON found in response: {raw[:200]!r}")


def rerank_candidates(
    question: EvalQuestion,
    hints: RouteHints,
    candidate_packs: List[Dict[str, Any]],
    llm_client: Any,
    top_k: int = 4,
    min_relevance: int = 1,
) -> List[Dict[str, Any]]:
    """Rerank candidate packs and return the top_k kept packs with scores."""
    raise NotImplementedError("Implemented in issue #8")
