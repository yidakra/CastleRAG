"""LLM-drafted review suggestions for the CastleRAG dashboard.

Two small, pure helpers used by the live :class:`~castlerag.ui.rag_engine.RagEngine`
to draft text the reviewer then edits:

* :func:`suggest_justification_text` — a one-line justification for a per-camera
  verdict (Confirm / Refine / Reject), grounded in the retrieved evidence.
* :func:`suggest_refined_query_text` — a refined retrieval query folding in the
  reviewer's per-camera verdicts and notes.

Both take an OpenAI-compatible chat client (the same one the answer/rerank
pipelines use) and mirror the call shape in
``castlerag.generation.answer._call_generation_llm_with_model``. They are
model-agnostic and side-effect free so they are trivial to unit test with a
fake client.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

_DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Human-readable phrasing for each stored verdict state.
_VERDICT_PHRASE = {
    "confirmed": "CONFIRMS",
    "flagged": "is INCONCLUSIVE for (needs a clearer angle on)",
    "rejected": "does NOT support",
}
# The three verdicts must drive the refined query differently:
#   confirmed -> POSITIVE: good evidence, retain / stay consistent with it.
#   flagged   -> seek a CLEARER view of this angle (relevant but unclear).
#   rejected  -> EXCLUDE (ruled out — the camera is also must_not-filtered).
# Kept separate so Confirm is a real positive signal, not "done, look elsewhere".
_CONFIRMED_STATES = {"confirmed"}
_FLAGGED_STATES = {"flagged"}
_REJECTED_STATES = {"rejected"}


def _complete(
    llm_client: Any,
    messages: List[Dict[str, str]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    """Dispatch one chat-completion and return its text (mirrors answer.py)."""
    if hasattr(llm_client, "generate_from_messages"):  # test double / shim
        return str(llm_client.generate_from_messages(messages))
    if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        if not response.choices:
            return ""
        return str(response.choices[0].message.content or "")
    raise TypeError("llm_client is not an OpenAI-compatible chat client")


def suggest_justification_text(
    claim: str,
    camera_id: str,
    verdict: str,
    evidence_text: Optional[str],
    meta: Optional[Mapping[str, object]],
    llm_client: Any,
    *,
    model: str = _DEFAULT_MODEL,
    timeout: float = 30.0,
) -> str:
    """Draft a one-sentence justification for ``verdict`` on ``camera_id``.

    ``verdict`` is the stored state (``confirmed`` / ``flagged`` / ``rejected``).
    ``meta`` may carry ``clock_label``, ``place_label`` and ``match_score``.
    """
    meta = meta or {}
    when = str(meta.get("clock_label") or "the moment")
    where = str(meta.get("place_label") or "the scene")
    score = meta.get("match_score")
    evidence = (evidence_text or "").strip()

    # No evidence text + confirmed verdict → skip the LLM; a blank textarea is
    # better than a generated "no evidence" message that contradicts the user.
    if not evidence and (verdict or "").lower() == "confirmed":
        return ""

    evidence = evidence or "(no retrieved text for this angle)"

    verdict_lower = (verdict or "").lower()
    if verdict_lower == "confirmed":
        instruction = (
            "The reviewer has CONFIRMED this camera angle. "
            "Write one sentence (max ~25 words) explaining what in the evidence "
            "supports this confirmation. Accept the reviewer's judgement — do not "
            "question or contradict it."
        )
    elif verdict_lower == "rejected":
        instruction = (
            "The reviewer has REJECTED this camera angle. "
            "Write one sentence (max ~25 words) explaining what in the evidence "
            "led to rejection or why this angle does not support the claim."
        )
    else:
        instruction = (
            "The reviewer has FLAGGED this camera angle as inconclusive. "
            "Write one sentence (max ~25 words) explaining what makes this angle "
            "ambiguous or unclear."
        )

    system = (
        "You are an analyst writing brief justifications for a human reviewer's "
        "per-camera verdicts on surveillance evidence. "
        + instruction
        + " Return the sentence only — no preamble, quotes, or labels."
    )
    score_str = (
        f" (match score {float(score):.2f})" if isinstance(score, (int, float)) else ""
    )
    user = (
        f"Claim under review: {claim}\n"
        f"Camera: {camera_id} at {when} in {where}{score_str}\n"
        f"Retrieved evidence from this camera: {evidence}\n\n"
        "Justification:"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _complete(
        llm_client,
        messages,
        model=model,
        max_tokens=64,
        temperature=0.3,
        timeout=timeout,
    ).strip()


def suggest_refined_query_text(
    claim: str,
    reviews: Dict[str, Dict[str, str]],
    llm_client: Any,
    *,
    question: Optional[str] = None,
    model: str = _DEFAULT_MODEL,
    timeout: float = 30.0,
) -> str:
    """Draft a refined retrieval query from the per-camera verdicts/notes.

    ``question`` is the original user question and should anchor the query. For a
    free-form question ``claim`` is the *previous answer* (the UI's claim text),
    which may be wrong — passing it as the sole anchor biases the refined search
    back toward that answer, so it is demoted to a fallible prior here.
    """
    lines = []
    confirmed = []
    flagged = []
    rejected = []
    for camera_id, info in reviews.items():
        state = info.get("state", "pending")
        note = (info.get("justification") or "").strip()
        phrase = _VERDICT_PHRASE.get(state, "reviewed")
        lines.append(f"- {camera_id}: {phrase}" + (f" — {note}" if note else ""))
        if state in _CONFIRMED_STATES:
            confirmed.append(camera_id)
        elif state in _FLAGGED_STATES:
            flagged.append(camera_id)
        elif state in _REJECTED_STATES:
            rejected.append(camera_id)
    confirmed_block = ", ".join(confirmed) if confirmed else "none"
    flagged_block = ", ".join(flagged) if flagged else "none"
    rejected_block = ", ".join(rejected) if rejected else "none"

    system = (
        "You write a SHORT retrieval search query (one sentence, max 15 words) "
        "for a multi-camera video evidence system. The query must describe WHAT "
        "to find — objects, actions, people, locations — NOT instructions or "
        "verdicts. Stay anchored on the ORIGINAL QUESTION topic. Use the "
        "reviewer's verdicts only to steer the topic: confirmed = on the right "
        "track, stay focused; flagged = relevant but unclear, seek a clearer "
        "view; rejected = irrelevant, search elsewhere. "
        "Output ONLY the query text — no preamble, no quotes, no instructions."
    )
    user = (
        f"Original question: {question or claim}\n"
        f"Confirmed cameras (good evidence): {confirmed_block}\n"
        f"Flagged cameras (need clearer view): {flagged_block}\n"
        f"Rejected cameras (exclude): {rejected_block}\n\n"
        "Search query (max 15 words, keywords/topic only):"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _complete(
        llm_client,
        messages,
        model=model,
        max_tokens=128,
        temperature=0.3,
        timeout=timeout,
    ).strip()
