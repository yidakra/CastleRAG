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
    phrase = _VERDICT_PHRASE.get(verdict, "was reviewed for")
    meta = meta or {}
    when = str(meta.get("clock_label") or "the moment")
    where = str(meta.get("place_label") or "the scene")
    score = meta.get("match_score")
    evidence = (evidence_text or "").strip() or "(no retrieved text for this angle)"

    system = (
        "You are an analyst reviewing multi-camera surveillance evidence. "
        "Write a single concise justification (max ~25 words) explaining the "
        "reviewer's verdict for one camera angle, grounded ONLY in the provided "
        "evidence. State plainly if the evidence is missing or insufficient. "
        "Return the sentence only — no preamble, quotes, or labels."
    )
    user = (
        f"Claim under review: {claim}\n"
        f"Camera: {camera_id} at {when} in {where}"
        + (
            f" (match score {float(score):.2f})"
            if isinstance(score, (int, float))
            else ""
        )
        + "\n"
        f"Reviewer verdict: this angle {phrase} the claim.\n"
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
    verdict_block = "\n".join(lines) if lines else "- (no verdicts)"
    confirmed_block = ", ".join(confirmed) if confirmed else "none"
    flagged_block = ", ".join(flagged) if flagged else "none"
    rejected_block = ", ".join(rejected) if rejected else "none"

    system = (
        "You compose ONE refined retrieval query for a multi-camera "
        "investigation. Anchor the query on the ORIGINAL QUESTION. The reviewer's "
        "per-camera notes are explicit instructions — treat them as the PRIMARY "
        "steering signal. The three verdicts mean different things and must be "
        "honoured differently: CONFIRMED angles are good evidence — KEEP and stay "
        "consistent with them, do NOT steer away; FLAGGED angles are relevant but "
        "unclear — seek a CLEARER view of them; REJECTED angles are ruled out — "
        "exclude them, do not seek further evidence from them. A prior tentative "
        "answer may be shown for context, but it MAY BE WRONG: do not assume it is "
        "correct. Return 1-2 sentences only — the query text, no preamble or quotes."
    )
    user = (
        f"Original question: {question or claim}\n"
        f"Prior tentative answer (context only, may be wrong): {claim}\n"
        f"Reviewer notes per camera (PRIMARY — follow these instructions):\n"
        f"{verdict_block}\n"
        f"Angles CONFIRMED (keep / stay consistent with — do not steer away): "
        f"{confirmed_block}\n"
        f"Angles needing a clearer view (seek more): {flagged_block}\n"
        f"Angles rejected (exclude from the search): {rejected_block}\n\n"
        "Refined query:"
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
