"""Chat engine contract and a placeholder implementation for the UI backbone.

The UI talks to a :class:`ChatEngine`: ``answer(question, choices)`` returns a
:class:`ChatTurnResult` carrying the predicted choice, a short rationale, the
per-choice support priors, and a list of :class:`EvidenceRef` rows that the UI
turns into YouTube embeds and Plotly figures.

:class:`PlaceholderEngine` fabricates deterministic, structurally valid results
without any models, Qdrant, or vLLM — enough to wire and demo the whole UI.  A
future ``RagEngine`` wrapping ``castlerag.eval.run_eval`` implements the same
protocol; :class:`EvidenceRef` mirrors the fields of ``schemas.RetrievalHit``
(plus ``hour`` / ``start_seconds`` needed to seek the embed) so the swap is
mechanical.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from castlerag.routing.question_router import route_question

_CHOICES = ("a", "b", "c", "d")

# Fallback roster used when no mirror keys are supplied: day1 egocentric cameras
# over the morning hours that the local partial-download slices ship.
_DEFAULT_ROSTER: Tuple[Tuple[str, str, int], ...] = tuple(
    ("day1", camera, hour)
    for camera in ("Allie", "Bjorn", "Celine")
    for hour in (8, 9, 10)
)

_SOURCE_TEMPLATES = {
    "transcript_window": "{camera} is heard discussing the topic in this window.",
    "main_clip": "{camera}'s ego clip shows the relevant activity on screen.",
    "main_event_summary": "Event summary covering {camera}'s actions over ~2 min.",
    "aux_photo": "Auxiliary photo associated with {camera} around this time.",
    "aux_video": "Auxiliary video segment near {camera}'s timeline.",
}


@dataclass
class EvidenceRef:
    """A single piece of evidence the UI can embed and chart.

    Mirrors ``schemas.RetrievalHit`` with the extra ``hour`` / ``start_seconds``
    / ``end_seconds`` fields the YouTube mirror needs to seek a clip.
    """

    record_id: str
    source_type: str
    modality: str
    day: str
    camera_id: str
    hour: int
    start_seconds: float
    end_seconds: float
    score: float
    text: str


@dataclass
class ChatTurnResult:
    """The full result of one chat turn, consumed by the UI callbacks."""

    answer_text: str
    route: str
    support_priors: Dict[str, float]
    evidence: List[EvidenceRef] = field(default_factory=list)
    predicted_choice: Optional[str] = None
    is_placeholder: bool = True


class ChatEngine(Protocol):
    """Protocol every chat backend (placeholder or real RAG) implements."""

    def answer(
        self, question: str, choices: Optional[Dict[str, str]] = None
    ) -> ChatTurnResult:
        """Return a structured answer for ``question`` and optional ``choices``."""
        ...


@dataclass
class PlaceholderEngine:
    """Deterministic stand-in engine — no models, no retrieval, no network.

    Given the same question it always returns the same route, priors, and
    evidence, so demos and tests are reproducible.  ``roster`` defaults to the
    keys of a supplied :class:`~castlerag.ui.youtube.YouTubeMirror` so fabricated
    evidence always resolves to a real embed.
    """

    roster: Tuple[Tuple[str, str, int], ...] = _DEFAULT_ROSTER
    n_evidence: int = 4

    @classmethod
    def from_mirror(cls, mirror: object, **kwargs: object) -> "PlaceholderEngine":
        """Build an engine whose evidence roster is the mirror's mapped keys."""
        mapping = getattr(mirror, "mapping", {}) or {}
        roster = tuple(sorted(mapping.keys())) or _DEFAULT_ROSTER
        return cls(roster=roster, **kwargs)  # type: ignore[arg-type]

    def answer(
        self, question: str, choices: Optional[Dict[str, str]] = None
    ) -> ChatTurnResult:
        """Produce a deterministic, structurally valid placeholder answer."""
        resolved_choices = choices or {key: f"Option {key.upper()}" for key in _CHOICES}
        hints = route_question(question=question, choices=resolved_choices)

        seed = int(hashlib.sha1(question.strip().lower().encode()).hexdigest(), 16)
        rng = random.Random(seed)

        priors = _support_priors(rng)
        predicted = max(_CHOICES, key=lambda key: (priors[key], key))
        evidence = self._fabricate_evidence(rng, hints.route)

        answer_text = (
            f"Placeholder answer: **{predicted}** — {resolved_choices[predicted]}. "
            f"(routed as '{hints.route}'; the RAG pipeline is not connected yet, so "
            f"this turn is synthetic with {len(evidence)} stub evidence rows.)"
        )
        return ChatTurnResult(
            answer_text=answer_text,
            route=hints.route,
            support_priors=priors,
            evidence=evidence,
            predicted_choice=predicted,
            is_placeholder=True,
        )

    def _fabricate_evidence(
        self, rng: random.Random, route: str
    ) -> List[EvidenceRef]:
        """Build a deterministic list of stub evidence rows for the roster."""
        if not self.roster:
            return []
        count = min(self.n_evidence, len(self.roster))
        keys = rng.sample(list(self.roster), count)
        source_types = list(_SOURCE_TEMPLATES.keys())
        evidence: List[EvidenceRef] = []
        for rank, (day, camera, hour) in enumerate(keys):
            source_type = source_types[rank % len(source_types)]
            start = float(rng.choice([0, 30, 60, 120, 300, 600]))
            modality = "text" if source_type == "transcript_window" else "video"
            evidence.append(
                EvidenceRef(
                    record_id=f"{day}_{camera}_{hour:02d}_{rank:04d}",
                    source_type=source_type,
                    modality=modality,
                    day=day,
                    camera_id=camera,
                    hour=hour,
                    start_seconds=start,
                    end_seconds=start + 30.0,
                    score=round(1.0 - rank * 0.15 - rng.random() * 0.05, 4),
                    text=_SOURCE_TEMPLATES[source_type].format(camera=camera),
                )
            )
        return evidence


def _support_priors(rng: random.Random) -> Dict[str, float]:
    """Return normalized per-choice support priors that sum to 1.0."""
    raw = {key: rng.random() + 0.05 for key in _CHOICES}
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}
