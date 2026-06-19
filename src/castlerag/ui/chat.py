"""Chat engine contract and a placeholder implementation for the UI backbone.

The UI talks to a :class:`ChatEngine` through two calls:

* ``answer(question, choices)`` opens a fresh investigation: it returns a
  :class:`ChatTurnResult` carrying a short answer, the single :class:`Claim`
  under review, and a ranked list of :class:`EvidenceMoment` rows — each moment
  being one ``(day, hour, start_seconds)`` seen from three synchronized cameras.
* ``refine(claim, refined_query, iteration)`` re-runs retrieval for the *same*
  claim with a sharper query, returning a stronger moment as the investigation
  converges.

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
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple

from castlerag.routing.question_router import route_question

_CHOICES = ("a", "b", "c", "d")

# The three synchronized cameras the evidence viewer shows per moment.  All are
# real CASTLE egocentric streams, so the YouTube mirror resolves each of them at
# the same (day, hour) — exactly what a "3 synchronized cameras" moment needs.
_CAMERA_TRIO: Tuple[str, str, str] = ("Bjorn", "Luca", "Klaus")

# Fallback roster used when no mirror keys are supplied: day1 egocentric cameras
# (real CASTLE stream names) over the morning hours.
_DEFAULT_ROSTER: Tuple[Tuple[str, str, int], ...] = tuple(
    ("day1", camera, hour)
    for camera in ("Allie", "Bjorn", "Cathal")
    for hour in (8, 9, 10)
)

_PLACES = ("Doorway", "Hallway", "Kitchen", "Living room", "Stairwell", "Reading nook")
_MOMENT_MINUTES = (29, 31, 44, 52)


class SupportLevel(str, Enum):
    """How strongly the gathered evidence backs the claim.

    A ``str`` enum so it serializes straight into a Dash ``dcc.Store`` (and back)
    without custom encoders.
    """

    UNSUPPORTED = "unsupported"
    PARTIAL = "partial"
    SUPPORTED = "supported"


# Dot colour shown next to each evidence moment, keyed by support level.
_DOT_COLORS: Dict[SupportLevel, str] = {
    SupportLevel.UNSUPPORTED: "#dc2626",
    SupportLevel.PARTIAL: "#d97706",
    SupportLevel.SUPPORTED: "#16a34a",
}


@dataclass
class EvidenceRef:
    """A single piece of evidence the legacy/real engine surfaces.

    Mirrors ``schemas.RetrievalHit`` with the extra ``hour`` / ``start_seconds``
    / ``end_seconds`` fields the YouTube mirror needs to seek a clip.  Retained
    so a real ``RagEngine`` keeps the same return contract.
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
class CameraAngle:
    """One of the three synchronized cameras viewing a moment."""

    camera_id: str
    day: str
    hour: int
    start_seconds: float
    match_score: float
    is_best: bool = False


@dataclass
class EvidenceMoment:
    """A single moment in time, viewed from three synchronized cameras."""

    moment_id: str
    clock_label: str
    place_label: str
    camera_count: int
    aggregate_score: float
    score_caption: str
    dot_color: str
    cameras: List[CameraAngle] = field(default_factory=list)

    def best_camera(self) -> CameraAngle:
        """Return the best-matching camera (or the first if none flagged)."""
        return next((cam for cam in self.cameras if cam.is_best), self.cameras[0])


@dataclass
class Claim:
    """The single claim an investigation verifies, and its current support."""

    text: str
    support: SupportLevel


@dataclass
class QueryGroup:
    """One entry in the left-hand thread: a query, its answer, and its moments."""

    group_id: str
    iteration: int
    question: str
    answer_text: str
    claim: Claim
    moments: List[EvidenceMoment] = field(default_factory=list)
    is_refinement: bool = False
    refined_query: Optional[str] = None


@dataclass
class ChatTurnResult:
    """The full result of one chat turn, consumed by the UI callbacks.

    ``claim`` / ``moments`` drive the new dashboard.  The legacy ``route`` /
    ``support_priors`` / ``evidence`` / ``predicted_choice`` fields stay
    populated so a real ``RagEngine`` keeps a single return contract.
    """

    answer_text: str
    route: str
    support_priors: Dict[str, float]
    evidence: List[EvidenceRef] = field(default_factory=list)
    predicted_choice: Optional[str] = None
    is_placeholder: bool = True
    claim: Optional[Claim] = None
    moments: List[EvidenceMoment] = field(default_factory=list)


class ChatEngine(Protocol):
    """Protocol every chat backend (placeholder or real RAG) implements."""

    def answer(
        self, question: str, choices: Optional[Dict[str, str]] = None
    ) -> ChatTurnResult:
        """Open an investigation: return a claim and ranked evidence moments."""
        ...

    def refine(
        self, claim: str, refined_query: str, iteration: int
    ) -> ChatTurnResult:
        """Re-run retrieval for the same ``claim`` with a sharper query."""
        ...


@dataclass
class PlaceholderEngine:
    """Deterministic stand-in engine — no models, no retrieval, no network.

    Given the same inputs it always returns the same claim, moments, and camera
    scores, so demos and tests are reproducible.  ``roster`` defaults to the keys
    of a supplied :class:`~castlerag.ui.youtube.YouTubeMirror` so the chosen scene
    always resolves to real embeds.
    """

    roster: Tuple[Tuple[str, str, int], ...] = _DEFAULT_ROSTER
    n_evidence: int = 3
    is_live: bool = False

    @classmethod
    def from_mirror(cls, mirror: object, **kwargs: object) -> "PlaceholderEngine":
        """Build an engine whose scene roster is the mirror's mapped keys."""
        mapping = getattr(mirror, "mapping", {}) or {}
        roster = tuple(sorted(mapping.keys())) or _DEFAULT_ROSTER
        return cls(roster=roster, **kwargs)  # type: ignore[arg-type]

    # -- public protocol ----------------------------------------------------

    def answer(
        self, question: str, choices: Optional[Dict[str, str]] = None
    ) -> ChatTurnResult:
        """Open an investigation with a partially-supported claim."""
        resolved_choices = choices or {key: f"Option {key.upper()}" for key in _CHOICES}
        hints = route_question(question=question, choices=resolved_choices)

        rng = random.Random(_seed(question))
        priors = _support_priors(rng)
        predicted = max(_CHOICES, key=lambda key: (priors[key], key))

        day, hour = self._pick_scene()
        moments = self._fabricate_moments(
            rng, day, hour, SupportLevel.PARTIAL, kind="rank", boost=0.0
        )
        best_cam = moments[0].best_camera().camera_id
        claim = Claim(
            text=f"The footage confirms {best_cam}'s involvement in the answer.",
            support=SupportLevel.PARTIAL,
        )
        answer_text = (
            f"The strongest footage places **{best_cam}** at the "
            f"{moments[0].place_label.lower()} around {moments[0].clock_label}."
        )
        return ChatTurnResult(
            answer_text=answer_text,
            route=hints.route,
            support_priors=priors,
            evidence=self._moments_to_evidence(moments),
            predicted_choice=predicted,
            is_placeholder=True,
            claim=claim,
            moments=moments,
        )

    def refine(
        self, claim: str, refined_query: str, iteration: int
    ) -> ChatTurnResult:
        """Re-run retrieval for ``claim``; support climbs as iterations rise."""
        rng = random.Random(_seed(f"{claim}|{refined_query}|{iteration}"))
        support = (
            SupportLevel.SUPPORTED if iteration >= 3 else SupportLevel.PARTIAL
        )
        boost = 0.08 * max(0, iteration - 1)
        kind = "converged" if support is SupportLevel.SUPPORTED else "refine"

        day, hour = self._pick_scene()
        moments = self._fabricate_moments(
            rng, day, hour, support, kind=kind, boost=boost, count=1
        )
        best_cam = moments[0].best_camera().camera_id
        priors = _support_priors(rng)
        predicted = max(_CHOICES, key=lambda key: (priors[key], key))
        answer_text = (
            f"A clearer {moments[0].place_label.lower()} angle shows **{best_cam}** "
            f"at {moments[0].clock_label} — "
            + ("the claim is supported." if support is SupportLevel.SUPPORTED
               else "the angle is sharper but still partial.")
        )
        return ChatTurnResult(
            answer_text=answer_text,
            route="mixed",
            support_priors=priors,
            evidence=self._moments_to_evidence(moments),
            predicted_choice=predicted,
            is_placeholder=True,
            claim=Claim(text=claim, support=support),
            moments=moments,
        )

    # -- internals ----------------------------------------------------------

    def _pick_scene(self) -> Tuple[str, int]:
        """Pick a ``(day, hour)`` where all three trio cameras recorded.

        Prefers the noon hour (so clocks read ``12:xx`` like the design); falls
        back to ``(day1, 8)`` when the roster lacks the full trio (the mirror
        still resolves the trio there).
        """
        scenes: Dict[Tuple[str, int], set] = {}
        for day, camera, hour in self.roster:
            scenes.setdefault((day, hour), set()).add(camera)
        trio = set(_CAMERA_TRIO)
        candidates = [key for key, cams in scenes.items() if trio <= cams]
        if candidates:
            candidates.sort(key=lambda dh: (dh[1] != 12, dh[0], dh[1]))
            return candidates[0]
        return ("day1", 8)

    def _fabricate_moments(
        self,
        rng: random.Random,
        day: str,
        hour: int,
        support: SupportLevel,
        *,
        kind: str,
        boost: float,
        count: Optional[int] = None,
    ) -> List[EvidenceMoment]:
        """Build a deterministic, ranked list of synchronized-camera moments."""
        n = count if count is not None else self.n_evidence
        offset = 0 if kind == "rank" else 2
        moments: List[EvidenceMoment] = []
        for rank in range(n):
            minute = _MOMENT_MINUTES[rank % len(_MOMENT_MINUTES)]
            start = float(minute * 60 + rng.randint(0, 59))
            clock = f"{hour:02d}:{minute:02d}"
            place = _PLACES[(rank + offset) % len(_PLACES)]
            if kind == "rank":
                agg = 0.74 - rank * 0.12 + rng.uniform(-0.02, 0.02) + boost
            elif kind == "refine":
                agg = 0.85 + rng.uniform(-0.02, 0.02) + boost
            else:  # converged
                agg = 0.95 + rng.uniform(-0.01, 0.01) + boost
            agg = round(min(0.99, max(0.40, agg)), 2)
            moments.append(
                EvidenceMoment(
                    moment_id=f"m{rank}",
                    clock_label=clock,
                    place_label=place,
                    camera_count=len(_CAMERA_TRIO),
                    aggregate_score=agg,
                    score_caption=_score_caption(kind, agg),
                    dot_color=_DOT_COLORS[support],
                    cameras=self._camera_angles(rng, day, hour, start, agg),
                )
            )
        return moments

    def _camera_angles(
        self, rng: random.Random, day: str, hour: int, start: float, agg: float
    ) -> List[CameraAngle]:
        """Build the three synchronized camera angles for one moment."""
        best = rng.randrange(len(_CAMERA_TRIO))
        angles: List[CameraAngle] = []
        for index, camera in enumerate(_CAMERA_TRIO):
            score = agg if index == best else round(
                max(0.30, agg - rng.uniform(0.12, 0.30)), 2
            )
            angles.append(
                CameraAngle(
                    camera_id=camera,
                    day=day,
                    hour=hour,
                    start_seconds=start,
                    match_score=score,
                    is_best=index == best,
                )
            )
        return angles

    def _moments_to_evidence(
        self, moments: List[EvidenceMoment]
    ) -> List[EvidenceRef]:
        """Project moments onto the legacy ``EvidenceRef`` contract."""
        evidence: List[EvidenceRef] = []
        for rank, moment in enumerate(moments):
            cam = moment.best_camera()
            evidence.append(
                EvidenceRef(
                    record_id=f"{cam.day}_{cam.camera_id}_{cam.hour:02d}_{rank:04d}",
                    source_type="main_clip",
                    modality="video",
                    day=cam.day,
                    camera_id=cam.camera_id,
                    hour=cam.hour,
                    start_seconds=cam.start_seconds,
                    end_seconds=cam.start_seconds + 30.0,
                    score=moment.aggregate_score,
                    text=(
                        f"{cam.camera_id} at {moment.clock_label} "
                        f"({moment.place_label})"
                    ),
                )
            )
        return evidence


def _seed(text: str) -> int:
    """Return a stable integer seed for ``text`` (case/space-insensitive)."""
    return int(hashlib.sha1(text.strip().lower().encode()).hexdigest(), 16)


def _score_caption(kind: str, agg: float) -> str:
    """Return the mono caption shown beside a moment's camera count."""
    if kind == "rank":
        return f"match {agg:.2f}"
    if kind == "refine":
        return f"newest evidence · match {agg:.2f}"
    return f"converged {agg:.2f}"


def _support_priors(rng: random.Random) -> Dict[str, float]:
    """Return normalized per-choice support priors that sum to 1.0."""
    raw = {key: rng.random() + 0.05 for key in _CHOICES}
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


# Verdicts that mark a camera angle as a weak / unconvincing perspective.
_WEAK_VERDICTS = {"flagged", "rejected"}


def compose_refined_query(claim: str, reviews: Dict[str, Dict[str, str]]) -> str:
    """Build an editable refined query from the reviewer's per-camera feedback.

    ``reviews`` maps ``camera_id -> {"state": ..., "justification": ...}``.  The
    query restates the claim and folds in each camera's justification, calling
    out the angles the reviewer flagged or rejected as the ones to strengthen.
    Deterministic and model-free so it works offline; a live engine may later
    replace it with an LLM-composed query.
    """
    claim = (claim or "the claim").strip().rstrip(".")
    notes = []
    weak = []
    for camera_id, info in reviews.items():
        justification = (info.get("justification") or "").strip()
        if justification:
            notes.append(f"{camera_id}: {justification}")
        if info.get("state") in _WEAK_VERDICTS:
            weak.append(camera_id)
    query = f"Re-examine whether {claim}."
    if notes:
        query += " Reviewer notes — " + "; ".join(notes) + "."
    if weak:
        query += " Find a clearer angle for " + ", ".join(weak) + "."
    return query
