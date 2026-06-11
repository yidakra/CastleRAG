"""Question router: structured hint extraction and route assignment."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from castlerag.schemas import QuestionRoute

_PARTICIPANTS = (
    "Allie",
    "Bjorn",
    "Celine",
    "Deon",
    "Estella",
    "Finn",
    "Greta",
    "Harvey",
    "Isla",
    "Jian",
)
_ROOM_PATTERNS = {
    "kitchen": "Kitchen",
    "living room": "Living1",
    "office": "Office",
    "hallway": "Hallway",
    "living1": "Living1",
    "living2": "Living2",
}
_DAY_PATTERNS = (
    (re.compile(r"\bday\s*([1-4])\b"), "digit"),
    (re.compile(r"\b(first|second|third|fourth)\s+day\b"), "ordinal"),
)
_DAY_ORDINALS = {
    "first": "day1",
    "second": "day2",
    "third": "day3",
    "fourth": "day4",
}
_TEMPORAL_KEYWORDS = frozenset(
    [
        "before",
        "after",
        "while",
        "during",
        "then",
        "when",
        "next",
        "previously",
        "later",
        "first",
        "last",
        "finally",
        "once",
    ]
)
_TEMPORAL_PHRASES = (
    "what happened before",
    "what happened after",
    "what was happening when",
    "in what order",
    "at the time",
    "by the time",
    "right before",
    "right after",
)
_TEMPORAL_DOMINANT_MARKERS = (
    "before",
    "after",
    "next",
    "previously",
    "later",
    "first",
    "last",
    "finally",
    "once",
    "in what order",
    "right before",
    "right after",
)
_SPEECH_KEYWORDS = frozenset(
    [
        "say",
        "said",
        "tell",
        "told",
        "ask",
        "asked",
        "speak",
        "spoken",
        "conversation",
        "transcript",
        "announce",
        "called",
        "call",
        "word",
        "words",
        "hear",
        "heard",
    ]
)
_SPEECH_PHRASES = (
    "what did",
    "what was said",
    "what did they say",
    "what did she say",
    "what did he say",
    "who said",
    "which words",
    "what was heard",
    "what did allie say",
)
_VISUAL_KEYWORDS = frozenset(
    [
        "wearing",
        "visible",
        "look",
        "see",
        "shown",
        "screen",
        "text",
        "logo",
        "object",
        "holding",
        "brand",
        "count",
        "color",
        "colour",
        "where",
        "which room",
        "what is on",
        "photo",
        "thermal",
    ]
)
_VISUAL_PHRASES = (
    "what color",
    "what colour",
    "what is on",
    "which room",
    "where is",
    "how many",
    "what does",
    "what was visible",
    "what can be seen",
)


@dataclass(frozen=True)
class RouteEvidenceProfile:
    """Route-scoped retrieval budget and modality-priority profile."""

    transcript_budget: int
    candidate_video_budget: int
    frames_per_candidate_video: int
    auxiliary_image_budget: int
    max_evidence_rows: int
    source_priority: Tuple[str, ...]


_ROUTE_PROFILES: Dict[QuestionRoute, RouteEvidenceProfile] = {
    "static_visual": RouteEvidenceProfile(
        transcript_budget=10,
        candidate_video_budget=4,
        frames_per_candidate_video=32,
        auxiliary_image_budget=16,
        max_evidence_rows=50,
        source_priority=(
            "main_clip",
            "main_event_summary",
            "aux_photo",
            "aux_thermal",
            "aux_video",
            "transcript_window",
            "aux_gaze",
            "aux_heartrate",
        ),
    ),
    "speech_text": RouteEvidenceProfile(
        transcript_budget=30,
        candidate_video_budget=4,
        frames_per_candidate_video=32,
        auxiliary_image_budget=16,
        max_evidence_rows=50,
        source_priority=(
            "transcript_window",
            "main_event_summary",
            "main_clip",
            "aux_video",
            "aux_photo",
            "aux_gaze",
            "aux_heartrate",
            "aux_thermal",
        ),
    ),
    "temporal": RouteEvidenceProfile(
        transcript_budget=30,
        candidate_video_budget=4,
        frames_per_candidate_video=32,
        auxiliary_image_budget=16,
        max_evidence_rows=50,
        source_priority=(
            "transcript_window",
            "main_event_summary",
            "main_clip",
            "aux_video",
            "aux_photo",
            "aux_gaze",
            "aux_heartrate",
            "aux_thermal",
        ),
    ),
    "mixed": RouteEvidenceProfile(
        transcript_budget=30,
        candidate_video_budget=4,
        frames_per_candidate_video=32,
        auxiliary_image_budget=16,
        max_evidence_rows=50,
        source_priority=(
            "transcript_window",
            "main_clip",
            "main_event_summary",
            "aux_photo",
            "aux_video",
            "aux_thermal",
            "aux_gaze",
            "aux_heartrate",
        ),
    ),
}


@dataclass
class RouteHints:
    route: QuestionRoute
    day: Optional[str] = None
    participant: Optional[str] = None
    room: Optional[str] = None
    has_visual_cue: bool = False
    has_speech_cue: bool = False
    has_temporal_cue: bool = False
    extracted_keywords: List[str] = field(default_factory=list)
    evidence_profile: Optional[RouteEvidenceProfile] = None

    def __post_init__(self) -> None:
        """Fill evidence_profile from the route default when not provided."""
        if self.evidence_profile is None:
            self.evidence_profile = _profile_for_route(self.route)


def route_question(
    question: str,
    choices: dict[str, str],
) -> RouteHints:
    """Assign one route and extract reusable retrieval hints."""
    question_lower = question.lower()
    tokens = set(re.findall(r"\b\w+\b", question_lower))

    day = _extract_day(question_lower)
    participant_matches = [
        (m.start(), name)
        for name in _PARTICIPANTS
        if (m := re.search(rf"\b{re.escape(name.lower())}\b", question_lower))
    ]
    participant = min(participant_matches, default=(None, None))[1]

    room_matches = [
        (m.start(), normalized)
        for phrase, normalized in _ROOM_PATTERNS.items()
        if (m := re.search(rf"\b{re.escape(phrase)}\b", question_lower))
    ]
    room = min(room_matches, default=(None, None))[1]

    temporal_score, temporal_hits = _cue_score(
        question_lower,
        tokens,
        keywords=_TEMPORAL_KEYWORDS,
        phrases=_TEMPORAL_PHRASES,
    )
    speech_score, speech_hits = _cue_score(
        question_lower,
        tokens,
        keywords=_SPEECH_KEYWORDS,
        phrases=_SPEECH_PHRASES,
    )
    visual_score, visual_hits = _cue_score(
        question_lower,
        tokens,
        keywords=_VISUAL_KEYWORDS,
        phrases=_VISUAL_PHRASES,
    )
    if room is not None:
        visual_score += 1
        visual_hits.append(room.lower())

    has_temporal_cue = temporal_score > 0
    has_speech_cue = speech_score > 0
    has_visual_cue = visual_score > 0

    route = _choose_route(
        temporal_score=temporal_score,
        speech_score=speech_score,
        visual_score=visual_score,
        question=question_lower,
    )
    extracted_keywords = sorted(
        {
            *temporal_hits,
            *speech_hits,
            *visual_hits,
        }
    )
    return RouteHints(
        route=route,
        day=day,
        participant=participant,
        room=room,
        has_visual_cue=has_visual_cue,
        has_speech_cue=has_speech_cue,
        has_temporal_cue=has_temporal_cue,
        extracted_keywords=extracted_keywords,
        evidence_profile=_profile_for_route(route),
    )


def _profile_for_route(route: QuestionRoute) -> RouteEvidenceProfile:
    """Return a fresh RouteEvidenceProfile copy for the given route."""
    profile = _ROUTE_PROFILES[route]
    return RouteEvidenceProfile(
        transcript_budget=profile.transcript_budget,
        candidate_video_budget=profile.candidate_video_budget,
        frames_per_candidate_video=profile.frames_per_candidate_video,
        auxiliary_image_budget=profile.auxiliary_image_budget,
        max_evidence_rows=profile.max_evidence_rows,
        source_priority=tuple(profile.source_priority),
    )


def _extract_day(text: str) -> Optional[str]:
    """Return a normalised day tag (e.g. 'day1') extracted from text, or None."""
    for pattern, kind in _DAY_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        value = match.group(1)
        if kind == "digit":
            return f"day{value}"
        return _DAY_ORDINALS[value]
    return None


def _cue_score(
    text: str,
    tokens: Iterable[str],
    *,
    keywords: Iterable[str],
    phrases: Iterable[str],
) -> tuple[int, List[str]]:
    """Return a cue score and the list of matched keywords and phrases."""
    hits: List[str] = []
    score = 0
    token_set = set(tokens)
    for keyword in keywords:
        if keyword in token_set:
            score += 1
            hits.append(keyword)
    for phrase in phrases:
        if phrase in text:
            score += 2
            hits.append(phrase)
    return score, hits


def _choose_route(
    *,
    temporal_score: int,
    speech_score: int,
    visual_score: int,
    question: str,
) -> QuestionRoute:
    """Return the best-matching route from temporal, speech, and visual cue scores."""
    if _has_temporal_anchor(question) or temporal_score >= 3:
        return "temporal"
    if speech_score > 0 and visual_score > 0:
        return "mixed"
    if speech_score > visual_score and speech_score > 0:
        return "speech_text"
    if visual_score > speech_score and visual_score > 0:
        return "static_visual"
    if speech_score > 0 and visual_score > 0:
        return "mixed"
    if speech_score > 0:
        return "speech_text"
    return "static_visual"


def _has_temporal_anchor(question: str) -> bool:
    """Return True if the question contains any dominant temporal ordering marker."""
    return any(marker in question for marker in _TEMPORAL_DOMINANT_MARKERS)
