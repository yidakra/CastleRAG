"""Question router: extract hints and assign exactly one route."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

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


def route_question(
    question: str,
    choices: dict[str, str],
) -> RouteHints:
    """Assign a route and extract metadata hints from the question text."""
    text = f"{question} {' '.join(choices.values())}".lower()
    tokens = set(re.findall(r"\b\w+\b", text))

    day = None
    day_match = re.search(r"\bday\s*([1-4])\b", text)
    if day_match:
        day = f"day{day_match.group(1)}"

    participant = next(
        (name for name in _PARTICIPANTS if name.lower() in text),
        None,
    )
    room = next(
        (normalized for phrase, normalized in _ROOM_PATTERNS.items() if phrase in text),
        None,
    )

    has_temporal_cue = bool(tokens.intersection(_TEMPORAL_KEYWORDS))
    has_speech_cue = bool(tokens.intersection(_SPEECH_KEYWORDS))
    has_visual_cue = bool(tokens.intersection(_VISUAL_KEYWORDS)) or room is not None

    if has_temporal_cue:
        route: QuestionRoute = "temporal"
    elif has_visual_cue and has_speech_cue:
        route = "mixed"
    elif has_speech_cue:
        route = "speech_text"
    else:
        route = "static_visual"

    extracted_keywords = sorted(
        tokens.intersection(_TEMPORAL_KEYWORDS | _SPEECH_KEYWORDS | _VISUAL_KEYWORDS)
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
    )
