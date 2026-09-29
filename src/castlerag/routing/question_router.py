"""Question router: structured hint extraction and route assignment."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from castlerag.schemas import QuestionRoute

_PARTICIPANTS = (
    "Allie",
    "Bao",
    "Bjorn",
    "Cathal",
    "Florian",
    "Klaus",
    "Luca",
    "Onanong",
    "Stevan",
    "Tien",
    "Werner",
)
_ROOM_PATTERNS = {
    "kitchen": "Kitchen",
    "living room": "Living1",
    "living1": "Living1",
    "living2": "Living2",
    "meeting room": "Meeting",
    "reading room": "Reading",
    "reading area": "Reading",
}
_DAY_PATTERNS = (
    (re.compile(r"\bday\s*([1-4])\b"), "digit"),
    # "the third and final day" / "the fourth and last day": one glossed day.
    (
        re.compile(
            r"\b(?:(first|second|third|fourth)\s+and\s+final"
            r"|(fourth)\s+and\s+(?:the\s+)?(?:last|final))\s+day\b"
        ),
        "ordinal",
    ),
    (re.compile(r"\b(first|second|third|fourth)\s+day\b"), "ordinal"),
)
_DAY_ORDINALS = {
    "first": "day1",
    "second": "day2",
    "third": "day3",
    "fourth": "day4",
}
# Day references ("on the first day", "day 1") are a DAY hint only. They are
# blanked out before temporal cue matching so the bare ordinal does not read
# as a temporal-ordering marker: "what is on the back of Werner's t-shirt on
# the first day" is a static visual question, not a before/after one.
_ORD = r"(?:first|second|third|fourth|last|final)"
_DAY_PHRASE_RE = re.compile(
    # Compound forms first, so the bare alternatives can't leave a stray
    # ordinal behind that reads as an ordering marker: "second to last day",
    # and the elliptical pair "from the first to the second day" / "the first
    # and the last day", where only the second ordinal carries the noun.
    r"\b(?:(?:second|third)[\s-]+to[\s-]+(?:the\s+)?last\s+days?"
    rf"|{_ORD}\s+(?:to|and|or|versus|vs\.?|until|through)\s+(?:the\s+)?{_ORD}\s+days?"
    rf"|{_ORD}\s+(?:(?:two|three|four|\d)\s+)?days?|day\s*[1-4])\b"
)
_TEMPORAL_KEYWORDS = frozenset(
    [
        "second",
        "third",
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
# Markers that alone send a question to the temporal route. Matched as whole
# words (so "after" no longer fires on "afternoon") on the day-stripped text,
# so "first"/"last" only count in their ordering sense ("who dealt first",
# "at first", "the first time"), not as part of "the first day". Bare
# "second"/"third" are NOT anchors: they are usually positional ("the second
# row", "the third drawer"); only their ordering phrases below anchor, and the
# bare words just add to the multi-cue temporal score via _TEMPORAL_KEYWORDS.
_TEMPORAL_DOMINANT_MARKERS = (
    "before",
    "after",
    "previously",
    "later",
    "second person to",
    "second one to",
    "second time",
    "third person to",
    "third one to",
    "third time",
    "finally",
    "once",
    "in what order",
    "right before",
    "right after",
)
# "first"/"last" anchor only in their ordering sense. In CASTLE questions that
# is the common case ("who dealt first", "the first category in the quiz",
# "the first person to unfold the mat"), so the bare words stay anchors, but
# a positional or quantity noun right after them ("the first drawer", "the
# last two", "first name") disqualifies the match. "next" is spatial in
# "next to" and an ordering marker otherwise ("what did she do next").
_POSITIONAL_FOLLOWERS = (
    r"(?!-)"  # hyphenated compounds: first-aid, last-minute, first-person
    r"(?!\s+(?:two|three|four|five|few|\d+|rows?|drawers?|shelf|shelves|"
    r"cupboards?|cabinets?|floors?|pages?|columns?|seats?|doors?|aisles?|"
    r"names?|letters?|words?|digits?|numbers?|items?|slots?|positions?)\b)"
)
_TEMPORAL_DOMINANT_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(m) for m in _TEMPORAL_DOMINANT_MARKERS)
    + r"|next(?!\s+to\b)"
    # "second to arrive" is ordering; "second to last drawer" is positional.
    + r"|(?:second|third)\s+to(?!\s+(?:the\s+)?last\b)"
    + r"|(?:first|last)" + _POSITIONAL_FOLLOWERS
    + r")\b"
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
    llm_key_entities: List[str] = field(default_factory=list)
    llm_focus_modalities: List[str] = field(default_factory=list)
    evidence_profile: Optional[RouteEvidenceProfile] = None
    # Camera ids the UI reviewer rejected; hard-excluded from dense retrieval on
    # subsequent refine iterations (must_not). Empty on the eval path.
    exclude_cameras: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Fill evidence_profile from the route default when not provided."""
        if self.evidence_profile is None:
            self.evidence_profile = _profile_for_route(self.route)


def route_question(
    question: str,
    choices: dict[str, str],
    vllm_base_url: Optional[str] = None,
    model_name: Optional[str] = None,
) -> RouteHints:
    """Assign one route and extract reusable retrieval hints."""
    question_lower = question.lower()
    tokens = set(re.findall(r"\b\w+\b", question_lower))

    day = _extract_day(question_lower)
    day_comparison = _has_day_comparison(question_lower)
    if day_comparison:
        # A cross-day question must not be pinned to the first day mentioned:
        # the dense lanes hard-filter on the day hint.
        day = None
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

    # Temporal cues are scored on the text with day references blanked out:
    # "the first day" is a day hint (extracted above), not an ordering marker.
    temporal_text = _strip_day_phrases(question_lower)
    temporal_score, temporal_hits = _cue_score(
        temporal_text,
        set(re.findall(r"\b\w+\b", temporal_text)),
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

    has_temporal_cue = temporal_score > 0 or day_comparison
    has_speech_cue = speech_score > 0
    has_visual_cue = visual_score > 0

    route = _choose_route(
        day_comparison=day_comparison,
        temporal_score=temporal_score,
        speech_score=speech_score,
        visual_score=visual_score,
        question=temporal_text,
    )
    extracted_keywords = sorted(
        {
            *temporal_hits,
            *speech_hits,
            *visual_hits,
        }
    )
    llm_entities: List[str] = []
    llm_modalities: List[str] = []
    if vllm_base_url and model_name:
        llm_entities, llm_modalities = _llm_route_hints(
            question, choices, vllm_base_url, model_name
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
        llm_key_entities=llm_entities,
        llm_focus_modalities=llm_modalities,
        evidence_profile=_profile_for_route(route),
    )


def _llm_route_hints(
    question: str,
    choices: dict[str, str],
    vllm_base_url: str,
    model_name: str,
) -> tuple[List[str], List[str]]:
    """Call VLM to extract key entities and focus modalities for retrieval."""
    import json

    prompt = (
        "You are helping a video retrieval system. Given this multiple-choice question "
        "about a multi-camera home video dataset, extract:\n"
        "1. key_entities: main objects, people, foods, or activities to search for "
        "(3-6 short noun phrases)\n"
        "2. focus_modalities: which evidence types to prioritise — choose from "
        "[\"caption\", \"transcript\", \"ocr\", \"scene_graph\"]\n\n"
        f"Question: {question}\n"
        f"Choices: A {choices.get('a', '')}. B {choices.get('b', '')}. "
        f"C {choices.get('c', '')}. D {choices.get('d', '')}.\n\n"
        "Return JSON only: "
        "{\"key_entities\": [...], \"focus_modalities\": [...]}"
    )
    try:
        from openai import OpenAI

        client = OpenAI(base_url=vllm_base_url, api_key="not-needed", timeout=30.0)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip().strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
        data = json.loads(text)
        entities = [str(e) for e in data.get("key_entities", [])[:6]]
        modalities = [str(m) for m in data.get("focus_modalities", [])[:4]]
        return entities, modalities
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug("LLM route hints extraction failed: %s", exc)
        return [], []


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
        value = next(g for g in match.groups() if g)
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
    day_comparison: bool = False,
) -> QuestionRoute:
    """Return the best-matching route from temporal, speech, and visual cue scores."""
    if day_comparison or _has_temporal_anchor(question) or temporal_score >= 3:
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


def _strip_day_phrases(text: str) -> str:
    """Blank out day references ("first day", "day 1") so they carry no temporal cue."""
    return _DAY_PHRASE_RE.sub(" ", text)


def _has_day_comparison(text: str) -> bool:
    """Return True when the question references two or more distinct days.

    "What changed from the first day to the second day?" is a temporal
    question even though every ordering word in it belongs to a day phrase,
    so this is checked on the raw text, before :func:`_strip_day_phrases`.
    """
    seen = set()
    for match in _DAY_PHRASE_RE.finditer(text):
        token = match.group(0)
        words = token.split()
        # "between the first and final day" / "from the first and last day":
        # a comparative lead-in overrides the single-day gloss reading.
        lead_in = text[max(0, match.start() - 24) : match.start()].split()[-3:]
        comparative = bool(_COMPARATIVE_LEAD_INS.intersection(lead_in))
        digit = re.fullmatch(r"day\s*([1-4])", token)
        if digit:  # "day 1" and the no-space "day1" alike
            seen.add(f"day{digit.group(1)}")
        elif re.fullmatch(
            r"(?:second|third)[\s-]+to[\s-]+(?:the\s+)?last\s+days?", token
        ):
            seen.add(f"{words[0].rstrip('-')}-to-last")
        elif len(words) >= 4:
            first, second = words[0], words[-2]
            if _is_appositive_final_day(words) and not comparative:
                # "the fourth and final day": one day glossed twice, not two.
                seen.add(_day_bucket(first))
            else:
                # Elliptical pair, "first to the second day": both count.
                seen.update(_day_bucket(w) for w in (first, second))
        elif len(words) == 3:
            # "first two days": a span. Keyed by ordinal and length so "the
            # first two days" vs "the last two days" stay distinct.
            seen.add(f"{_day_bucket(words[0])}-span-{words[1]}")
        else:
            seen.add(_day_bucket(words[0]))
    return len(seen) >= 2


_COMPARATIVE_LEAD_INS = frozenset(
    ["between", "from", "compare", "compared", "comparing", "versus", "vs"]
)


def _is_appositive_final_day(words: List[str]) -> bool:
    """True for "<ordinal> and final day" / "fourth and last day" glosses.

    "first and last day" is a pair of days, so only the word "final" (a gloss,
    never a day reference on its own here) or the collection's actual last
    ordinal ("fourth") makes the phrase a single day.
    """
    if len(words) == 5 and words[2] == "the":
        gloss, article = words[3], True
    elif len(words) == 4:
        gloss, article = words[2], False
    else:
        return False
    if words[1] != "and" or gloss not in ("last", "final"):
        return False
    if words[0] == "fourth":
        return True  # "the fourth and (the) final/last day": already the last day
    return gloss == "final" and not article


def _day_bucket(ordinal: str) -> str:
    """Map an ordinal word to its comparison bucket ("day1", ..., "last")."""
    return _DAY_ORDINALS.get(ordinal, "last")  # "last" / "final"


def _has_temporal_anchor(question: str) -> bool:
    """Return True if the question contains a dominant temporal ordering marker.

    Markers match as whole words. Callers pass the day-stripped text (see
    :func:`_strip_day_phrases`) so "on the first day" alone never anchors.
    """
    return _TEMPORAL_DOMINANT_RE.search(question) is not None
