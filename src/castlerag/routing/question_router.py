"""Question router: extract hints and assign exactly one route.

Routes (mandatory — SPEC §4.2):
  static_visual  — questions about objects, frames, visible content
  speech_text    — questions answered by transcript / spoken content
  temporal       — questions about order, before/after/while relations
  mixed          — requires both visual and speech evidence

Routing is mandatory; a single prompt strategy is insufficient for CASTLE
(validated by WDL and MARS — see SPEC §4.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from castlerag.schemas import QuestionRoute


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
    """Assign a route and extract metadata hints from the question text.

    This is a lightweight heuristic pass — no LLM call.  It looks for:
      - day / participant / room references
      - visual/OCR keyword patterns
      - speech/transcript patterns
      - temporal keywords (before, after, while, then, when, ...)
    """
    raise NotImplementedError("Implemented in issue #7")
