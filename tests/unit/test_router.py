"""Tests for structured question routing."""

from __future__ import annotations

from castlerag.routing.question_router import RouteHints, route_question


def test_route_question_temporal_extracts_structured_hints():
    hints = route_question(
        question="On the first day, what did Allie say before entering the kitchen?",
        choices={"a": "hello", "b": "bye", "c": "thanks", "d": "nothing"},
    )
    assert hints.route == "temporal"
    assert hints.day == "day1"
    assert hints.participant == "Allie"
    assert hints.room == "Kitchen"
    assert hints.has_speech_cue is True
    assert hints.has_temporal_cue is True
    assert hints.evidence_profile.transcript_budget == 30
    assert "before" in hints.extracted_keywords


def test_route_question_speech_text_prefers_lexical_evidence():
    hints = route_question(
        question="What did Bjorn say to Celine during the call?",
        choices={
            "a": "He was leaving",
            "b": "He was hungry",
            "c": "He needed help",
            "d": "He was tired",
        },
    )
    assert hints.route == "speech_text"
    assert hints.participant == "Bjorn"
    assert hints.has_speech_cue is True
    assert hints.has_visual_cue is False
    assert hints.evidence_profile.source_priority[0] == "transcript_window"


def test_route_question_static_visual_prefers_visual_sources():
    hints = route_question(
        question="What color shirt was Greta wearing in the hallway photo?",
        choices={"a": "Blue", "b": "Black", "c": "White", "d": "Red"},
    )
    assert hints.route == "static_visual"
    assert hints.participant == "Greta"
    assert hints.room == "Hallway"
    assert hints.has_visual_cue is True
    assert hints.has_speech_cue is False
    assert hints.evidence_profile.source_priority[0] == "main_clip"
    assert hints.evidence_profile.frames_per_candidate_video == 32


def test_route_question_mixed_combines_visual_and_speech_cues():
    hints = route_question(
        question=(
            "Which room was visible on screen when Jian said the password out loud?"
        ),
        choices={
            "a": "Kitchen",
            "b": "Office",
            "c": "Living room",
            "d": "Hallway",
        },
    )
    assert hints.route == "mixed"
    assert hints.participant == "Jian"
    assert hints.has_visual_cue is True
    assert hints.has_speech_cue is True
    assert hints.has_temporal_cue is True
    assert "screen" in hints.extracted_keywords
    assert "said" in hints.extracted_keywords


def test_route_hints_default_profile_matches_route_and_is_not_shared():
    speech_hints = RouteHints(route="speech_text")
    visual_hints = RouteHints(route="static_visual")
    assert speech_hints.evidence_profile is not None
    assert visual_hints.evidence_profile is not None
    assert speech_hints.evidence_profile.transcript_budget == 30
    assert visual_hints.evidence_profile.transcript_budget == 10
    assert speech_hints.evidence_profile.frames_per_candidate_video == 32
    assert visual_hints.evidence_profile.frames_per_candidate_video == 32
    assert speech_hints.evidence_profile is not visual_hints.evidence_profile


def test_route_question_does_not_leak_filter_hints_from_answer_options():
    hints = route_question(
        question="What did the person say after breakfast?",
        choices={
            "a": "Allie said hello in the kitchen",
            "b": "Bjorn waved from the office",
            "c": "Celine entered the hallway",
            "d": "Deon looked at the screen",
        },
    )
    assert hints.day is None
    assert hints.participant is None
    assert hints.room is None
