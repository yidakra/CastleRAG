"""Tests for structured question routing."""

from __future__ import annotations

import pytest

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
        question="What did Bjorn say to Cathal during the call?",
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
        question="What color shirt was Florian wearing in the reading room photo?",
        choices={"a": "Blue", "b": "Black", "c": "White", "d": "Red"},
    )
    assert hints.route == "static_visual"
    assert hints.participant == "Florian"
    assert hints.room == "Reading"
    assert hints.has_visual_cue is True
    assert hints.has_speech_cue is False
    assert hints.evidence_profile.source_priority[0] == "main_clip"
    assert hints.evidence_profile.frames_per_candidate_video == 32


def test_route_question_mixed_combines_visual_and_speech_cues():
    hints = route_question(
        question=(
            "Which room was visible on screen when Werner said the password out loud?"
        ),
        choices={
            "a": "Kitchen",
            "b": "Meeting room",
            "c": "Living room",
            "d": "Reading room",
        },
    )
    assert hints.route == "mixed"
    assert hints.participant == "Werner"
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


@pytest.mark.parametrize(
    "question",
    [
        "What is on the back of Werner's t-shirt on the first day?",
        "What colour was the rim of Werner's plate during breakfast on the first day?",
        "What brand is the fridge in the kitchen on day 1?",
    ],
)
def test_day_ordinal_is_a_day_hint_not_a_temporal_marker(question):
    """'on the first day' / 'day 1' set the day but must not force temporal."""
    hints = route_question(question, {})
    assert hints.route == "static_visual"
    assert hints.day == "day1"
    assert "first" not in hints.extracted_keywords


def test_day_ordinal_alone_sets_no_temporal_cue():
    hints = route_question(
        "What is on the back of Werner's t-shirt on the first day?", {}
    )
    assert hints.has_temporal_cue is False
    assert hints.has_visual_cue is True


@pytest.mark.parametrize(
    ("question", "day"),
    [
        ("Who was the second person to present slides at the workshop on the first day?", "day1"),  # noqa: E501
        ("Who gave the first presentation for the workshop on the first day?", "day1"),
        ("Who dealt first in the first game of poker?", None),
        ("What time did Allie and Linh plan to leave on Saturday at first?", None),
        ("What was the first category in the happy quiz?", None),
        ("On the first day, what did Allie say before entering the kitchen?", "day1"),
        ("What did Bjorn do right after breakfast on day 2?", "day2"),
    ],
)
def test_genuinely_temporal_questions_still_route_temporal(question, day):
    """Ordering markers in their temporal sense keep the temporal route."""
    hints = route_question(question, {})
    assert hints.route == "temporal"
    assert hints.day == day


def test_temporal_markers_match_whole_words_only():
    """'afternoon' / 'lasted' no longer anchor the temporal route by substring."""
    hints = route_question("What was for lunch in the afternoon?", {})
    assert hints.route != "temporal"
    hints = route_question("How long the meeting lasted?", {})
    assert hints.route != "temporal"


def test_route_question_does_not_leak_filter_hints_from_answer_options():
    hints = route_question(
        question="What did the person say after breakfast?",
        choices={
            "a": "Allie said hello in the kitchen",
            "b": "Bjorn waved from the meeting room",
            "c": "Cathal entered the reading room",
            "d": "Werner looked at the screen",
        },
    )
    assert hints.day is None
    assert hints.participant is None
    assert hints.room is None


@pytest.mark.parametrize(
    "question",
    [
        "What colour is the car in the second row?",
        "What is in the third drawer of the kitchen cabinet?",
        "Which book is second from the left on the shelf?",
    ],
)
def test_positional_ordinals_do_not_anchor_temporal(question):
    """Bare 'second'/'third' are usually positional, not ordering markers."""
    assert route_question(question, {}).route != "temporal"


@pytest.mark.parametrize(
    "question",
    [
        "What changed in the kitchen from the first day to the second day?",
        "Did Allie wear the same shirt on day 1 and day 2?",
        "Was the whiteboard fuller on the third day than on the first day?",
    ],
)
def test_cross_day_comparison_routes_temporal_without_day_pin(question):
    """Two day references are a temporal cue and must not pin one day."""
    hints = route_question(question, {})
    assert hints.route == "temporal"
    assert hints.has_temporal_cue is True
    assert hints.day is None


def test_single_day_reference_still_sets_day_hint():
    hints = route_question("What is on the whiteboard on day 2?", {})
    assert hints.day == "day2"
    assert hints.route != "temporal"


@pytest.mark.parametrize(
    "question",
    [
        "What is in the cupboard next to the fridge?",
        "What is in the first drawer under the sink?",
        "What are the last two items on the shopping list?",
        "What is Werner's first name?",
    ],
)
def test_spatial_next_to_and_positional_first_last_do_not_anchor(question):
    assert route_question(question, {}).route != "temporal"


@pytest.mark.parametrize(
    "question",
    ["What did Bjorn do next?", "Who dealt first in the first game of poker?"],
)
def test_ordering_next_and_first_still_anchor(question):
    assert route_question(question, {}).route == "temporal"


@pytest.mark.parametrize(
    "question",
    [
        "What was on the whiteboard on the last day?",
        "How did the kitchen look on the first two days?",
        "Which board game was played on the final day?",
    ],
)
def test_last_day_and_day_spans_are_day_phrases_not_anchors(question):
    hints = route_question(question, {})
    assert hints.route != "temporal"
    assert hints.day is None


def test_first_to_last_day_is_a_comparison():
    hints = route_question(
        "What changed on the whiteboard from the first day to the last day?", {}
    )
    assert hints.route == "temporal"
    assert hints.day is None


@pytest.mark.parametrize(
    "question",
    [
        "What changed between day1 and day2?",
        "Was the same lamp on the desk on day1 and on the second day?",
    ],
)
def test_no_space_day_form_counts_in_day_comparison(question):
    hints = route_question(question, {})
    assert hints.route == "temporal"
    assert hints.day is None


@pytest.mark.parametrize(
    "question",
    ["Where is the first-aid kit?", "Who made a last-minute change to the slides?"],
)
def test_hyphenated_first_last_compounds_do_not_anchor(question):
    assert route_question(question, {}).route != "temporal"


@pytest.mark.parametrize(
    "question",
    [
        "What was on the whiteboard on the second to last day?",
        "Which game was played on the second-to-last day?",
    ],
)
def test_second_to_last_day_is_a_day_phrase_not_an_anchor(question):
    hints = route_question(question, {})
    assert hints.route != "temporal"
    assert hints.day is None


def test_second_to_last_day_versus_last_day_is_a_comparison():
    hints = route_question(
        "Was the kitchen tidier on the second to last day than on the last day?", {}
    )
    assert hints.route == "temporal"


@pytest.mark.parametrize(
    "question",
    [
        "How did the score change from the first to the second day?",
        "Was the same lamp on the desk on the first and the last day?",
        "How did the kitchen change between the first two days and the last two days?",
    ],
)
def test_elliptical_and_two_span_day_comparisons_route_temporal(question):
    hints = route_question(question, {})
    assert hints.route == "temporal"
    assert hints.day is None


def test_single_span_is_not_a_comparison():
    hints = route_question("How did the kitchen look on the first two days?", {})
    assert hints.route != "temporal"


def test_appositive_final_day_is_a_single_day():
    hints = route_question("What was served at dinner on the fourth and final day?", {})
    assert hints.route != "temporal"
    assert hints.day == "day4"


def test_second_to_the_last_day_is_a_single_day_phrase():
    hints = route_question("Which game was played on the second to the last day?", {})
    assert hints.route != "temporal"


def test_from_the_first_to_the_last_day_is_a_comparison():
    hints = route_question(
        "How did the whiteboard change from the first to the last day?", {}
    )
    assert hints.route == "temporal"
    assert hints.day is None
