"""Tests for src/castlerag/schemas.py"""

import pytest

from castlerag.schemas import (
    AuxRecord,
    ClipRecord,
    EvalQuestion,
    EventSummaryRecord,
    Prediction,
    QdrantPoint,
    RerankerOutput,
    TranscriptSegment,
    TranscriptWindow,
)

# ---------------------------------------------------------------------------
# ClipRecord
# ---------------------------------------------------------------------------


def test_clip_record_defaults():
    rec = ClipRecord(
        clip_id="clip_day1_Allie_08_000",
        parent_source_id="day1_Allie_08",
        day="day1",
        hour=8,
        camera_id="Allie",
        camera_type="ego",
        participant_id="Allie",
        start_seconds=0.0,
        end_seconds=30.0,
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_230_000,
        source_video_path="/data/main/day1/Allie/video/08.mp4",
    )
    assert rec.source_type == "main_clip"
    assert rec.modality == "video"
    assert rec.is_placeholder is False
    assert rec.has_speech is False
    assert rec.sampled_frame_paths == []


def test_clip_record_rejects_end_le_start():
    with pytest.raises(Exception):
        ClipRecord(
            clip_id="bad",
            parent_source_id="x",
            day="day1",
            hour=8,
            camera_id="Allie",
            camera_type="ego",
            start_seconds=30.0,
            end_seconds=0.0,
            absolute_start=1_672_531_230_000,
            absolute_end=1_672_531_200_000,  # end < start
            source_video_path="/data/08.mp4",
        )


def test_clip_record_equal_timestamps_rejected():
    with pytest.raises(Exception):
        ClipRecord(
            clip_id="bad",
            parent_source_id="x",
            day="day1",
            hour=8,
            camera_id="Allie",
            camera_type="ego",
            start_seconds=0.0,
            end_seconds=30.0,
            absolute_start=1_000,
            absolute_end=1_000,  # equal, not strictly greater
            source_video_path="/data/08.mp4",
        )


# ---------------------------------------------------------------------------
# TranscriptWindow
# ---------------------------------------------------------------------------


def test_transcript_window_valid():
    tw = TranscriptWindow(
        transcript_window_id="tw_0",
        day="day1",
        camera_id="Allie",
        camera_type="ego",
        participant_id="Allie",
        room=None,
        hour=8,
        transcript_text="Hello world",
        has_speech=True,
        transcript_char_len=11,
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_215_000,
    )
    assert tw.has_speech is True
    assert tw.transcript_segments == []


def test_transcript_window_rejects_bad_timestamps():
    with pytest.raises(Exception):
        TranscriptWindow(
            transcript_window_id="tw_bad",
            day="day1",
            camera_id="Allie",
            camera_type="ego",
            participant_id="Allie",
            room=None,
            hour=8,
            transcript_text="oops",
            has_speech=True,
            transcript_char_len=4,
            absolute_start=2_000,
            absolute_end=1_000,
        )


def test_transcript_segment():
    seg = TranscriptSegment(start=0.0, end=5.0, text="Hello")
    assert seg.text == "Hello"


# ---------------------------------------------------------------------------
# EventSummaryRecord
# ---------------------------------------------------------------------------


def test_event_summary_record_valid():
    ev = EventSummaryRecord(
        event_summary_id="ev_0",
        day="day1",
        camera_id="Allie",
        camera_type="ego",
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_320_000,
        member_clip_ids=["c0", "c1", "c2", "c3"],
        event_summary="Allie prepared lunch in the kitchen.",
    )
    assert ev.source_type == "main_event_summary"
    assert len(ev.member_clip_ids) == 4


def test_event_summary_end_after_start():
    with pytest.raises(Exception):
        EventSummaryRecord(
            event_summary_id="ev_bad",
            day="day1",
            camera_id="Allie",
            camera_type="ego",
            absolute_start=1_000,
            absolute_end=500,
        )


# ---------------------------------------------------------------------------
# AuxRecord
# ---------------------------------------------------------------------------


def test_aux_record_photo():
    r = AuxRecord(
        clip_id="photo_day1_Allie_001",
        source_type="aux_photo",
        modality="image",
        day="day1",
        participant_id="Allie",
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_201_000,
        asset_path="/data/aux/photo/Allie/001.jpg",
    )
    assert r.modality == "image"


def test_aux_record_rejects_non_aux_source_type():
    with pytest.raises(Exception):
        AuxRecord(
            clip_id="bad",
            source_type="main_clip",  # not an aux type
            modality="video",
            day="day1",
            absolute_start=1_000,
            absolute_end=2_000,
        )


# ---------------------------------------------------------------------------
# RerankerOutput
# ---------------------------------------------------------------------------


def test_reranker_output_valid():
    out = RerankerOutput(
        relevance=3,
        support={"a": 2, "b": 1, "c": 0, "d": 3},
        keep=True,
        rationale="Strong visual match.",
    )
    assert out.relevance == 3
    assert out.keep is True


def test_reranker_output_missing_support_key():
    with pytest.raises(Exception):
        RerankerOutput(
            relevance=2,
            support={"a": 1, "b": 2},  # missing c and d
            keep=True,
            rationale="incomplete",
        )


def test_reranker_output_support_out_of_range():
    with pytest.raises(Exception):
        RerankerOutput(
            relevance=2,
            support={"a": 5, "b": 0, "c": 0, "d": 0},  # a=5 out of range
            keep=True,
            rationale="bad",
        )


# ---------------------------------------------------------------------------
# EvalQuestion
# ---------------------------------------------------------------------------


def test_eval_question_valid():
    q = EvalQuestion(
        question_id="2026_q1",
        query="What did Allie do after breakfast?",
        answers={"a": "Worked", "b": "Slept", "c": "Cooked", "d": "Walked"},
    )
    assert q.question_id == "2026_q1"
    assert q.ground_truth is None


def test_eval_question_missing_choice():
    with pytest.raises(Exception):
        EvalQuestion(
            question_id="bad",
            query="Q?",
            answers={"a": "A", "b": "B"},
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_prediction_valid_choices():
    for choice in ("a", "b", "c", "d"):
        p = Prediction(question_id="q1", predicted_answer=choice)
        assert p.predicted_answer == choice


def test_prediction_invalid_choice():
    with pytest.raises(Exception):
        Prediction(question_id="q1", predicted_answer="e")


# ---------------------------------------------------------------------------
# QdrantPoint
# ---------------------------------------------------------------------------


def test_qdrant_point_minimal():
    pt = QdrantPoint(
        point_id="abc123",
        record_id="clip_0",
        source_type="main_clip",
        modality="video",
    )
    assert pt.is_placeholder is False
    assert pt.sampled_frame_paths == []
