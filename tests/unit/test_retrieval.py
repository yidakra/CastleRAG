"""Tests for routing and retrieval logic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from castlerag.retrieval.candidate_expand import _collect_frame_paths
from castlerag.retrieval.filters import build_filter
from castlerag.retrieval.search import (
    _collapse_hits,
    _dense_search,
    _query_variants,
    reciprocal_rank_fusion,
    retrieve,
)
from castlerag.retrieval.transcript_lexical import score_windows
from castlerag.routing.question_router import (
    RouteEvidenceProfile,
    RouteHints,
    route_question,
)
from castlerag.schemas import (
    EvalQuestion,
    RetrievalHit,
    TranscriptSegment,
    TranscriptWindow,
)


def _question() -> EvalQuestion:
    return EvalQuestion(
        question_id="q1",
        query="What did Allie say after breakfast in the kitchen?",
        answers={
            "a": "She went to work",
            "b": "She cooked soup",
            "c": "She called Bjorn",
            "d": "She left the house",
        },
    )


def test_query_variants_mcq_includes_choices():
    """A real MCQ question expands its choices into a dense query variant."""
    variants = _query_variants(_question(), RouteHints(route="speech_text"))
    assert any("Choices:" in v and "She cooked soup" in v for v in variants)


def test_query_variants_free_form_drops_choices():
    """An open question must NOT inject blank 'Choices: A . B . ...' noise."""
    free = EvalQuestion(
        question_id="q_ff",
        query="What instrument did Cathal teach Allie to play?",
        answers={"a": "", "b": "", "c": "", "d": ""},
    )
    variants = _query_variants(free, RouteHints(route="speech_text"))
    assert variants == ["What instrument did Cathal teach Allie to play?"]
    assert not any("Choices:" in v for v in variants)


def _windows() -> list[TranscriptWindow]:
    return [
        TranscriptWindow(
            transcript_window_id="tw_1",
            day="day1",
            camera_id="Allie",
            camera_type="ego",
            participant_id="Allie",
            room="Kitchen",
            hour=8,
            transcript_text=(
                "After breakfast Allie said she would call Bjorn from the "
                "kitchen."
            ),
            transcript_segments=[
                TranscriptSegment(start=0.0, end=4.0, text="After breakfast")
            ],
            has_speech=True,
            transcript_char_len=66,
            absolute_start=1_672_531_200_000,
            absolute_end=1_672_531_215_000,
        ),
        TranscriptWindow(
            transcript_window_id="tw_2",
            day="day1",
            camera_id="Allie",
            camera_type="ego",
            participant_id="Allie",
            room="Meeting",
            hour=9,
            transcript_text="Allie quietly walked into the meeting room.",
            transcript_segments=[],
            has_speech=True,
            transcript_char_len=37,
            absolute_start=1_672_531_300_000,
            absolute_end=1_672_531_315_000,
        ),
    ]


def test_route_question_extracts_route_and_hints():
    hints = route_question(
        question="On day 1, what did Allie say before entering the kitchen?",
        choices={"a": "hello", "b": "bye", "c": "thanks", "d": "nothing"},
    )
    assert hints.route == "temporal"
    assert hints.day == "day1"
    assert hints.participant == "Allie"
    assert hints.room == "Kitchen"
    assert hints.has_speech_cue is True
    assert hints.has_temporal_cue is True


def test_score_windows_prefers_exact_overlap_and_hints():
    class FakeBM25:
        def get_scores(self, tokens: list[str]) -> np.ndarray:
            return np.asarray([2.0, 1.0], dtype=np.float32)

    bundle = SimpleNamespace(bm25=FakeBM25())
    hits = score_windows(
        bm25_index=bundle,
        windows=_windows(),
        query=_question().query,
        choices=_question().answers,
        day_hint="day1",
        person_hint="Allie",
        room_hint="Kitchen",
        top_k=2,
    )
    assert hits[0].record_id == "tw_1"
    assert hits[0].score > hits[1].score
    # Lexical hits explicitly carry within-hour second offsets so downstream
    # consumers don't fall back to recomputing them from absolute_start.
    assert hits[0].start_seconds == 0.0
    assert hits[0].end_seconds == 15.0


def test_reciprocal_rank_fusion_merges_on_record_id():
    hit_a = RetrievalHit(
        rank=1,
        score=1.0,
        point_id="p1",
        record_id="r1",
        source_type="main_clip",
        modality="video",
    )
    hit_b = RetrievalHit(
        rank=2,
        score=0.8,
        point_id="p2",
        record_id="r2",
        source_type="main_clip",
        modality="video",
    )
    hit_c = RetrievalHit(
        rank=1,
        score=0.9,
        point_id="x1",
        record_id="r1",
        source_type="transcript_window",
        modality="text",
    )
    fused = reciprocal_rank_fusion([[hit_a, hit_b], [hit_c]], k=60)
    assert fused[0].record_id == "r1"
    assert fused[1].record_id == "r2"
    assert fused[0].rank == 1


def test_retrieve_fuses_transcript_and_multimodal_hits():
    class FakeBM25:
        def get_scores(self, tokens: list[str]) -> np.ndarray:
            return np.asarray([3.0, 1.0], dtype=np.float32)

    class FakeEmbedClient:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            assert len(texts) == 2
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    class FakePoint:
        def __init__(self, pid: str, score: float, payload: dict) -> None:
            self.id = pid
            self.score = score
            self.payload = payload

    class FakeQdrantClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def query_points(self, **kwargs):
            self.calls.append(kwargs)
            source_type = next(
                condition.match.value
                for condition in kwargs["query_filter"].must
                if condition.key == "source_type"
            )
            if source_type == "transcript_window":
                points = [
                    FakePoint(
                        "pt_tx_1",
                        0.9,
                        {
                            "record_id": "tw_1",
                            "source_type": "transcript_window",
                            "modality": "text",
                            "day": "day1",
                            "camera_id": "Allie",
                            "participant_id": "Allie",
                            "absolute_start": 1_672_531_200_000,
                            "absolute_end": 1_672_531_215_000,
                            "transcript_text": "Allie said she would call Bjorn.",
                        },
                    )
                ]
            elif source_type == "main_clip":
                points = [
                    FakePoint(
                        "pt_clip_1",
                        0.8,
                        {
                            "record_id": "clip_1",
                            "source_type": "main_clip",
                            "modality": "video",
                            "day": "day1",
                            "camera_id": "Allie",
                            "participant_id": "Allie",
                            "absolute_start": 1_672_531_200_000,
                            "absolute_end": 1_672_531_230_000,
                            "event_summary": "Allie speaks in the kitchen.",
                            "asset_path": "/tmp/clip.mp4",
                        },
                    )
                ]
            else:
                points = []
            return SimpleNamespace(points=points)

    bm25_bundle = SimpleNamespace(bm25=FakeBM25(), windows=_windows())
    retrieval_cfg = SimpleNamespace(
        transcript_top_k=30,
        event_summary_top_k=20,
        video_top_k=20,
        photo_top_k=16,
        aux_video_top_k=8,
        heartrate_top_k=8,
        gaze_top_k=8,
        thermal_top_k=8,
        rrf_k=60,
        max_candidate_videos=4,
        frames_per_candidate=32,
        max_aux_images=16,
        max_evidence_rows=50,
        modality_score_thresholds={},
    )
    hints = route_question(_question().query, _question().answers)
    qdrant = FakeQdrantClient()
    hits = retrieve(
        question=_question(),
        hints=hints,
        qdrant_client=qdrant,
        collection_name="castle_test",
        bm25_index=bm25_bundle,
        embed_client=FakeEmbedClient(),
        retrieval_cfg=retrieval_cfg,
    )
    assert hits
    assert hits[0].record_id == "tw_1"
    assert any(hit.source_type == "main_clip" for hit in hits)
    assert len(hits) <= 50


def test_retrieve_excludes_rejected_cameras_from_bm25_lane():
    # Regression: dense lanes hard-exclude via must_not server-side, but the
    # BM25 transcript lane is fused locally — excluded cameras must be dropped
    # there too, or they leak back through RRF (PR #56 review finding).
    class FakeBM25:
        def get_scores(self, tokens: list[str]) -> np.ndarray:
            return np.asarray([3.0, 1.0], dtype=np.float32)

    class FakeEmbedClient:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    class EmptyQdrant:
        # Dense lanes return nothing, isolating the BM25 transcript lane.
        def query_points(self, **kwargs):
            return SimpleNamespace(points=[])

    bm25_bundle = SimpleNamespace(bm25=FakeBM25(), windows=_windows())  # both "Allie"
    cfg = SimpleNamespace(
        transcript_top_k=30, event_summary_top_k=20, video_top_k=20,
        photo_top_k=16, aux_video_top_k=8, heartrate_top_k=8, gaze_top_k=8,
        thermal_top_k=8, rrf_k=60, max_candidate_videos=4, frames_per_candidate=32,
        max_aux_images=16, max_evidence_rows=50, modality_score_thresholds={},
    )

    def _run(exclude: tuple) -> list:
        hints = route_question(_question().query, _question().answers)
        hints.exclude_cameras = exclude
        return retrieve(
            question=_question(), hints=hints, qdrant_client=EmptyQdrant(),
            collection_name="castle_test", bm25_index=bm25_bundle,
            embed_client=FakeEmbedClient(), retrieval_cfg=cfg,
        )

    # Without exclusion the BM25 "Allie" windows surface through the lane...
    assert any(h.camera_id == "Allie" for h in _run(()))
    # ...and excluding "Allie" drops them from the fused transcript lane.
    assert all(h.camera_id != "Allie" for h in _run(("Allie",)))


def test_retrieve_consumes_router_budget_profile_without_reparsing():
    class FakeBM25:
        def get_scores(self, tokens: list[str]) -> np.ndarray:
            return np.asarray([5.0, 4.0], dtype=np.float32)

    class FakeEmbedClient:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    class FakePoint:
        def __init__(self, pid: str, score: float, payload: dict) -> None:
            self.id = pid
            self.score = score
            self.payload = payload

    class FakeQdrantClient:
        def query_points(self, **kwargs):
            source_type = next(
                condition.match.value
                for condition in kwargs["query_filter"].must
                if condition.key == "source_type"
            )
            if source_type == "transcript_window":
                return SimpleNamespace(
                    points=[
                        FakePoint(
                            "pt_tx_3",
                            0.95,
                            {
                                "record_id": "tw_2",
                                "source_type": "transcript_window",
                                "modality": "text",
                                "day": "day1",
                                "camera_id": "Allie",
                                "participant_id": "Allie",
                                "absolute_start": 1_672_531_300_000,
                                "absolute_end": 1_672_531_315_000,
                                "transcript_text": (
                                    "Allie quietly walked into the office."
                                ),
                            },
                        )
                    ]
                )
            if source_type == "main_clip":
                return SimpleNamespace(
                    points=[
                        FakePoint(
                            "pt_clip_2",
                            0.7,
                            {
                                "record_id": "clip_2",
                                "source_type": "main_clip",
                                "modality": "video",
                                "day": "day1",
                                "camera_id": "Allie",
                                "participant_id": "Allie",
                                "absolute_start": 1_672_531_320_000,
                                "absolute_end": 1_672_531_350_000,
                                "asset_path": "/tmp/clip_2.mp4",
                            },
                        )
                    ]
                )
            return SimpleNamespace(points=[])

    bm25_bundle = SimpleNamespace(bm25=FakeBM25(), windows=_windows())
    retrieval_cfg = SimpleNamespace(
        transcript_top_k=30,
        event_summary_top_k=20,
        video_top_k=20,
        photo_top_k=16,
        aux_video_top_k=8,
        heartrate_top_k=8,
        gaze_top_k=8,
        thermal_top_k=8,
        rrf_k=60,
        max_candidate_videos=4,
        frames_per_candidate=32,
        max_aux_images=16,
        max_evidence_rows=50,
        modality_score_thresholds={},
    )
    hints = route_question(
        "What color shirt was Allie wearing in the kitchen?",
        {"a": "Blue", "b": "Black", "c": "White", "d": "Red"},
    )
    hints.evidence_profile = RouteEvidenceProfile(
        transcript_budget=1,
        candidate_video_budget=4,
        frames_per_candidate_video=32,
        auxiliary_image_budget=16,
        max_evidence_rows=50,
        source_priority=("main_clip", "transcript_window"),
    )
    hits = retrieve(
        question=_question(),
        hints=hints,
        qdrant_client=FakeQdrantClient(),
        collection_name="castle_test",
        bm25_index=bm25_bundle,
        embed_client=FakeEmbedClient(),
        retrieval_cfg=retrieval_cfg,
    )
    assert sum(1 for hit in hits if hit.source_type == "transcript_window") == 1
    assert hits[0].source_type == "main_clip"


# ---------------------------------------------------------------------------
# filters.py — uncovered branches
# ---------------------------------------------------------------------------


def test_build_filter_camera_id():
    f = build_filter(camera_id="Allie")
    assert f is not None
    keys = [c.key for c in f.must]
    assert "camera_id" in keys


def test_build_filter_exclude_camera_ids_adds_must_not():
    f = build_filter(day="day1", exclude_camera_ids=["Kitchen", "Allie"])
    assert f is not None
    assert [c.key for c in f.must] == ["day"]
    assert [c.match.value for c in f.must_not] == ["Kitchen", "Allie"]


def test_build_filter_only_exclude_returns_must_not_only():
    f = build_filter(exclude_camera_ids=["Kitchen"])
    assert f is not None
    assert f.must is None
    assert len(f.must_not) == 1


def test_build_filter_empty_exclude_is_noop():
    # An empty exclusion must not create a filter on its own (keeps eval path
    # unfiltered when no cameras were rejected).
    assert build_filter(exclude_camera_ids=[]) is None
    assert build_filter(exclude_camera_ids=None) is None


def test_build_filter_participant_id():
    f = build_filter(participant_id="Bjorn")
    assert f is not None
    keys = [c.key for c in f.must]
    assert "participant_id" in keys


def test_build_filter_room():
    f = build_filter(room="Kitchen")
    assert f is not None
    keys = [c.key for c in f.must]
    assert "room" in keys


def test_build_filter_has_speech():
    f = build_filter(has_speech=True)
    assert f is not None
    keys = [c.key for c in f.must]
    assert "has_speech" in keys


def test_build_filter_time_range_start_ms():
    f = build_filter(time_range_start_ms=1000)
    assert f is not None
    keys = [c.key for c in f.must]
    assert "absolute_end" in keys


def test_build_filter_time_range_end_ms():
    f = build_filter(time_range_end_ms=5000)
    assert f is not None
    keys = [c.key for c in f.must]
    assert "absolute_start" in keys


def test_build_filter_invalid_time_range_raises():
    with pytest.raises(ValueError, match="time_range_start_ms"):
        build_filter(time_range_start_ms=9000, time_range_end_ms=1000)


def test_build_filter_source_type_and_modality():
    f = build_filter(source_type="aux_photo", modality="image")
    assert f is not None
    keys = [c.key for c in f.must]
    assert "source_type" in keys
    assert "modality" in keys


def test_build_filter_no_conditions_returns_none():
    assert build_filter() is None


# ---------------------------------------------------------------------------
# search.py — uncovered branches
# ---------------------------------------------------------------------------


def test_retrieve_raises_on_non_2d_embedding():
    """Line 72: ValueError when embed_texts returns 1D array."""

    class FakeBM25:
        def get_scores(self, tokens: list[str]) -> np.ndarray:
            return np.asarray([1.0, 0.5], dtype=np.float32)

    class FakeEmbedClient1D:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            # Return a 1D array instead of 2D
            return np.asarray([1.0, 0.0], dtype=np.float32)

    bm25_bundle = SimpleNamespace(bm25=FakeBM25(), windows=_windows())
    retrieval_cfg = SimpleNamespace(
        transcript_top_k=5,
        event_summary_top_k=5,
        video_top_k=5,
        photo_top_k=5,
        aux_video_top_k=5,
        heartrate_top_k=5,
        gaze_top_k=5,
        thermal_top_k=5,
        rrf_k=60,
        max_candidate_videos=4,
        frames_per_candidate=32,
        max_aux_images=16,
        max_evidence_rows=50,
        modality_score_thresholds={},
    )
    hints = route_question(_question().query, _question().answers)
    with pytest.raises(ValueError, match="2D"):
        retrieve(
            question=_question(),
            hints=hints,
            qdrant_client=None,
            collection_name="test",
            bm25_index=bm25_bundle,
            embed_client=FakeEmbedClient1D(),
            retrieval_cfg=retrieval_cfg,
        )


def _make_hit(
    record_id: str, source_type: str = "transcript_window", rank: int = 1
) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=0.9,
        point_id=f"pt_{record_id}",
        record_id=record_id,
        source_type=source_type,
        modality="text",
    )


def test_collect_frame_paths_max_frames_zero_disables_visual_inputs():
    """max_frames=0 returns no frames even when hits have sampled frames."""
    hit = _make_hit("vid_1", source_type="multimodal_frame")
    hit.sampled_frame_paths = ["frame_0.jpg", "frame_1.jpg"]
    assert _collect_frame_paths([hit], max_frames=0) == []


def test_collapse_hits_stops_at_max_rows():
    """Lines 233-234: early break when len(kept) >= max_rows."""
    hits = [_make_hit(f"tx_{i}", rank=i + 1) for i in range(10)]
    retrieval_cfg = SimpleNamespace(
        transcript_top_k=10,
        max_candidate_videos=10,
        max_aux_images=10,
        max_evidence_rows=3,
    )
    hints = route_question(_question().query, _question().answers)
    hints.evidence_profile = RouteEvidenceProfile(
        transcript_budget=10,
        candidate_video_budget=10,
        frames_per_candidate_video=32,
        auxiliary_image_budget=10,
        max_evidence_rows=3,
        source_priority=("transcript_window",),
    )
    result = _collapse_hits(hits, hints, retrieval_cfg)
    assert len(result) == 3


def test_route_priority_fallback_for_unknown_source_type():
    """_route_priority returns len(source_priority) for an unknown source_type."""
    hit = _make_hit("unknown_1", source_type="aux_unknown")
    retrieval_cfg = SimpleNamespace(
        transcript_top_k=10,
        max_candidate_videos=10,
        max_aux_images=10,
        max_evidence_rows=10,
    )
    hints = route_question(_question().query, _question().answers)
    # source_priority does not include "aux_unknown"
    hints.evidence_profile = RouteEvidenceProfile(
        transcript_budget=10,
        candidate_video_budget=10,
        frames_per_candidate_video=32,
        auxiliary_image_budget=10,
        max_evidence_rows=10,
        source_priority=("transcript_window", "main_clip"),
    )
    # Should not raise; unknown source type goes to end of priority
    result = _collapse_hits([hit], hints, retrieval_cfg)
    assert len(result) == 1
    assert result[0].source_type == "aux_unknown"


def test_retrieve_does_not_hard_filter_dense_search_by_room():
    """Regression for issue #50: the router's room hint must not become a hard
    Qdrant filter. Ego clips/windows carry room=None (only fixed cameras set
    it), so filtering dense retrieval by hints.room zeroes out all ego evidence
    in ego scope. Room must stay a soft signal (BM25 + reranker) only."""

    class FakeBM25:
        def get_scores(self, tokens: list[str]) -> np.ndarray:
            return np.asarray([0.0, 0.0], dtype=np.float32)

    class FakeEmbedClient:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    class FakeQdrantClient:
        def __init__(self) -> None:
            self.filter_keys: list[set[str]] = []

        def query_points(self, **kwargs):
            self.filter_keys.append(
                {condition.key for condition in kwargs["query_filter"].must}
            )
            return SimpleNamespace(points=[])

    question = EvalQuestion(
        question_id="q_fridge",
        query="What brand is the fridge in the kitchen?",
        answers={"a": "Samsung", "b": "Whirlpool", "c": "Bosch", "d": "Liebherr"},
    )
    hints = route_question(question.query, question.answers)
    # The router DID extract the room hint...
    assert hints.room == "Kitchen"

    retrieval_cfg = SimpleNamespace(
        transcript_top_k=30,
        event_summary_top_k=20,
        video_top_k=20,
        photo_top_k=16,
        aux_video_top_k=8,
        heartrate_top_k=8,
        gaze_top_k=8,
        thermal_top_k=8,
        rrf_k=60,
        max_candidate_videos=4,
        frames_per_candidate=32,
        max_aux_images=16,
        max_evidence_rows=50,
        modality_score_thresholds={},
    )
    qdrant = FakeQdrantClient()
    retrieve(
        question=question,
        hints=hints,
        qdrant_client=qdrant,
        collection_name="castle_test",
        bm25_index=SimpleNamespace(bm25=FakeBM25(), windows=_windows()),
        embed_client=FakeEmbedClient(),
        retrieval_cfg=retrieval_cfg,
    )
    # ...but no dense search filtered on room.
    assert qdrant.filter_keys  # dense searches actually ran
    assert all("room" not in keys for keys in qdrant.filter_keys)


def test_dense_search_maps_caption_scene_graph_and_ocr_from_payload():
    """clip_caption / scene_graph_text / ocr_text in the payload land on the hit."""

    class FakePoint:
        def __init__(self, pid: str, score: float, payload: dict) -> None:
            self.id = pid
            self.score = score
            self.payload = payload

    class FakeQdrantClient:
        def query_points(self, **kwargs):
            return SimpleNamespace(
                points=[
                    FakePoint(
                        "pt_clip_9",
                        0.8,
                        {
                            "record_id": "clip_9",
                            "source_type": "main_clip",
                            "modality": "video",
                            "day": "day1",
                            "camera_id": "Kitchen",
                            "clip_caption": "Werner at the stove in a W3C apron.",
                            "scene_graph_text": "person at stove (center); apron",
                            "ocr_text": "W3C",
                            "asset_path": "/tmp/clip_9.mp4",
                            "sampled_frame_paths": ["/tmp/f0.jpg"],
                        },
                    )
                ]
            )

    hits = _dense_search(
        qdrant_client=FakeQdrantClient(),
        collection_name="castle_test",
        query_vector=[1.0, 0.0],
        limit=5,
        source_type="main_clip",
        modality="video",
    )
    assert len(hits) == 1
    hit = hits[0]
    assert hit.clip_caption == "Werner at the stove in a W3C apron."
    assert hit.scene_graph_text == "person at stove (center); apron"
    assert hit.ocr_text == "W3C"
    assert hit.asset_path == "/tmp/clip_9.mp4"
    assert hit.sampled_frame_paths == ["/tmp/f0.jpg"]
    # Fields survive the model_copy calls used by RRF and collapse.
    fused = reciprocal_rank_fusion([hits])
    assert fused[0].clip_caption == hit.clip_caption
    assert fused[0].scene_graph_text == hit.scene_graph_text


def test_dense_search_leaves_caption_fields_none_when_payload_lacks_them():
    class FakePoint:
        def __init__(self, pid: str, score: float, payload: dict) -> None:
            self.id = pid
            self.score = score
            self.payload = payload

    class FakeQdrantClient:
        def query_points(self, **kwargs):
            return SimpleNamespace(
                points=[
                    FakePoint(
                        "pt_tw_9",
                        0.8,
                        {
                            "record_id": "tw_9",
                            "source_type": "transcript_window",
                            "modality": "text",
                            "transcript_text": "hello",
                        },
                    )
                ]
            )

    (hit,) = _dense_search(
        qdrant_client=FakeQdrantClient(),
        collection_name="castle_test",
        query_vector=[1.0, 0.0],
        limit=5,
        source_type="transcript_window",
        modality="text",
    )
    assert hit.clip_caption is None
    assert hit.scene_graph_text is None
    assert hit.ocr_text is None


# ---------------------------------------------------------------------------
# _collapse_hits — separate clip / event-summary budgets and the clip floor
# (issue #50: visual evidence retrieved but lost at collapse time)
# ---------------------------------------------------------------------------


def _mixed_source_hits(
    *, transcripts: int = 30, summaries: int = 6, clips: int = 6
) -> list[RetrievalHit]:
    """Fused hits ranked transcripts first, then summaries, then clips."""
    hits: list[RetrievalHit] = []
    rank = 1
    for i in range(transcripts):
        hits.append(_make_hit(f"tw_{i}", "transcript_window", rank))
        rank += 1
    for i in range(summaries):
        hits.append(_make_hit(f"es_{i}", "main_event_summary", rank))
        rank += 1
    for i in range(clips):
        clip = _make_hit(f"clip_{i}", "main_clip", rank)
        hits.append(clip.model_copy(update={"modality": "video"}))
        rank += 1
    return hits


def _budget_cfg(**overrides) -> SimpleNamespace:
    cfg = dict(
        transcript_top_k=30,
        max_candidate_videos=4,
        max_event_summaries=4,
        min_clip_hits=2,
        max_aux_images=16,
        max_evidence_rows=50,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _count(hits: list[RetrievalHit], source_type: str) -> int:
    return sum(1 for hit in hits if hit.source_type == source_type)


@pytest.mark.parametrize("route", ["temporal", "speech_text", "mixed"])
def test_collapse_hits_keeps_clips_when_summaries_rank_first(route):
    """Routes that rank event summaries ahead of clips still keep clip rows."""
    hints = RouteHints(route=route)
    result = _collapse_hits(_mixed_source_hits(), hints, _budget_cfg())
    assert _count(result, "main_event_summary") == 4
    assert _count(result, "main_clip") == 4
    assert _count(result, "transcript_window") == 30


def test_collapse_hits_static_visual_keeps_event_summaries_alongside_clips():
    """static_visual ranks clips first; event summaries (with OCR) must survive."""
    hints = RouteHints(route="static_visual")
    result = _collapse_hits(_mixed_source_hits(), hints, _budget_cfg())
    assert _count(result, "main_clip") == 4
    assert _count(result, "main_event_summary") == 4
    # static_visual transcript budget is 10 (route profile), not 30.
    assert _count(result, "transcript_window") == 10
    assert result[0].source_type == "main_clip"


def test_collapse_hits_reserves_min_clip_hits_under_max_rows():
    """The max_rows cap cannot squeeze out the top clips behind transcripts."""
    hints = RouteHints(route="temporal")  # transcripts, summaries, then clips
    cfg = _budget_cfg(max_evidence_rows=5, min_clip_hits=2)
    result = _collapse_hits(_mixed_source_hits(), hints, cfg)
    assert len(result) == 5
    assert _count(result, "main_clip") == 2
    assert [hit.record_id for hit in result if hit.source_type == "main_clip"] == [
        "clip_0",
        "clip_1",
    ]
    assert _count(result, "transcript_window") == 3
    # Output ranks are re-numbered contiguously.
    assert [hit.rank for hit in result] == [1, 2, 3, 4, 5]


def test_collapse_hits_clip_floor_is_noop_without_clips():
    """No clips retrieved -> nothing is reserved and rows fill as before."""
    hints = RouteHints(route="temporal")
    cfg = _budget_cfg(max_evidence_rows=5)
    result = _collapse_hits(_mixed_source_hits(clips=0), hints, cfg)
    assert len(result) == 5
    assert _count(result, "transcript_window") == 5


def test_collapse_hits_clip_floor_capped_by_clip_budget():
    """min_clip_hits never exceeds the clip budget."""
    hints = RouteHints(route="temporal")
    cfg = _budget_cfg(max_candidate_videos=1, min_clip_hits=3, max_evidence_rows=4)
    result = _collapse_hits(_mixed_source_hits(), hints, cfg)
    assert _count(result, "main_clip") == 1
    assert len(result) == 4


def test_collapse_hits_counts_lexical_and_dense_hits_for_same_record_once():
    """A clip reached via a lexical lane (visual_lexical:<id>) and via dense
    Qdrant shares one record_id; it must occupy one budget slot, not two."""
    hits = _mixed_source_hits(transcripts=0, summaries=0, clips=4)
    lexical_dupes = [
        hits[0].model_copy(update={"point_id": "visual_lexical:clip_0", "rank": 9}),
        hits[1].model_copy(update={"point_id": "visual_lexical:clip_1", "rank": 10}),
    ]
    hints = RouteHints(route="static_visual")
    cfg = _budget_cfg(max_candidate_videos=3, min_clip_hits=2)
    result = _collapse_hits(hits + lexical_dupes, hints, cfg)
    ids = [hit.record_id for hit in result]
    assert ids == ["clip_0", "clip_1", "clip_2"]
    assert len(set(ids)) == len(ids)
    # The first occurrence in priority order (the better rank) is the one kept.
    assert all(not hit.point_id.startswith("visual_lexical:") for hit in result)


def test_collapse_hits_defaults_when_config_lacks_new_keys():
    """Configs without max_event_summaries/min_clip_hits use the module defaults."""
    hints = RouteHints(route="temporal")
    legacy_cfg = SimpleNamespace(
        transcript_top_k=30,
        max_candidate_videos=4,
        max_aux_images=16,
        max_evidence_rows=50,
    )
    result = _collapse_hits(_mixed_source_hits(), hints, legacy_cfg)
    assert _count(result, "main_event_summary") == 4
    assert _count(result, "main_clip") == 4


def test_collect_frame_descriptions_caps_described_clips():
    from types import SimpleNamespace

    from castlerag.retrieval.candidate_expand import (
        MAX_DESCRIBED_CLIPS,
        _collect_frame_descriptions,
    )

    rows = [
        SimpleNamespace(
            source_type="main_clip",
            record_id=f"clip{i}",
            camera_id="Allie",
            clip_caption=f"caption {i}",
            scene_graph_text=f"graph {i}",
            asset_path=None,
        )
        for i in range(MAX_DESCRIBED_CLIPS + 5)
    ]
    values = _collect_frame_descriptions(rows)
    assert len(values) == 2 * MAX_DESCRIBED_CLIPS
    assert values[0].startswith("clip clip0 ")
    assert not any(f"clip clip{MAX_DESCRIBED_CLIPS} " in v for v in values)


def test_collect_frame_descriptions_cap_counts_only_described_clips():
    from types import SimpleNamespace

    from castlerag.retrieval.candidate_expand import (
        MAX_DESCRIBED_CLIPS,
        _collect_frame_descriptions,
    )

    def _row(i, caption):
        return SimpleNamespace(
            source_type="main_clip",
            record_id=f"clip{i}",
            camera_id="Allie",
            clip_caption=caption,
            scene_graph_text=None,
            asset_path=f"/clips/{i}.mp4",
        )

    # Un-annotated clips first, then more captioned clips than the cap.
    rows = [_row(i, None) for i in range(MAX_DESCRIBED_CLIPS)] + [
        _row(100 + i, f"caption {i}") for i in range(MAX_DESCRIBED_CLIPS + 2)
    ]
    values = _collect_frame_descriptions(rows)
    captions = [v for v in values if " caption: " in v]
    assets = [v for v in values if v.startswith("clip asset: ")]
    # Un-annotated clips did not eat the cap; the first N captioned clips render.
    assert len(captions) == MAX_DESCRIBED_CLIPS
    assert captions[0].startswith("clip clip100 ")
    # Every other main_clip row, including the two past the cap, keeps its path.
    assert len(assets) == MAX_DESCRIBED_CLIPS + 2


def test_collect_frame_paths_shares_budget_and_samples_each_row_evenly():
    from types import SimpleNamespace

    from castlerag.retrieval.candidate_expand import _collect_frame_paths

    rows = [
        SimpleNamespace(sampled_frame_paths=[f"/c{r}/{i:02d}.jpg" for i in range(30)])
        for r in range(3)
    ]
    picked = _collect_frame_paths(rows, max_frames=32)
    assert len(picked) == 32 and len(set(picked)) == 32
    per_row = {r: [p for p in picked if p.startswith(f"/c{r}/")] for r in range(3)}
    assert [len(per_row[r]) for r in range(3)] == [11, 11, 10]
    # Every row contributes mid- and late-clip frames, not just its opening.
    for r in range(3):
        secs = sorted(int(p.split("/")[-1][:2]) for p in per_row[r])
        assert secs[0] < 5 and secs[-1] > 24
    # Without a cap, all frames are kept in order (deduplicated).
    assert len(_collect_frame_paths(rows, max_frames=None)) == 90
