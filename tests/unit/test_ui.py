"""Tests for the UI backbone: YouTube mirror, placeholder engine, app factory."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from castlerag.ui.chat import (
    CameraAngle,
    ChatTurnResult,
    Claim,
    EvidenceMoment,
    PlaceholderEngine,
    QueryGroup,
    SupportLevel,
)
from castlerag.ui.youtube import PLACEHOLDER_VIDEO_ID, YouTubeMirror

# ---------------------------------------------------------------------------
# YouTube mirror
# ---------------------------------------------------------------------------


def test_mirror_from_csv_loads_real_mapping():
    mirror = YouTubeMirror.from_csv()
    assert len(mirror.mapping) == 666  # full CASTLE mirror
    # Real id from the CASTLE viewer's videos.json.
    assert mirror.video_id("day1", "Allie", 8) == "XYkq1PPZu9A"
    assert mirror.is_placeholder("day1", "Allie", 8) is False


def test_mirror_embed_url_includes_start_offset_and_video_id():
    mirror = YouTubeMirror(mapping={("day1", "Allie", 8): "ABCDEFGHIJK"})
    url = mirror.embed_url("day1", "Allie", 8, start_seconds=42.7)
    assert "/embed/ABCDEFGHIJK?" in url
    assert "start=42" in url
    assert url.startswith("https://www.youtube-nocookie.com")


def test_mirror_default_and_watch_urls():
    mirror = YouTubeMirror.from_csv()
    assert "/embed/" in mirror.default_embed_url()
    watch = mirror.watch_url("day1", "Allie", 8, 90)
    assert watch == "https://www.youtube.com/watch?v=XYkq1PPZu9A&t=90s"


def test_mirror_unknown_key_falls_back_to_placeholder():
    mirror = YouTubeMirror(mapping={})
    assert mirror.video_id("day9", "Nobody", 23) == PLACEHOLDER_VIDEO_ID
    assert mirror.is_placeholder("day9", "Nobody", 23) is True


def test_mirror_is_placeholder_tracks_resolved_id():
    mirror = YouTubeMirror(mapping={("day1", "Allie", 8): "REALVIDEOID0"})
    assert mirror.is_placeholder("Day1", "Allie", 8) is False  # day case-normalized
    assert mirror.is_placeholder("day1", "Bjorn", 8) is True


def test_mirror_from_csv_missing_file_is_empty(tmp_path):
    mirror = YouTubeMirror.from_csv(tmp_path / "absent.csv")
    assert mirror.mapping == {}


# ---------------------------------------------------------------------------
# Placeholder engine — answer()
# ---------------------------------------------------------------------------


def test_answer_populates_claim_and_moments():
    result = PlaceholderEngine().answer("Who pranks Bjorn when he returns?")
    assert isinstance(result, ChatTurnResult)
    assert result.is_placeholder is True
    assert isinstance(result.claim, Claim)
    assert result.claim.support is SupportLevel.PARTIAL
    assert result.moments
    for moment in result.moments:
        assert isinstance(moment, EvidenceMoment)
        # Exactly three synchronized cameras, all at the same (day, hour, start).
        assert len(moment.cameras) == 3
        sync = {(c.day, c.hour, c.start_seconds) for c in moment.cameras}
        assert len(sync) == 1  # all three cameras share day/hour/start
        # Exactly one camera flagged best.
        assert sum(1 for c in moment.cameras if c.is_best) == 1


def test_answer_keeps_legacy_contract():
    result = PlaceholderEngine().answer("How many cups are on the table?")
    assert result.predicted_choice in {"a", "b", "c", "d"}
    assert result.route in {"static_visual", "speech_text", "temporal", "mixed"}
    assert set(result.support_priors) == {"a", "b", "c", "d"}
    assert abs(sum(result.support_priors.values()) - 1.0) < 1e-6
    assert result.evidence  # legacy EvidenceRef rows still populated


def test_answer_is_deterministic():
    engine = PlaceholderEngine()
    first = engine.answer("Where did Bjorn go after lunch?")
    second = engine.answer("Where did Bjorn go after lunch?")
    assert first.claim.text == second.claim.text
    assert [m.moment_id for m in first.moments] == [m.moment_id for m in second.moments]
    assert [
        c.match_score for m in first.moments for c in m.cameras
    ] == [c.match_score for m in second.moments for c in m.cameras]


def test_moments_resolve_to_real_embeds():
    mirror = YouTubeMirror.from_csv()
    engine = PlaceholderEngine.from_mirror(mirror)
    result = engine.answer("What was happening when the timer went off?")
    for moment in result.moments:
        for cam in moment.cameras:
            url = mirror.embed_url(cam.day, cam.camera_id, cam.hour, cam.start_seconds)
            assert "/embed/" in url
            assert mirror.is_placeholder(cam.day, cam.camera_id, cam.hour) is False


# ---------------------------------------------------------------------------
# Placeholder engine — refine()
# ---------------------------------------------------------------------------


def test_refine_strengthens_support_and_converges():
    engine = PlaceholderEngine()
    claim = "Luca is the one who sets up the prank."
    early = engine.refine(claim, "Show the doorway more clearly.", 2)
    late = engine.refine(claim, "Confirm Luca's hand on the frame.", 3)
    assert early.claim.support is SupportLevel.PARTIAL
    assert late.claim.support is SupportLevel.SUPPORTED
    # The claim text is preserved across refinements.
    assert early.claim.text == claim == late.claim.text


def test_refine_is_deterministic():
    engine = PlaceholderEngine()
    a = engine.refine("c", "sharper angle", 2)
    b = engine.refine("c", "sharper angle", 2)
    assert [m.moment_id for m in a.moments] == [m.moment_id for m in b.moments]
    assert a.moments[0].aggregate_score == b.moments[0].aggregate_score


# ---------------------------------------------------------------------------
# Serialization (str-Enum round-trips into a dcc.Store)
# ---------------------------------------------------------------------------


def test_query_group_json_round_trips():
    import json

    cam = CameraAngle("Klaus", "day1", 12, 749.0, 0.91, is_best=True)
    moment = EvidenceMoment(
        "m0", "12:29", "Doorway", 3, 0.74, "match 0.74", "#d97706", [cam]
    )
    group = QueryGroup(
        group_id="g1",
        iteration=1,
        question="q",
        answer_text="a",
        claim=Claim("c", SupportLevel.PARTIAL),
        moments=[moment],
    )
    payload = json.loads(json.dumps(asdict(group)))
    assert payload["claim"]["support"] == "partial"
    assert payload["moments"][0]["cameras"][0]["is_best"] is True


# ---------------------------------------------------------------------------
# App factory (requires dash)
# ---------------------------------------------------------------------------


def test_build_app_assembles_layout_and_callbacks():
    pytest.importorskip("dash")
    from castlerag.ui.app import build_app

    app = build_app()
    assert app.layout is not None
    assert app.callback_map  # callbacks registered
    # The Plotly evidence chart is wired as a callback output in the viewer.
    assert any("evidence-figure" in key for key in app.callback_map)


def test_viewer_outputs_match_their_output_specs():
    """`_viewer_outputs` must return exactly one value per declared Output."""
    import plotly.graph_objects as go

    from castlerag.ui.callbacks import _viewer_output_specs, _viewer_outputs

    moment = {
        "moment_id": "m0",
        "clock_label": "12:29",
        "camera_count": 3,
        "cameras": [
            {
                "camera_id": "Luca",
                "match_score": 0.8,
                "is_best": True,
                "embed_url": "https://example.test/embed",
            },
        ],
    }
    group = {
        "is_refinement": False,
        "claim": {"support": "partial"},
        "moments": [moment],
    }
    out = _viewer_outputs(group, moment, {"Luca": {"state": "pending"}}, 1)
    assert len(out) == len(_viewer_output_specs())
    assert isinstance(out[-1], go.Figure)  # last output is the evidence figure
