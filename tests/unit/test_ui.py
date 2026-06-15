"""Tests for the UI backbone: YouTube mirror, placeholder engine, figures, app."""

from __future__ import annotations

import pytest

from castlerag.ui.chat import ChatTurnResult, EvidenceRef, PlaceholderEngine
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
# Placeholder engine
# ---------------------------------------------------------------------------


def test_placeholder_engine_returns_valid_structure():
    engine = PlaceholderEngine()
    result = engine.answer("What did Allie say before entering the kitchen?")
    assert isinstance(result, ChatTurnResult)
    assert result.predicted_choice in {"a", "b", "c", "d"}
    assert result.route in {"static_visual", "speech_text", "temporal", "mixed"}
    assert result.is_placeholder is True
    assert result.evidence
    assert all(isinstance(item, EvidenceRef) for item in result.evidence)


def test_placeholder_engine_support_priors_normalized():
    result = PlaceholderEngine().answer("How many cups are on the table?")
    assert set(result.support_priors) == {"a", "b", "c", "d"}
    assert abs(sum(result.support_priors.values()) - 1.0) < 1e-6
    best = max(result.support_priors, key=result.support_priors.get)
    assert result.predicted_choice == best


def test_placeholder_engine_is_deterministic():
    engine = PlaceholderEngine()
    first = engine.answer("Where did Bjorn go after lunch?")
    second = engine.answer("Where did Bjorn go after lunch?")
    assert first.predicted_choice == second.predicted_choice
    assert first.support_priors == second.support_priors
    assert [e.record_id for e in first.evidence] == [
        e.record_id for e in second.evidence
    ]


def test_placeholder_engine_evidence_maps_to_mirror_embeds():
    mirror = YouTubeMirror.from_csv()
    engine = PlaceholderEngine.from_mirror(mirror)
    result = engine.answer("What was happening when the timer went off?")
    for item in result.evidence:
        url = mirror.embed_url(item.day, item.camera_id, item.hour, item.start_seconds)
        assert "/embed/" in url


# ---------------------------------------------------------------------------
# Figures (require plotly)
# ---------------------------------------------------------------------------


def test_figures_build_from_engine_output():
    go = pytest.importorskip("plotly.graph_objects")
    from castlerag.ui.figures import (
        empty_figure,
        evidence_timeline,
        modality_breakdown,
        support_bar,
    )

    result = PlaceholderEngine().answer("What color was the mug Allie held?")
    rows = [vars(item) for item in result.evidence]
    assert isinstance(evidence_timeline(rows), go.Figure)
    assert isinstance(modality_breakdown(rows), go.Figure)
    assert isinstance(support_bar(result.support_priors), go.Figure)
    assert isinstance(empty_figure(), go.Figure)
    # Empty inputs degrade to a placeholder figure rather than raising.
    assert isinstance(evidence_timeline([]), go.Figure)
    assert isinstance(support_bar({}), go.Figure)


# ---------------------------------------------------------------------------
# App factory (requires dash)
# ---------------------------------------------------------------------------


def test_build_app_assembles_layout_and_callbacks():
    pytest.importorskip("dash")
    from castlerag.ui.app import build_app

    app = build_app()
    assert app.layout is not None
    assert app.callback_map  # callbacks registered
