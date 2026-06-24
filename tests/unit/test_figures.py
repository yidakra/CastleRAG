"""Tests for the evidence-viewer Plotly figures."""

from __future__ import annotations

from castlerag.ui.figures import (
    _ACCENT,
    _MUTED,
    camera_match_figure,
    empty_figure,
    pipeline_funnel_figure,
)


def _moment(cameras):
    return {"moment_id": "m0", "clock_label": "12:29", "cameras": cameras}


def test_empty_figure_has_no_bars_and_a_message():
    fig = empty_figure("nothing yet")
    assert fig.data == ()  # no bar trace
    assert any(a.text == "nothing yet" for a in fig.layout.annotations)


def test_camera_match_figure_plots_scores_best_first():
    moment = _moment(
        [
            {"camera_id": "Bjorn", "match_score": 0.42, "is_best": False},
            {"camera_id": "Luca", "match_score": 0.81, "is_best": True},
            {"camera_id": "Klaus", "match_score": 0.55, "is_best": False},
        ]
    )
    fig = camera_match_figure(moment)
    bar = fig.data[0]
    # Lowest score first so the strongest camera renders on top.
    assert list(bar.y) == ["Bjorn", "Klaus", "Luca"]
    assert list(bar.x) == [0.42, 0.55, 0.81]
    # The best camera (Luca, last after sorting) is drawn in the accent colour.
    assert bar.marker.color == (_MUTED, _MUTED, _ACCENT)


def test_camera_match_figure_clamps_scores_to_unit_range():
    moment = _moment(
        [
            {"camera_id": "Luca", "match_score": 1.4, "is_best": True},
            {"camera_id": "Klaus", "match_score": -0.2, "is_best": False},
        ]
    )
    bar = camera_match_figure(moment).data[0]
    assert list(bar.x) == [0.0, 1.0]


def test_camera_match_figure_without_cameras_is_empty():
    fig = camera_match_figure(_moment([]))
    assert fig.data == ()


def test_camera_match_figure_with_multi_cameras_includes_heatmap():
    moment = _moment(
        [
            {"camera_id": "A", "match_score": 0.9, "is_best": True},
            {"camera_id": "B", "match_score": 0.6, "is_best": False},
            {"camera_id": "C", "match_score": 0.3, "is_best": False},
        ]
    )
    fig = camera_match_figure(moment)
    trace_types = [type(t).__name__ for t in fig.data]
    assert "Bar" in trace_types
    assert "Heatmap" in trace_types
    heatmap = next(t for t in fig.data if type(t).__name__ == "Heatmap")
    # Diagonal must be 1.0 (a camera agrees with itself).
    for i in range(3):
        assert abs(heatmap.z[i][i] - 1.0) < 1e-9


def test_pipeline_funnel_figure_bars_in_funnel_order():
    stats = {"retrieved": 150, "reranked": 40, "candidates": 12, "displayed": 3}
    fig = pipeline_funnel_figure(stats)
    bar = fig.data[0]
    # y list must be bottom-to-top: Displayed at bottom, Retrieved at top.
    assert list(bar.y) == ["Displayed", "Candidates", "Reranked", "Retrieved"]
    assert list(bar.x) == [3, 12, 40, 150]


def test_pipeline_funnel_figure_empty_stats_does_not_crash():
    fig = pipeline_funnel_figure({})
    assert fig.data[0].x[0] == 0  # Displayed count is 0
