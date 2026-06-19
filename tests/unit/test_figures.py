"""Tests for the evidence-viewer Plotly figure."""

from __future__ import annotations

from castlerag.ui.figures import _ACCENT, _MUTED, camera_match_figure, empty_figure


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
