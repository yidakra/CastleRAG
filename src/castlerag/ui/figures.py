"""Plotly figure for the CastleRAG evidence viewer.

One compact horizontal bar chart comparing the focused moment's synchronized
cameras by match score, with the best-matching angle drawn in the dashboard
accent colour.  Deliberately small: the viewer is camera-first, so the chart is
a glanceable score summary that sits under the camera grid and refreshes
whenever a moment is focused.

Themed to match ``assets/styles.css`` (light panels, indigo accent).  Consumed
by :func:`castlerag.ui.callbacks._viewer_outputs`; the offline ``PlaceholderEngine``
and the real ``RagEngine`` both feed it the same serialized-moment dict.
"""

from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go

# Palette aligned with assets/styles.css :root variables.
_ACCENT = "#4f46e5"  # best camera
_MUTED = "#cbcbd6"  # other cameras
_TEXT = "#3a3a44"  # --text-soft
_FAINT = "#9a9aa4"  # --faint (empty-state copy)
_GRID = "rgba(20, 20, 40, 0.06)"
_MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"


def _base_layout() -> go.Layout:
    """Return the shared transparent, tight-margin layout for viewer figures."""
    return go.Layout(
        height=148,
        margin=dict(l=8, r=16, t=8, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_TEXT, family=_MONO, size=12),
        showlegend=False,
        bargap=0.45,
    )


def empty_figure(
    message: str = "Select a moment to compare camera match scores",
) -> go.Figure:
    """Return a placeholder figure shown before any moment is focused."""
    fig = go.Figure()
    fig.update_layout(_base_layout())
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color=_FAINT, size=12),
    )
    return fig


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]`` for display."""
    return max(low, min(high, value))


def camera_match_figure(moment: Dict[str, object]) -> go.Figure:
    """Build the per-camera match-score bar chart for one focused moment.

    ``moment`` is the serialized store dict with a ``cameras`` list of
    ``{camera_id, match_score, is_best, ...}``.  Cameras are drawn lowest-first
    so the strongest angle reads at the top; the best angle uses the accent
    colour, the rest a muted grey.
    """
    cameras: List[Dict[str, object]] = list(moment.get("cameras", []))  # type: ignore[arg-type]
    if not cameras:
        return empty_figure()

    # Lowest score at the bottom -> strongest camera ends up on top.
    ordered = sorted(cameras, key=lambda c: float(c.get("match_score", 0.0)))
    names = [str(c.get("camera_id", "?")) for c in ordered]
    scores = [_clamp(float(c.get("match_score", 0.0))) for c in ordered]
    colors = [_ACCENT if c.get("is_best") else _MUTED for c in ordered]
    labels = [f"{s:.2f}" for s in scores]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=colors,
            text=labels,
            textposition="outside",
            textfont=dict(color=_TEXT, size=11),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(_base_layout())
    fig.update_xaxes(
        range=[0, 1.08],
        showgrid=True,
        gridcolor=_GRID,
        zeroline=False,
        tickformat=".1f",
        tickfont=dict(color=_FAINT, size=10),
    )
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(color=_TEXT))
    return fig
