"""Plotly figures for the CastleRAG evidence viewer.

Two visual panels per focused moment:
* A horizontal bar chart comparing camera match scores.
* A cross-camera agreement heatmap showing pairwise score proximity.

A compact pipeline funnel chart lives in each thread card and shows how many
clips survived each stage of the retrieval pipeline.

Themed to match ``assets/styles.css`` (light panels, indigo accent).
"""

from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Palette aligned with assets/styles.css :root variables.
_ACCENT = "#4f46e5"  # best camera / most-curated stage
_MUTED = "#cbcbd6"   # other cameras
_TEXT = "#3a3a44"    # --text-soft
_FAINT = "#9a9aa4"   # --faint (empty-state copy, axis ticks)
_GRID = "rgba(20, 20, 40, 0.06)"
_MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"

# Heatmap: light lavender → mid-indigo → accent (agreement 0 → 0.5 → 1.0).
_HEATMAP_SCALE = [[0.0, "#eeeef8"], [0.5, "#9590dc"], [1.0, _ACCENT]]


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
    """Build a combined bar chart + cross-camera agreement heatmap.

    ``moment`` is the serialized store dict with a ``cameras`` list of
    ``{camera_id, match_score, is_best, ...}``.  Cameras are drawn lowest-first
    so the strongest angle reads at the top; the best angle uses the accent
    colour.

    When ≥ 2 cameras are present a second panel shows a pairwise agreement
    heatmap: ``agreement(i, j) = 1 - |score_i - score_j|``.  The diagonal is
    always 1.0; off-diagonal cells show how closely two cameras corroborate each
    other's relevance signal.
    """
    cameras: List[Dict[str, object]] = list(moment.get("cameras", []))  # type: ignore[arg-type]
    if not cameras:
        return empty_figure()

    # --- bar chart data (all cameras, sorted lowest → top) ---
    ordered = sorted(cameras, key=lambda c: float(c.get("match_score", 0.0)))
    bar_names = [str(c.get("camera_id", "?")) for c in ordered]
    bar_scores = [_clamp(float(c.get("match_score", 0.0))) for c in ordered]
    bar_colors = [_ACCENT if c.get("is_best") else _MUTED for c in ordered]
    bar_labels = [f"{s:.2f}" for s in bar_scores]

    bar_trace = go.Bar(
        x=bar_scores,
        y=bar_names,
        orientation="h",
        marker_color=bar_colors,
        text=bar_labels,
        textposition="outside",
        textfont=dict(color=_TEXT, size=11),
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    )

    n = len(cameras)
    if n < 2:
        # Single camera: keep the original compact layout.
        fig = go.Figure(bar_trace)
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

    # --- agreement matrix (all cameras, pairwise) ---
    mat_names = [str(c.get("camera_id", "?")) for c in cameras]
    mat_scores = [_clamp(float(c.get("match_score", 0.0))) for c in cameras]
    z = [
        [1.0 - abs(mat_scores[i] - mat_scores[j]) for j in range(n)]
        for i in range(n)
    ]
    z_text = [[f"{z[i][j]:.2f}" for j in range(n)] for i in range(n)]

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.52, 0.48],
        vertical_spacing=0.18,
        subplot_titles=["Camera match scores", "Cross-camera agreement"],
    )
    fig.add_trace(bar_trace, row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=mat_names,
            y=mat_names,
            colorscale=_HEATMAP_SCALE,
            zmin=0.0,
            zmax=1.0,
            showscale=False,
            text=z_text,
            texttemplate="%{text}",
            textfont=dict(size=10, color=_TEXT),
            hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=320,
        margin=dict(l=8, r=16, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_TEXT, family=_MONO, size=12),
        showlegend=False,
        bargap=0.45,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(color=_FAINT, size=10, family=_MONO)
    fig.update_xaxes(
        range=[0, 1.08],
        showgrid=True,
        gridcolor=_GRID,
        zeroline=False,
        tickformat=".1f",
        tickfont=dict(color=_FAINT, size=10),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color=_TEXT),
        row=1,
        col=1,
    )
    fig.update_xaxes(tickfont=dict(color=_TEXT, size=10), side="bottom", row=2, col=1)
    fig.update_yaxes(
        tickfont=dict(color=_TEXT, size=10),
        autorange="reversed",
        row=2,
        col=1,
    )
    return fig


def pipeline_funnel_figure(stats: Dict[str, int]) -> go.Figure:
    """Compact horizontal funnel showing clip counts at each pipeline stage.

    Bars are listed bottom-to-top so ``Retrieved`` (widest) reads at the top
    and ``Displayed`` (narrowest) at the bottom — a visual funnel.  Color
    darkens toward the most-curated stage to draw the eye to what the user sees.
    """
    # Listed bottom → top (Plotly renders first item at the bottom of the y-axis).
    stages = ["Displayed", "Candidates", "Reranked", "Retrieved"]
    counts = [
        stats.get("displayed", 0),
        stats.get("candidates", 0),
        stats.get("reranked", 0),
        stats.get("retrieved", 0),
    ]
    colors = [_ACCENT, "#7c75eb", "#a5a0f3", "#c8c5f7"]
    retrieved = max(counts[-1], 1)

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=stages,
            orientation="h",
            marker_color=colors,
            text=[str(c) for c in counts],
            textposition="outside",
            textfont=dict(color=_TEXT, size=10),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        )
    )
    fig.update_layout(
        height=108,
        margin=dict(l=8, r=36, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_TEXT, family=_MONO, size=10),
        showlegend=False,
        bargap=0.30,
    )
    fig.update_xaxes(
        range=[0, retrieved * 1.25],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, tickfont=dict(color=_TEXT, size=10)
    )
    return fig
