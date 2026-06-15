"""Plotly figure builders for the CastleRAG dashboard.

These satisfy the course's Plotly/Dash requirement and form the seam for the
point-2 multimodal-analytics widgets.  Every builder takes JSON-friendly inputs
(lists of evidence dicts, a priors dict) so it can be driven directly from a
Dash ``dcc.Store`` and reused once real retrieval traces replace the placeholder
engine output.
"""

from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go

_SOURCE_COLORS = {
    "transcript_window": "#4C78A8",
    "main_clip": "#F58518",
    "main_event_summary": "#54A24B",
    "aux_photo": "#E45756",
    "aux_video": "#B279A2",
}
_TEMPLATE = "plotly_white"


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce a JSON-store value to float, falling back to ``default``."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def empty_figure(message: str = "Ask a question to populate evidence") -> go.Figure:
    """Return a blank placeholder figure carrying a centered message."""
    fig = go.Figure()
    fig.update_layout(
        template=_TEMPLATE,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=13, color="#888"),
            )
        ],
    )
    return fig


def evidence_timeline(evidence: List[Dict[str, object]]) -> go.Figure:
    """Gantt-style timeline of evidence clips, one bar per row, by camera."""
    if not evidence:
        return empty_figure("No evidence yet")
    fig = go.Figure()
    for row in evidence:
        start = _as_float(row.get("start_seconds"), 0.0)
        end = _as_float(row.get("end_seconds"), start + 30.0)
        source_type = str(row.get("source_type", "main_clip"))
        camera = str(row.get("camera_id", "?"))
        fig.add_trace(
            go.Bar(
                x=[end - start],
                y=[f"{row.get('day', '?')} · {camera}"],
                base=[start],
                orientation="h",
                name=source_type,
                marker_color=_SOURCE_COLORS.get(source_type, "#888"),
                hovertemplate=(
                    f"{camera} ({source_type})<br>"
                    f"start={start:.0f}s end={end:.0f}s<br>"
                    f"score={row.get('score', 0)}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        template=_TEMPLATE,
        title="Evidence timeline (within source hour)",
        xaxis_title="seconds into hour",
        margin=dict(l=20, r=20, t=50, b=30),
        barmode="overlay",
        height=260,
    )
    return fig


def support_bar(support_priors: Dict[str, float]) -> go.Figure:
    """Bar chart of per-choice support priors, highlighting the argmax choice."""
    choices = ["a", "b", "c", "d"]
    values = [float(support_priors.get(key, 0.0)) for key in choices]
    if not any(values):
        return empty_figure("No support priors yet")
    best = max(range(len(values)), key=lambda i: values[i])
    colors = ["#54A24B" if i == best else "#B7C6D9" for i in range(len(values))]
    fig = go.Figure(
        go.Bar(
            x=[c.upper() for c in choices],
            y=values,
            marker_color=colors,
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Answer support priors",
        yaxis_title="support",
        yaxis_range=[0, 1],
        margin=dict(l=20, r=20, t=50, b=30),
        height=260,
    )
    return fig


def modality_breakdown(evidence: List[Dict[str, object]]) -> go.Figure:
    """Bar chart of evidence counts grouped by source type."""
    if not evidence:
        return empty_figure("No evidence yet")
    counts: Dict[str, int] = {}
    for row in evidence:
        source_type = str(row.get("source_type", "unknown"))
        counts[source_type] = counts.get(source_type, 0) + 1
    labels = list(counts.keys())
    fig = go.Figure(
        go.Bar(
            x=[counts[label] for label in labels],
            y=labels,
            orientation="h",
            marker_color=[_SOURCE_COLORS.get(label, "#888") for label in labels],
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Evidence by source type",
        xaxis_title="count",
        margin=dict(l=20, r=20, t=50, b=30),
        height=260,
    )
    return fig
