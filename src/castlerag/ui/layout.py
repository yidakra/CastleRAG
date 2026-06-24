"""Dash layout for the CastleRAG claim-verification dashboard.

Built with Dash Mantine Components (DMC): every visual surface — the top bar,
cards, badges, buttons, inputs — is a themed DMC component (see
:data:`castlerag.ui.app.THEME`).  A thin layer of structural CSS in
``assets/styles.css`` still owns the two-column flex skeleton, the scrollable
thread, and the camera/review grids.

* **Left** — a scrollable thread of *query groups*: question, answer, the claim
  under review with a support badge, and a ranked list of evidence moments.
* **Right** — a pinned evidence viewer for the focused moment: synchronized
  camera embeds, a Plotly match-score chart, a per-camera review row, and a
  compose box for refining the query.

The thread, viewer, camera grid, and review row are rendered by callbacks; their
state lives in ``dcc.Store`` components so the callbacks stay pure.  Value- and
``n_submit``-bound text inputs stay as ``dcc`` controls for their stable Dash
contracts; everything else is DMC.
"""

from __future__ import annotations

from typing import Optional

import dash_mantine_components as dmc
from dash import dcc, html

from castlerag.ui.figures import empty_figure
from castlerag.ui.youtube import YouTubeMirror

_SCORE_MODE_TOOLTIP: dict[str, str] = {
    "rrf_normalized": (
        "RRF-normalised rank score (score_mode = rrf_normalized). "
        "Each retrieval pass contributes 1 / (60 + rank) per document; scores "
        "sum across BM25, dense transcript, and dense multimodal passes. "
        "The raw RRF score is then divided by the top hit's score for this query, "
        "so the best moment is always 1.0 and the rest are relative to it. "
        "Not comparable across different queries."
    ),
    "cosine": (
        "Raw cosine similarity (score_mode = cosine). "
        "Computed between the OmniEmbed query vector and each stored clip or "
        "transcript embedding before rank fusion. Range [0, 1]: 1.0 = identical "
        "embedding directions, 0.0 = orthogonal. "
        "Values ≥ 0.7 typically indicate strong semantic match; ≤ 0.4 is weak. "
        "Comparable across queries — not normalised per-query."
    ),
    "reranker": (
        "Reranker relevance score (score_mode = reranker). "
        "Qwen3-VL-8B rates each evidence pack on two 0–4 Likert axes: "
        "relevance to the question and per-choice support. "
        "Final score = (0.7 × relevance + 0.3 × max_support) / 4, "
        "normalised to [0, 1]. "
        "Falls back to RRF-normalised when the reranker did not run (offline mode)."
    ),
}


def _top_bar(mode: str, cfg: Optional[object] = None) -> dmc.Group:
    """Return the top bar with the CastleRAG mark, status badges, and clear button.

    ``mode`` is ``"live"`` (real RAG backend) or ``"offline"`` (placeholder).
    Badges are derived from ``cfg`` when available, otherwise fall back to
    hardcoded defaults.
    """
    live = mode == "live"

    if cfg is not None:
        ds = getattr(cfg, "dataset", None)
        days: list = getattr(ds, "days", [1]) if ds else [1]
        scope: str = getattr(ds, "camera_scope", "ego") if ds else "ego"
        ego_cams: list = getattr(ds, "ego_cameras", []) if ds else []
        exo_cams: list = getattr(ds, "exo_cameras", []) if ds else []
        n_cams = len(ego_cams) + (len(exo_cams) if scope == "all" else 0)
        day_label = f"Day {days[0]}" if len(days) == 1 else f"Days {days[0]}–{days[-1]}"
        cam_label = f"{n_cams} cams"
    else:
        day_label = "Day 1"
        cam_label = "15 cams"

    return dmc.Group(
        className="top-bar",
        justify="space-between",
        children=[
            dmc.Group(
                gap="xs",
                children=[
                    html.Span(className="brand-mark"),
                    dmc.Text("CastleRAG", fw=700, size="lg"),
                ],
            ),
            dmc.Group(
                gap="xs",
                children=[
                    dmc.Badge(day_label, variant="light", color="gray"),
                    dmc.Badge(cam_label, variant="light", color="gray"),
                    dmc.Badge(
                        "live RAG" if live else "offline",
                        variant="light" if live else "outline",
                        color="teal" if live else "gray",
                    ),
                    dmc.Button(
                        "Clear",
                        id="clear-thread-button",
                        variant="subtle",
                        color="gray",
                        size="xs",
                        n_clicks=0,
                    ),
                ],
            ),
        ],
    )


def _thread_column() -> html.Div:
    """Return the left column: the scrollable thread and the ask-new bar.

    The outer container stays a plain ``html.Div`` (not ``dmc.Stack``) so the
    flex/scroll skeleton in ``styles.css`` reliably constrains the thread's
    height and lets it scroll.
    """
    return html.Div(
        className="thread-col",
        children=[
            # thread owns the scroll directly. Keep it as a plain html.Div so the
            # left column can stay fixed-height while only the thread content scrolls.
            html.Div(
                id="thread",
                className="thread",
                children=[_thread_hint()],
            ),

            dmc.Group(
                className="ask-new",
                gap="sm",
                align="flex-end",
                wrap="nowrap",
                children=[
                    # dcc.Textarea expands as the user types (auto-resize via
                    # query-input.js); Enter submits, Shift+Enter inserts a newline.
                    dcc.Textarea(
                        id="new-question-input",
                        placeholder="Ask a question",
                        className="ask-new-input",
                    ),
                    dmc.Button(
                        "Send",
                        id="ask-new-button",
                        n_clicks=0,
                        variant="filled",
                    ),
                ],
            ),
        ],
    )


def _thread_hint() -> dmc.Text:
    """Return the empty-thread hint shown before any question is asked."""
    return dmc.Text(
        "Ask a question about the CASTLE recordings to begin an investigation.",
        className="thread-hint",
        c="dimmed",
        size="sm",
    )


def _viewer_column(score_mode: str = "rrf_normalized") -> html.Div:
    """Return the right column: the pinned evidence viewer for the focus moment."""
    tooltip_text = _SCORE_MODE_TOOLTIP.get(
        score_mode, _SCORE_MODE_TOOLTIP["rrf_normalized"]
    )
    return html.Div(
        className="viewer-col",
        children=[
            dmc.Group(
                className="viewer-head",
                justify="space-between",
                children=[
                    dmc.Text("Selected moment", id="viewer-title", fw=600),
                    dmc.Text("", id="viewer-subtitle", c="dimmed", size="sm"),
                ],
            ),
            html.Div(
                id="camera-grid",
                className="camera-grid",
                children=[
                    dmc.Text(
                        "Select an evidence moment to see its synchronized cameras.",
                        className="viewer-hint",
                        c="dimmed",
                        size="sm",
                    )
                ],
            ),
            dmc.Paper(
                className="evidence-figure-wrap",
                withBorder=True,
                p="sm",
                children=[
                    dmc.Group(
                        gap=4,
                        align="center",
                        mb=4,
                        children=[
                            dmc.Text("Camera match scores", size="sm", fw=600),
                            dmc.Tooltip(
                                label=tooltip_text,
                                multiline=True,
                                w=380,
                                withArrow=True,
                                position="top-start",
                                children=dmc.Text(
                                    "ⓘ",
                                    size="sm",
                                    c="dimmed",
                                    style={"cursor": "default", "lineHeight": 1},
                                ),
                            ),
                        ],
                    ),
                    dcc.Graph(
                        id="evidence-figure",
                        className="evidence-figure",
                        figure=empty_figure(),
                        config={"displayModeBar": False, "staticPlot": False},
                    ),
                ],
            ),
            # Spinner covers the review controls + compose box while the engine
            # drafts justification/refined-query suggestions (and during retrieval).
            dcc.Loading(
                type="dot",
                className="review-loading",
                children=[
                    html.Div(id="review-row", className="review-row"),
                    # Appears once all three cameras have a verdict; clicking it
                    # commits the reviews and drafts the refined query.
                    html.Div(
                        id="submit-wrap",
                        hidden=True,
                        className="submit-wrap",
                        children=[
                            dmc.Button(
                                "Submit reviews →",
                                id="submit-reviews-button",
                                n_clicks=0,
                                variant="filled",
                                color="indigo",
                                fullWidth=True,
                            ),
                        ],
                    ),
                    # html.Div (not DMC) so the callback can toggle `hidden`.
                    html.Div(
                        id="compose-wrap",
                        hidden=True,
                        children=[
                            dmc.Paper(
                                className="compose-box",
                                withBorder=True,
                                p="sm",
                                children=[
                                    dmc.Text(
                                        "Refine the query · re-run retrieval "
                                        "for this claim",
                                        size="sm",
                                        fw=600,
                                        mb=6,
                                    ),
                                    dcc.Textarea(
                                        id="refined-query-input",
                                        placeholder=(
                                            "Describe a sharper angle for the "
                                            "same claim…"
                                        ),
                                        className="compose-input",
                                    ),
                                    dmc.Button(
                                        "✓ Confirm & run refined search",
                                        id="send-refined-button",
                                        n_clicks=0,
                                        variant="filled",
                                        mt="sm",
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
            html.Div(id="converged-banner", hidden=True),
        ],
    )


def build_layout(
    mirror: YouTubeMirror,
    mode: str = "offline",
    score_mode: str = "rrf_normalized",
    cfg: Optional[object] = None,
) -> html.Div:
    """Build the full dashboard layout (state lives in the stores below)."""
    return html.Div(
        className="app-root",
        children=[
            _top_bar(mode, cfg),
            html.Div(
                className="app-body",
                children=[
                    _thread_column(),
                    # Draggable splitter between the chat and the evidence viewer;
                    # assets/splitter.js drives the resize, assets/styles.css the
                    # look.
                    html.Div(
                        className="app-gutter",
                        id="app-gutter",
                        title="Drag to resize · double-click to reset",
                    ),
                    _viewer_column(score_mode),
                ],
            ),
            dcc.Store(id="thread-store", data=[]),
            dcc.Store(id="focus-store", data={}),
            dcc.Store(id="review-store", data={}),
            dcc.Store(
                id="iteration-store",
                data={"claim": None, "iteration": 0, "next_seq": 1},
            ),
            # Hidden sink used by the auto-focus clientside callback.
            html.Div(id="_focus-dummy", style={"display": "none"}),
        ],
    )
