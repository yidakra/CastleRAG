"""Dash layout for the CastleRAG claim-verification dashboard.

A two-column workspace that persists across the whole task:

* **Left** — a scrollable thread of *query groups*.  Each group shows the
  question, a short answer, the single claim under review with a support badge,
  and a ranked list of evidence moments.  Refinements append new groups; an
  "Ask a new question" bar starts a fresh thread.
* **Right** — a pinned evidence viewer for the focused moment: three
  synchronized camera embeds, a per-camera confirm/refine/reject review row, and
  a compose box for sending a refined query.

The thread, viewer, camera grid, and review row are rendered by callbacks; their
state lives in ``dcc.Store`` components so the callbacks stay pure.
"""

from __future__ import annotations

from dash import dcc, html

from castlerag.ui.youtube import YouTubeMirror


def _top_bar(mode: str) -> html.Header:
    """Return the static top bar with the CastleRAG mark and status chips.

    ``mode`` is ``"live"`` (real RAG backend) or ``"offline"`` (placeholder).
    """
    live = mode == "live"
    return html.Header(
        className="top-bar",
        children=[
            html.Span(className="brand-mark"),
            html.Span("CastleRAG", className="brand-name"),
            html.Div(
                className="top-chips",
                children=[
                    html.Span("Day 1", className="chip"),
                    html.Span("15 cams", className="chip"),
                    html.Span(
                        "live RAG" if live else "offline",
                        className="chip mode-live" if live else "chip mode-offline",
                    ),
                ],
            ),
        ],
    )


def _thread_column() -> html.Div:
    """Return the left column: the scrollable thread and the ask-new bar."""
    return html.Div(
        className="thread-col",
        children=[
            html.Div(
                id="thread",
                className="thread",
                children=[
                    html.Div(
                        "Ask a question about the CASTLE recordings to begin "
                        "an investigation.",
                        className="thread-hint",
                    )
                ],
            ),
            html.Div(
                className="ask-new",
                children=[
                    dcc.Textarea(
                        id="new-question-input",
                        placeholder="Ask a new question…",
                        className="ask-new-input",
                    ),
                    html.Button(
                        "New query",
                        id="ask-new-button",
                        n_clicks=0,
                        className="ask-new-button",
                    ),
                ],
            ),
        ],
    )


def _viewer_column() -> html.Div:
    """Return the right column: the pinned evidence viewer for the focus moment."""
    return html.Div(
        className="viewer-col",
        children=[
            html.Div(
                className="viewer-head",
                children=[
                    html.Span(
                        "Selected moment", id="viewer-title", className="viewer-title"
                    ),
                    html.Span("", id="viewer-subtitle", className="viewer-subtitle"),
                ],
            ),
            html.Div(
                id="camera-grid",
                className="camera-grid",
                children=[
                    html.Div(
                        "Select an evidence moment to see its synchronized cameras.",
                        className="viewer-hint",
                    )
                ],
            ),
            html.Div(id="review-row", className="review-row"),
            html.Div(
                id="compose-wrap",
                className="compose-box",
                hidden=True,
                children=[
                    html.Div(
                        "Refine the query · re-run retrieval for this claim",
                        className="compose-label",
                    ),
                    dcc.Textarea(
                        id="refined-query-input",
                        placeholder="Describe a sharper angle for the same claim…",
                        className="compose-input",
                    ),
                    html.Button(
                        "↑ Send refined query",
                        id="send-refined-button",
                        n_clicks=0,
                        className="compose-button",
                    ),
                ],
            ),
            html.Div(id="converged-banner", className="converged-banner", hidden=True),
        ],
    )


def build_layout(mirror: YouTubeMirror, mode: str = "offline") -> html.Div:
    """Build the full dashboard layout (state lives in the stores below)."""
    return html.Div(
        className="app-root",
        children=[
            _top_bar(mode),
            html.Div(
                className="app-body",
                children=[_thread_column(), _viewer_column()],
            ),
            dcc.Store(id="thread-store", data=[]),
            dcc.Store(id="focus-store", data={}),
            dcc.Store(id="review-store", data={}),
            dcc.Store(id="iteration-store", data={"claim": None, "iteration": 0}),
        ],
    )
