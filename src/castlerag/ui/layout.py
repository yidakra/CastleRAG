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

import dash_mantine_components as dmc
from dash import dcc, html

from castlerag.ui.figures import empty_figure
from castlerag.ui.youtube import YouTubeMirror


def _top_bar(mode: str) -> dmc.Group:
    """Return the static top bar with the CastleRAG mark and status badges.

    ``mode`` is ``"live"`` (real RAG backend) or ``"offline"`` (placeholder).
    """
    live = mode == "live"
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
                    dmc.Badge("Day 1", variant="light", color="gray"),
                    dmc.Badge("15 cams", variant="light", color="gray"),
                    dmc.Badge(
                        "live RAG" if live else "offline",
                        variant="light" if live else "outline",
                        color="teal" if live else "gray",
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
            # #thread owns its scroll via max-height + overflow-y (in styles.css),
            # which does NOT depend on flex-height propagation — so it scrolls no
            # matter what wraps it. That lets us put it back INSIDE dcc.Loading,
            # which is what actually shows the "retrieving" spinner over the thread
            # while a slow callback updates thread.children. delay_show avoids a
            # flicker on fast re-renders (e.g. moment clicks).
            dcc.Loading(
                type="circle",
                color="#4f46e5",
                delay_show=250,
                children=html.Div(
                    id="thread", className="thread", children=[_thread_hint()]
                ),
            ),
            dmc.Group(
                className="ask-new",
                gap="sm",
                align="flex-end",
                wrap="nowrap",
                children=[
                    # dcc.Input (not a DMC input) so pressing Enter fires n_submit.
                    dcc.Input(
                        id="new-question-input",
                        type="text",
                        placeholder="Ask a new question…  (press Enter)",
                        className="ask-new-input",
                        debounce=False,
                        n_submit=0,
                    ),
                    dmc.Button(
                        "New query",
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


def _viewer_column() -> html.Div:
    """Return the right column: the pinned evidence viewer for the focus moment."""
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
                    dmc.Text("Camera match scores", size="sm", fw=600, mb=4),
                    dcc.Graph(
                        id="evidence-figure",
                        className="evidence-figure",
                        figure=empty_figure(),
                        config={"displayModeBar": False, "staticPlot": False},
                    ),
                ],
            ),
            html.Div(id="review-row", className="review-row"),
            # html.Div (not DMC) so the callback can toggle the `hidden` attribute.
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
                                "Refine the query · re-run retrieval for this claim",
                                size="sm",
                                fw=600,
                                mb=6,
                            ),
                            dcc.Textarea(
                                id="refined-query-input",
                                placeholder=(
                                    "Describe a sharper angle for the same claim…"
                                ),
                                className="compose-input",
                            ),
                            dmc.Button(
                                "↑ Send refined query",
                                id="send-refined-button",
                                n_clicks=0,
                                variant="filled",
                                mt="sm",
                            ),
                        ],
                    )
                ],
            ),
            html.Div(id="converged-banner", hidden=True),
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
