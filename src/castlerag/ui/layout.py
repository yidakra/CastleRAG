"""Dash layout for the CastleRAG dashboard shell.

A two-column shell: a chat panel on the left, and on the right a YouTube embed
of the selected evidence clip plus a row of placeholder Plotly analytics panels.
Stateful pieces (conversation history, current evidence) live in ``dcc.Store``
components so the callbacks stay pure.
"""

from __future__ import annotations

from dash import dcc, html

from castlerag.ui.figures import empty_figure
from castlerag.ui.youtube import YouTubeMirror

_CHOICE_KEYS = ("a", "b", "c", "d")


def _choice_inputs() -> html.Div:
    """Return the optional four-choice input row for multiple-choice questions."""
    return html.Div(
        className="choice-grid",
        children=[
            dcc.Input(
                id=f"choice-{key}",
                type="text",
                placeholder=f"Choice {key.upper()} (optional)",
                className="choice-input",
                debounce=True,
            )
            for key in _CHOICE_KEYS
        ],
    )


def _chat_panel() -> html.Div:
    """Return the left-hand chat panel (history, question, choices, submit)."""
    return html.Div(
        className="chat-panel",
        children=[
            html.H2("Chat", className="panel-title"),
            html.Div(
                id="chat-history",
                className="chat-history",
                children=[
                    html.Div(
                        "Ask a question about the CASTLE recordings to begin.",
                        className="chat-hint",
                    )
                ],
            ),
            dcc.Textarea(
                id="question-input",
                placeholder="e.g. What did Allie say before entering the kitchen?",
                className="question-input",
            ),
            _choice_inputs(),
            html.Button(
                "Send", id="send-button", n_clicks=0, className="send-button"
            ),
        ],
    )


def _analytics_panel() -> html.Div:
    """Return the Plotly analytics row (timeline, support priors, modality mix)."""
    return html.Div(
        className="analytics-row",
        children=[
            dcc.Graph(id="evidence-timeline", figure=empty_figure()),
            dcc.Graph(id="support-bar", figure=empty_figure()),
            dcc.Graph(id="modality-breakdown", figure=empty_figure()),
        ],
    )


def _video_panel(initial_src: str) -> html.Div:
    """Return the YouTube embed panel and the clickable evidence list."""
    return html.Div(
        className="video-panel",
        children=[
            html.H2("Evidence video", className="panel-title"),
            html.Div(id="video-caption", className="video-caption", children="—"),
            html.Iframe(
                id="video-embed",
                src=initial_src,
                className="video-embed",
                allow="accelerometer; encrypted-media; picture-in-picture",
            ),
            html.H3("Evidence rows", className="panel-subtitle"),
            html.Div(id="evidence-list", className="evidence-list"),
        ],
    )


def build_layout(mirror: YouTubeMirror) -> html.Div:
    """Build the full dashboard layout, seeding the embed with a default clip."""
    initial_src = mirror.default_embed_url()
    return html.Div(
        className="app-root",
        children=[
            html.Header(
                className="app-header",
                children=[
                    html.H1("CastleRAG", className="app-title"),
                    html.Span(
                        "multimodal RAG dashboard · placeholder backbone",
                        className="app-subtitle",
                    ),
                ],
            ),
            html.Div(
                className="app-body",
                children=[_chat_panel(), _video_panel(initial_src), _analytics_panel()],
            ),
            dcc.Store(id="conversation-store", data=[]),
            dcc.Store(id="evidence-store", data=[]),
        ],
    )
