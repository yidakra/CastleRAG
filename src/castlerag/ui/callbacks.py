"""Dash callbacks wiring the chat engine, YouTube mirror, and Plotly figures.

Two callbacks:

* ``on_send`` runs the :class:`~castlerag.ui.chat.ChatEngine` for a question,
  appends to the conversation, and refreshes the embed, evidence list, and all
  three figures.
* ``on_evidence_click`` re-seeks the embed to whichever evidence row the user
  clicks (pattern-matching callback).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

from dash import ALL, Input, Output, State, ctx, html
from dash.exceptions import PreventUpdate

from castlerag.ui.chat import ChatEngine, EvidenceRef
from castlerag.ui.figures import evidence_timeline, modality_breakdown, support_bar
from castlerag.ui.youtube import YouTubeMirror


def _format_clock(hour: int, start_seconds: float) -> str:
    """Return an ``HH:MM:SS`` clock label for an offset within the source hour."""
    minutes, seconds = divmod(int(start_seconds), 60)
    return f"{int(hour):02d}:{minutes:02d}:{seconds:02d}"


def _evidence_to_store(
    evidence: List[EvidenceRef], mirror: YouTubeMirror
) -> List[Dict[str, object]]:
    """Serialize evidence to dicts, precomputing embed URL and caption."""
    rows: List[Dict[str, object]] = []
    for item in evidence:
        row: Dict[str, object] = asdict(item)
        row["embed_url"] = mirror.embed_url(
            item.day, item.camera_id, item.hour, item.start_seconds
        )
        clock = _format_clock(item.hour, item.start_seconds)
        placeholder = " · placeholder mirror" if mirror.is_placeholder(
            item.day, item.camera_id, item.hour
        ) else ""
        row["caption"] = (
            f"{item.day} · {item.camera_id} · {clock} "
            f"({item.source_type}){placeholder}"
        )
        rows.append(row)
    return rows


def _render_history(messages: List[Dict[str, object]]) -> List[html.Div]:
    """Render conversation messages into styled chat bubbles."""
    if not messages:
        return [html.Div("No messages yet.", className="chat-hint")]
    bubbles: List[html.Div] = []
    for msg in messages:
        role = msg.get("role", "user")
        children: List[html.Div] = [
            html.Div(str(msg.get("text", "")), className="bubble-text")
        ]
        if role == "assistant":
            badge = f"route: {msg.get('route', '?')}"
            if msg.get("predicted"):
                badge += f" · predicted: {str(msg['predicted']).upper()}"
            children.append(html.Div(badge, className="bubble-badge"))
        bubbles.append(
            html.Div(children, className=f"chat-bubble chat-{role}")
        )
    return bubbles


def _render_evidence_items(rows: List[Dict[str, object]]) -> List[html.Button]:
    """Render evidence rows as clickable buttons that re-seek the embed."""
    items: List[html.Button] = []
    for index, row in enumerate(rows):
        label = (
            f"[{index + 1}] {row.get('caption', '')} — score "
            f"{row.get('score', 0)}"
        )
        items.append(
            html.Button(
                label,
                id={"type": "evidence-item", "index": index},
                n_clicks=0,
                className="evidence-item",
            )
        )
    return items


def register_callbacks(
    app: object, engine: ChatEngine, mirror: YouTubeMirror
) -> None:
    """Register the dashboard callbacks on ``app``."""

    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        Output("chat-history", "children"),
        Output("conversation-store", "data"),
        Output("evidence-store", "data"),
        Output("video-embed", "src"),
        Output("video-caption", "children"),
        Output("evidence-list", "children"),
        Output("evidence-timeline", "figure"),
        Output("support-bar", "figure"),
        Output("modality-breakdown", "figure"),
        Input("send-button", "n_clicks"),
        State("question-input", "value"),
        State("choice-a", "value"),
        State("choice-b", "value"),
        State("choice-c", "value"),
        State("choice-d", "value"),
        State("conversation-store", "data"),
        prevent_initial_call=True,
    )
    def on_send(
        n_clicks: int,
        question: Optional[str],
        choice_a: Optional[str],
        choice_b: Optional[str],
        choice_c: Optional[str],
        choice_d: Optional[str],
        conversation: Optional[List[Dict[str, object]]],
    ) -> tuple[object, ...]:
        """Run the engine for the submitted question and refresh the dashboard."""
        if not question or not question.strip():
            raise PreventUpdate

        choices = _collect_choices(choice_a, choice_b, choice_c, choice_d)
        result = engine.answer(question.strip(), choices)
        rows = _evidence_to_store(result.evidence, mirror)

        messages = list(conversation or [])
        messages.append({"role": "user", "text": question.strip()})
        messages.append(
            {
                "role": "assistant",
                "text": result.answer_text,
                "route": result.route,
                "predicted": result.predicted_choice,
            }
        )

        first = rows[0] if rows else None
        video_src = first["embed_url"] if first else mirror.embed_url(
            "day1", "Allie", 8, 0
        )
        caption = first["caption"] if first else "—"

        return (
            _render_history(messages),
            messages,
            rows,
            video_src,
            caption,
            _render_evidence_items(rows),
            evidence_timeline(rows),
            support_bar(result.support_priors),
            modality_breakdown(rows),
        )

    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        Output("video-embed", "src", allow_duplicate=True),
        Output("video-caption", "children", allow_duplicate=True),
        Input({"type": "evidence-item", "index": ALL}, "n_clicks"),
        State("evidence-store", "data"),
        prevent_initial_call=True,
    )
    def on_evidence_click(
        n_clicks: List[int], rows: Optional[List[Dict[str, object]]]
    ) -> tuple[object, object]:
        """Re-seek the embed to the clicked evidence row."""
        triggered = ctx.triggered_id
        if not triggered or not rows:
            raise PreventUpdate
        index = triggered["index"]
        if index < 0 or index >= len(rows):
            raise PreventUpdate
        row = rows[index]
        return row["embed_url"], row["caption"]


def _collect_choices(*values: Optional[str]) -> Optional[Dict[str, str]]:
    """Return a choices dict only when all four choices are provided."""
    keys = ("a", "b", "c", "d")
    cleaned = [(v or "").strip() for v in values]
    if all(cleaned):
        return dict(zip(keys, cleaned))
    return None
