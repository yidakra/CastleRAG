"""Dash callbacks wiring the chat engine, YouTube mirror, and evidence viewer.

Four callbacks drive the workspace:

* ``on_ask_new`` opens a fresh investigation thread for a new question.
* ``on_send_refined`` re-runs retrieval for the *same* claim with a sharper
  query, appending a new group to the thread (capped at five iterations).
* ``on_moment_click`` focuses an evidence moment, re-seeking the three
  synchronized camera embeds (pattern-matching callback).
* ``on_review_action`` records a per-camera confirm / refine / reject verdict and
  its justification (pattern-matching callback).

Thread state lives in ``dcc.Store``s; the moment store carries precomputed embed
URLs so the click/review callbacks never need the mirror.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from castlerag.ui.chat import ChatEngine, ChatTurnResult, compose_refined_query
from castlerag.ui.figures import camera_match_figure
from castlerag.ui.youtube import YouTubeMirror

_MAX_ITERATIONS = 5
_SUPPORT_LABEL = {
    "unsupported": "Unsupported",
    "partial": "Partial support",
    "supported": "Supported",
}
# Mantine theme colors for each support level (badges, etc.).
_SUPPORT_COLOR = {
    "unsupported": "red",
    "partial": "yellow",
    "supported": "green",
}
_REVIEW_ACTION_STATE = {
    "confirm": "confirmed",
    "refine": "flagged",
    "reject": "rejected",
}
# Mantine color per review action button.
_REVIEW_ACTION_COLOR = {"confirm": "green", "refine": "yellow", "reject": "red"}


# ---------------------------------------------------------------------------
# Serialization: turn an engine result into a JSON-friendly thread group
# ---------------------------------------------------------------------------


def _serialize_group(
    result: ChatTurnResult,
    *,
    group_id: str,
    iteration: int,
    question: str,
    mirror: YouTubeMirror,
    is_refinement: bool,
    refined_query: Optional[str] = None,
) -> Dict[str, object]:
    """Serialize an engine turn into a store-friendly group dict.

    Each camera angle gets its ``embed_url`` precomputed so the moment-click and
    review callbacks stay pure (no mirror needed).
    """
    moments: List[Dict[str, object]] = []
    for moment in result.moments:
        cameras = [
            {
                "camera_id": cam.camera_id,
                "day": cam.day,
                "hour": cam.hour,
                "start_seconds": cam.start_seconds,
                "match_score": cam.match_score,
                "is_best": cam.is_best,
                # None when the mirror has no real upload for this triple, so the
                # viewer shows a "no footage" tile instead of a placeholder video.
                "embed_url": (
                    None
                    if mirror.is_placeholder(cam.day, cam.camera_id, cam.hour)
                    else mirror.embed_url(
                        cam.day, cam.camera_id, cam.hour, cam.start_seconds
                    )
                ),
            }
            for cam in moment.cameras
        ]
        moments.append(
            {
                "moment_id": moment.moment_id,
                "clock_label": moment.clock_label,
                "place_label": moment.place_label,
                "camera_count": moment.camera_count,
                "aggregate_score": moment.aggregate_score,
                "score_caption": moment.score_caption,
                "dot_color": moment.dot_color,
                "cameras": cameras,
            }
        )
    claim = result.claim
    return {
        "group_id": group_id,
        "iteration": iteration,
        "question": question,
        "answer_text": result.answer_text,
        "claim": {
            "text": claim.text if claim else "",
            "support": claim.support.value if claim else "partial",
        },
        "is_refinement": is_refinement,
        "refined_query": refined_query,
        "moments": moments,
    }


def _find_group(
    thread: List[Dict[str, object]], group_id: str
) -> Optional[Dict[str, object]]:
    """Return the group with ``group_id`` from the thread, or ``None``."""
    return next((g for g in thread if g["group_id"] == group_id), None)


def _find_moment(
    group: Dict[str, object], moment_id: str
) -> Optional[Dict[str, object]]:
    """Return the moment with ``moment_id`` from a group, or ``None``."""
    return next(
        (m for m in group["moments"] if m["moment_id"] == moment_id), None  # type: ignore[index]
    )


def _pending_review(moment: Dict[str, object]) -> Dict[str, Dict[str, str]]:
    """Return an all-pending review state for a moment's cameras (in order)."""
    return {
        cam["camera_id"]: {"state": "pending", "justification": ""}
        for cam in moment["cameras"]  # type: ignore[index]
    }


def _focused_claim_text(
    thread: Optional[List[Dict[str, object]]],
    focus: Optional[Dict[str, object]],
) -> str:
    """Return the claim text of the focused group (empty string if not found)."""
    group = _find_group(thread or [], (focus or {}).get("group_id", ""))
    if group is None:
        return ""
    return str(group["claim"]["text"])  # type: ignore[index]


# ---------------------------------------------------------------------------
# Renderers (pure: read the store dicts, never the mirror)
# ---------------------------------------------------------------------------


def _render_moment(
    group_id: str, moment: Dict[str, object], focused: bool
) -> html.Button:
    """Render one ranked evidence moment as a clickable thread row.

    The clickable shell stays an ``html.Button`` (so the pattern-matching
    ``n_clicks`` callback keeps working); its content is DMC.
    """
    meta = f"{moment['camera_count']} cameras · {moment['score_caption']}"
    return html.Button(
        id={"type": "moment", "gid": group_id, "mid": moment["moment_id"]},
        n_clicks=0,
        className="moment focused" if focused else "moment",
        children=dmc.Group(
            justify="space-between",
            wrap="nowrap",
            w="100%",
            children=[
                dmc.Stack(
                    gap=2,
                    children=[
                        dmc.Text(
                            f"{moment['clock_label']} · {moment['place_label']}",
                            fw=600,
                            size="sm",
                        ),
                        dmc.Text(meta, size="xs", c="dimmed"),
                    ],
                ),
                dmc.Box(
                    w=9,
                    h=9,
                    style={
                        "background": moment["dot_color"],
                        "borderRadius": "50%",
                        "flex": "none",
                    },
                ),
            ],
        ),
    )


def _render_group(group: Dict[str, object], focus: Dict[str, object]) -> dmc.Card:
    """Render one query group (question, answer, claim, ranked moments)."""
    claim = group["claim"]  # type: ignore[index]
    support = str(claim["support"])
    focus_gid = focus.get("group_id")
    focus_mid = focus.get("moment_id")

    header: List[object] = []
    if group["is_refinement"]:
        header.append(
            dmc.Badge(
                f"↻ Refined · {group['iteration']} / {_MAX_ITERATIONS}",
                variant="light",
                color="indigo",
                mb="xs",
            )
        )

    moments = [
        _render_moment(
            str(group["group_id"]),
            moment,  # type: ignore[arg-type]
            focused=group["group_id"] == focus_gid
            and moment["moment_id"] == focus_mid,  # type: ignore[index]
        )
        for moment in group["moments"]  # type: ignore[index]
    ]

    return dmc.Card(
        className="group-card",
        withBorder=True,
        shadow="sm",
        radius="md",
        p="md",
        children=[
            *header,
            dmc.Group(
                gap="xs",
                wrap="nowrap",
                children=[
                    html.Span(className="question-icon"),
                    dmc.Text(str(group["question"]), fw=600, className="question-text"),
                ],
            ),
            dmc.Text("Answer", size="xs", c="dimmed", mt="sm"),
            dcc.Markdown(str(group["answer_text"]), className="answer-text"),
            dmc.Paper(
                className="claim-block",
                withBorder=True,
                radius="sm",
                p="sm",
                mt="sm",
                children=[
                    dmc.Text("Claim under review", size="xs", c="dimmed"),
                    dmc.Text(str(claim["text"]), mb="xs"),
                    dmc.Badge(
                        _SUPPORT_LABEL.get(support, support),
                        variant="dot",
                        color=_SUPPORT_COLOR.get(support, "gray"),
                    ),
                ],
            ),
            dmc.Text("Top evidence moments", size="xs", fw=600, mt="sm", mb="xs"),
            dmc.Stack(moments, gap="xs", className="moment-list"),
        ],
    )


def _render_thread(
    thread: List[Dict[str, object]], focus: Dict[str, object]
) -> List[object]:
    """Render the full thread; a hint when empty."""
    if not thread:
        return [
            dmc.Text(
                "Ask a question about the CASTLE recordings to begin an "
                "investigation.",
                className="thread-hint",
                c="dimmed",
                size="sm",
            )
        ]
    return [_render_group(group, focus) for group in thread]


def _render_camera_grid(moment: Dict[str, object]) -> List[dmc.Card]:
    """Render the synchronized camera tiles (live embeds or a no-footage tile)."""
    tiles: List[dmc.Card] = []
    for cam in moment["cameras"]:  # type: ignore[index]
        is_best = bool(cam["is_best"])
        embed_url = cam.get("embed_url")  # type: ignore[union-attr]
        if embed_url:
            inner: object = html.Iframe(
                src=str(embed_url),
                className="camera-frame",
                allow="encrypted-media; picture-in-picture",
            )
        else:
            inner = dmc.Center(
                dmc.Text(
                    "No mirror footage for this angle",
                    className="camera-missing",
                    size="xs",
                    c="dimmed",
                    ta="center",
                )
            )
        media_children: List[object] = [
            dmc.AspectRatio(inner, ratio=16 / 10),
            html.Span(str(moment["clock_label"]), className="cam-time"),
        ]
        if is_best:
            media_children.append(
                dmc.Badge("best", color="indigo", size="xs", className="best-badge")
            )
        tiles.append(
            dmc.Card(
                className="camera-tile best" if is_best else "camera-tile",
                withBorder=True,
                radius="md",
                p=6,
                children=[
                    dmc.Box(media_children, className="camera-media"),
                    dmc.Group(
                        justify="space-between",
                        mt=6,
                        children=[
                            dmc.Text(str(cam["camera_id"]), size="sm", fw=600),
                            dmc.Text(
                                f"{float(cam['match_score']):.2f}",
                                size="sm",
                                c="dimmed",
                                ff="monospace",
                            ),
                        ],
                    ),
                ],
            )
        )
    return tiles


def _review_column(camera_id: str, info: Dict[str, str]) -> dmc.Paper:
    """Render one per-camera review column with confirm/refine/reject controls.

    The verdict buttons are ``dmc.Button``s carrying the same pattern-matching
    ids/``n_clicks`` the review callback listens on; the active verdict is shown
    as a filled button, the others light.
    """
    state = info.get("state", "pending")
    justification = info.get("justification", "")
    actions = (("confirm", "✓ Confirm"), ("refine", "↻ Refine"), ("reject", "✕"))
    buttons = [
        dmc.Button(
            label,
            id={"type": "review-btn", "cam": camera_id, "action": action},
            n_clicks=0,
            size="xs",
            color=_REVIEW_ACTION_COLOR[action],
            variant="filled" if _REVIEW_ACTION_STATE[action] == state else "light",
        )
        for action, label in actions
    ]
    return dmc.Paper(
        className=f"review review-{state}",
        withBorder=True,
        radius="sm",
        p="xs",
        children=[
            dmc.Text(camera_id, size="sm", fw=600, mb=4),
            dcc.Textarea(
                id={"type": "review-just", "cam": camera_id},
                value=justification,
                placeholder="Justify, then pick a verdict…",
                className="review-just",
            ),
            dmc.Group(buttons, gap=4, mt=6, className="review-controls"),
        ],
    )


def _render_review_row(review: Dict[str, Dict[str, str]]) -> List[html.Div]:
    """Render the three review columns from the review store."""
    return [_review_column(cam, info) for cam, info in review.items()]


def _all_verdicts_in(review: Dict[str, Dict[str, str]]) -> bool:
    """Return True when every camera has been given a verdict (not pending)."""
    return bool(review) and all(
        info.get("state", "pending") != "pending" for info in review.values()
    )

def _all_confirmed(review: Dict[str, Dict[str, str]]) -> bool:
    """Return True when every camera has been confirmed."""
    return bool(review) and all(
        info.get("state", "pending") == "confirmed" for info in review.values()
    )


def _needs_refinement(review: Dict[str, Dict[str, str]]) -> bool:
    """Return True when any camera was flagged for refinement or rejected."""
    return any(
        info.get("state", "pending") in {"flagged", "rejected"}
        for info in review.values()
    )


def _converged_banner() -> List[dmc.Alert]:
    """Return the review-driven convergence banner."""
    return [
        dmc.Alert(
            "All synchronized camera views were confirmed — no further "
            "refinement needed.",
            title="✓ Search converged",
            color="green",
            variant="light",
        )
    ]


def _viewer_outputs(
    group: Dict[str, object],
    moment: Dict[str, object],
    review: Dict[str, Dict[str, str]],
    iteration: int,
) -> tuple:
    """Build the seven viewer outputs for a focused (group, moment).

    Returns: title, subtitle, camera-grid, review-row, banner children, banner
    hidden, evidence figure. (The compose box and its prefilled query are managed
    separately, as they appear only once all three cameras have a verdict.)
    """
    title = "Refined moment" if group["is_refinement"] else "Selected moment"
    subtitle = (
        f"{moment['clock_label']} · {moment['camera_count']} synchronized cameras"
    )
    support = str(group["claim"]["support"])  # type: ignore[index]
    converged = support == "supported"
    if converged:
        banner = [
            dmc.Alert(
                f"Claim supported after {iteration} of {_MAX_ITERATIONS} max "
                f"iterations — no further refinement needed.",
                title="✓ Search converged",
                color="green",
                variant="light",
            )
        ]
    else:
        banner = []
    return (
        title,
        subtitle,
        _render_camera_grid(moment),
        _render_review_row(review),
        banner,
        not converged,
        camera_match_figure(moment),
    )


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

# The viewer outputs that `_viewer_outputs` fills, in order. Shared by every
# callback that re-renders the right column so positions stay in lockstep.
def _viewer_output_specs() -> List[Output]:
    return [
        Output("viewer-title", "children", allow_duplicate=True),
        Output("viewer-subtitle", "children", allow_duplicate=True),
        Output("camera-grid", "children", allow_duplicate=True),
        Output("review-row", "children", allow_duplicate=True),
        Output("converged-banner", "children", allow_duplicate=True),
        Output("converged-banner", "hidden", allow_duplicate=True),
        Output("evidence-figure", "figure", allow_duplicate=True),
    ]


# Outputs shared by the investigation-opening callbacks (ask-new / send-refined).
def _thread_outputs() -> List[Output]:
    return [
        Output("thread-store", "data", allow_duplicate=True),
        Output("iteration-store", "data", allow_duplicate=True),
        Output("focus-store", "data", allow_duplicate=True),
        Output("review-store", "data", allow_duplicate=True),
        Output("thread", "children", allow_duplicate=True),
        *_viewer_output_specs(),
        Output("compose-wrap", "hidden", allow_duplicate=True),
        Output("refined-query-input", "value", allow_duplicate=True),
    ]


def register_callbacks(
    app: object, engine: ChatEngine, mirror: YouTubeMirror
) -> None:
    """Register the dashboard callbacks on ``app``."""

    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        *_thread_outputs(),
        Input("ask-new-button", "n_clicks"),
        Input("new-question-input", "n_submit"),
        State("new-question-input", "value"),
        prevent_initial_call=True,
    )
    def on_ask_new(
        n_clicks: Optional[int], n_submit: Optional[int], question: Optional[str]
    ) -> tuple:
        """Open a fresh investigation thread (button click or Enter in the box)."""
        if not question or not question.strip():
            raise PreventUpdate
        question = question.strip()
        result = engine.answer(question)
        group = _serialize_group(
            result,
            group_id="g1",
            iteration=1,
            question=question,
            mirror=mirror,
            is_refinement=False,
        )
        thread = [group]
        moment = group["moments"][0]  # type: ignore[index]
        focus = {"group_id": "g1", "moment_id": moment["moment_id"]}
        review = _pending_review(moment)
        iteration_store = {"claim": group["claim"]["text"], "iteration": 1}  # type: ignore[index]
        return (
            thread,
            iteration_store,
            focus,
            review,
            _render_thread(thread, focus),
            *_viewer_outputs(group, moment, review, 1),
            True,  # compose-wrap hidden until all cameras are reviewed
            "",  # clear any prior refined query
        )

    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        *_thread_outputs(),
        Input("send-refined-button", "n_clicks"),
        State("refined-query-input", "value"),
        State("thread-store", "data"),
        State("iteration-store", "data"),
        prevent_initial_call=True,
    )
    def on_send_refined(
        n_clicks: int,
        refined_query: Optional[str],
        thread: Optional[List[Dict[str, object]]],
        iteration_store: Optional[Dict[str, object]],
    ) -> tuple:
        """Re-run retrieval for the same claim, appending a refinement group."""
        thread = list(thread or [])
        store = iteration_store or {}
        claim = store.get("claim")
        iteration = int(store.get("iteration", 0) or 0)
        if not thread or not claim or iteration >= _MAX_ITERATIONS:
            raise PreventUpdate
        if not refined_query or not refined_query.strip():
            raise PreventUpdate
        refined_query = refined_query.strip()

        new_iteration = iteration + 1
        result = engine.refine(str(claim), refined_query, new_iteration)
        group = _serialize_group(
            result,
            group_id=f"g{len(thread) + 1}",
            iteration=new_iteration,
            question=refined_query,
            mirror=mirror,
            is_refinement=True,
            refined_query=refined_query,
        )
        thread.append(group)
        moment = group["moments"][0]  # type: ignore[index]
        focus = {"group_id": group["group_id"], "moment_id": moment["moment_id"]}
        review = _pending_review(moment)
        new_store = {"claim": claim, "iteration": new_iteration}
        return (
            thread,
            new_store,
            focus,
            review,
            _render_thread(thread, focus),
            *_viewer_outputs(group, moment, review, new_iteration),
            True,  # re-hide compose until the new moment's cameras are reviewed
            "",
        )

    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        Output("focus-store", "data", allow_duplicate=True),
        Output("review-store", "data", allow_duplicate=True),
        Output("thread", "children", allow_duplicate=True),
        *_viewer_output_specs(),
        Output("compose-wrap", "hidden", allow_duplicate=True),
        Output("refined-query-input", "value", allow_duplicate=True),
        Input({"type": "moment", "gid": ALL, "mid": ALL}, "n_clicks"),
        State("thread-store", "data"),
        State("iteration-store", "data"),
        prevent_initial_call=True,
    )
    def on_moment_click(
        n_clicks: List[int],
        thread: Optional[List[Dict[str, object]]],
        iteration_store: Optional[Dict[str, object]],
    ) -> tuple:
        """Focus the clicked moment, re-seeking the three camera embeds."""
        triggered = ctx.triggered_id
        if not triggered or not thread:
            raise PreventUpdate
        # Newly injected moment buttons fire this callback with n_clicks 0/None
        # even under prevent_initial_call; only act on a real click.
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            raise PreventUpdate
        group = _find_group(thread, triggered["gid"])
        if group is None:
            raise PreventUpdate
        moment = _find_moment(group, triggered["mid"])
        if moment is None:
            raise PreventUpdate
        focus = {"group_id": triggered["gid"], "moment_id": triggered["mid"]}
        review = _pending_review(moment)
        iteration = int((iteration_store or {}).get("iteration", 0) or 0)
        return (
            focus,
            review,
            _render_thread(thread, focus),
            *_viewer_outputs(group, moment, review, iteration),
            True,  # compose hidden until this moment's cameras are reviewed
            "",
        )

    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        Output("review-store", "data", allow_duplicate=True),
        Output("review-row", "children", allow_duplicate=True),
        Output("compose-wrap", "hidden", allow_duplicate=True),
        Output("refined-query-input", "value", allow_duplicate=True),
        Output("converged-banner", "children", allow_duplicate=True),
        Output("converged-banner", "hidden", allow_duplicate=True),
        Input({"type": "review-btn", "cam": ALL, "action": ALL}, "n_clicks"),
        State({"type": "review-just", "cam": ALL}, "value"),
        State("review-store", "data"),
        State("thread-store", "data"),
        State("focus-store", "data"),
        prevent_initial_call=True,
    )
    def on_review_action(
        n_clicks: List[int],
        justifications: List[Optional[str]],
        review: Optional[Dict[str, Dict[str, str]]],
        thread: Optional[List[Dict[str, object]]],
        focus: Optional[Dict[str, object]],
    ) -> tuple:
        """Record a per-camera verdict; reveal the prefilled refine box when done.

        The reviewer types a justification, then clicks one verdict per camera.
        Once all three cameras have a verdict, the refine compose box appears
        pre-filled with a query generated from those justifications.
        """
        triggered = ctx.triggered_id
        review = dict(review or {})
        if not triggered or not review:
            raise PreventUpdate
        # Newly injected verdict buttons fire this callback with n_clicks 0/None
        # even under prevent_initial_call; only act on a real click.
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            raise PreventUpdate

        # Capture every justification field (mapped by camera via its id).
        just_states = ctx.states_list[0] if ctx.states_list else []
        for item in just_states:
            cam = item["id"]["cam"]
            if cam in review:
                review[cam] = {
                    **review[cam],
                    "justification": item.get("value") or "",
                }

        cam = triggered["cam"]
        if cam in review:
            review[cam] = {
                **review[cam],
                "state": _REVIEW_ACTION_STATE[triggered["action"]],
            }

        if _all_verdicts_in(review):
            if _all_confirmed(review):
                return review, _render_review_row(review), True, "", _converged_banner(), False

            if _needs_refinement(review):
                claim_text = _focused_claim_text(thread, focus)
                prefilled = compose_refined_query(claim_text, review)
                return review, _render_review_row(review), False, prefilled, [], True

        return review, _render_review_row(review), True, "", [], True

