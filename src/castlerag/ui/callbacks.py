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

from castlerag.ui.chat import ChatEngine, ChatTurnResult
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
# How each recorded verdict reads on a frozen (read-only) iteration.
_REVIEW_STATE_DISPLAY = {
    "confirmed": ("✓ Confirmed", "green"),
    "flagged": ("↻ Flagged for refine", "yellow"),
    "rejected": ("✕ Rejected", "red"),
    "pending": ("— No verdict", "gray"),
}


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
                # Retrieved evidence snippet for this camera (None offline), used
                # to ground LLM justification suggestions in the review UI.
                "evidence_text": getattr(cam, "evidence_text", None),
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
                # Namespace the moment id with the group so no button id is ever
                # reused across groups (prevents React/Dash reconciling a new
                # group's moments onto an earlier group's DOM nodes).
                "moment_id": f"{group_id}-{moment.moment_id}",
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
        # Frozen per-camera verdicts captured when the user refines past this
        # iteration, keyed by the reviewed moment_id. Empty until then.
        "reviews": {},
        # The refined query the user sent from this iteration (set on freeze).
        "sent_refined_query": None,
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


def _render_group(
    group: Dict[str, object], focus: Dict[str, object], order: int = 0
) -> dmc.Card:
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
    elif order > 0:
        # A fresh question appended after earlier investigations.
        header.append(
            dmc.Badge(
                "🔎 New question",
                variant="light",
                color="teal",
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
    return [_render_group(group, focus, i) for i, group in enumerate(thread)]


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


def _review_column(
    camera_id: str, info: Dict[str, str], frozen: bool = False
) -> dmc.Paper:
    """Render one per-camera review column with confirm/refine/reject controls.

    The verdict buttons are ``dmc.Button``s carrying the same pattern-matching
    ids/``n_clicks`` the review callback listens on; the active verdict is shown
    as a filled button, the others light. When ``frozen`` (a past iteration), the
    controls are disabled so the recorded verdict is visible but read-only.
    """
    state = info.get("state", "pending")
    justification = info.get("justification", "")

    if frozen:
        # Read-only: show the recorded verdict as a clear coloured badge plus the
        # justification text, so it's obvious which verdict was chosen.
        verdict_label, verdict_color = _REVIEW_STATE_DISPLAY.get(
            state, _REVIEW_STATE_DISPLAY["pending"]
        )
        just_node = (
            dmc.Text(justification, size="xs", c="dimmed")
            if justification
            else dmc.Text(
                "No justification recorded", size="xs", c="dimmed", fs="italic"
            )
        )
        return dmc.Paper(
            className=f"review review-{state} review-frozen",
            withBorder=True,
            radius="sm",
            p="xs",
            children=[
                dmc.Text(camera_id, size="sm", fw=600, mb=4),
                dmc.Badge(verdict_label, color=verdict_color, variant="filled", mb=6),
                just_node,
            ],
        )

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
    children: List[object] = [
        dmc.Text(camera_id, size="sm", fw=600, mb=4),
        dmc.Group(buttons, gap=4, className="review-controls"),
    ]
    # The justification box only appears once a verdict is picked — it is then
    # pre-filled with an editable AI draft.
    if state != "pending":
        children.append(
            dcc.Textarea(
                id={"type": "review-just", "cam": camera_id},
                value=justification,
                placeholder="Edit the AI-drafted justification…",
                className="review-just",
            )
        )
    return dmc.Paper(
        className=f"review review-{state}",
        withBorder=True,
        radius="sm",
        p="xs",
        children=children,
    )


def _frozen_query_panel(refined_query: str) -> dmc.Paper:
    """Full-width banner showing the refined query sent from a frozen iteration."""
    return dmc.Paper(
        className="frozen-query",
        withBorder=True,
        radius="sm",
        p="xs",
        style={"gridColumn": "1 / -1"},
        children=[
            dmc.Text("↻ Refined query sent from this iteration", size="xs", c="dimmed"),
            dmc.Text(refined_query or "—", size="sm", fw=500),
        ],
    )


def _render_review_row(
    review: Dict[str, Dict[str, str]],
    frozen: bool = False,
    refined_query: str = "",
) -> List[object]:
    """Render the review columns; frozen rows lead with the sent refined query."""
    columns = [_review_column(cam, info, frozen) for cam, info in review.items()]
    if frozen and refined_query:
        return [_frozen_query_panel(refined_query), *columns]
    return columns


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


def _is_live_group(thread: List[Dict[str, object]], group_id: str) -> bool:
    """Return True when ``group_id`` is the latest (editable) iteration.

    Only the last group in the thread is live; every earlier iteration is frozen
    and shown read-only.
    """
    return bool(thread) and thread[-1]["group_id"] == group_id


def _submit_hidden(review: Dict[str, Dict[str, str]], frozen: bool) -> bool:
    """Submit-reviews button shows once every camera has a verdict (live only)."""
    return frozen or not _all_verdicts_in(review)


def _capture_justifications(
    review: Dict[str, Dict[str, str]], states_list: object
) -> Dict[str, Dict[str, str]]:
    """Fold the live justification textarea values back into the review dict."""
    just_states = states_list[0] if states_list else []  # type: ignore[index]
    for item in just_states:
        cam = item["id"]["cam"]
        if cam in review:
            review[cam] = {**review[cam], "justification": item.get("value") or ""}
    return review


def _persist_review(
    thread: List[Dict[str, object]],
    focus: Optional[Dict[str, object]],
    review: Dict[str, Dict[str, str]],
) -> None:
    """Persist the in-progress verdicts onto the live group (mutates ``thread``)."""
    moment_id = (focus or {}).get("moment_id")
    if moment_id and thread:
        current = dict(thread[-1])
        reviews = dict(current.get("reviews") or {})
        reviews[str(moment_id)] = review
        current["reviews"] = reviews
        thread[-1] = current


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
    frozen: bool = False,
) -> tuple:
    """Build the seven viewer outputs for a focused (group, moment).

    Returns: title, subtitle, camera-grid, review-row, banner children, banner
    hidden, evidence figure. (The compose box and its prefilled query are managed
    separately, as they appear only once all three cameras have a verdict.) When
    ``frozen`` the review controls render read-only for a past iteration.
    """
    if frozen:
        title = f"Iteration {iteration} · frozen"
    else:
        title = "Refined moment" if group["is_refinement"] else "Selected moment"
    subtitle = (
        f"{moment['clock_label']} · {moment['camera_count']} synchronized cameras"
    )
    # Convergence and the refined-query suggestion are gated behind the explicit
    # "Submit reviews" button (see ``on_submit_reviews``); opening/focusing a
    # moment never auto-declares them. The submit button itself appears once all
    # three live cameras have a verdict.
    sent_query = str(group.get("sent_refined_query") or "") if frozen else ""
    return (
        title,
        subtitle,
        _render_camera_grid(moment),
        _render_review_row(review, frozen, sent_query),
        [],  # converged banner: managed by on_submit_reviews
        True,  # converged banner hidden
        camera_match_figure(moment),
        _submit_hidden(review, frozen),
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
        Output("submit-wrap", "hidden", allow_duplicate=True),
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
        Output("new-question-input", "value", allow_duplicate=True),
        Input("ask-new-button", "n_clicks"),
        Input("new-question-input", "n_submit"),
        State("new-question-input", "value"),
        State("thread-store", "data"),
        State("iteration-store", "data"),
        State("focus-store", "data"),
        State("review-store", "data"),
        prevent_initial_call=True,
    )
    def on_ask_new(
        n_clicks: Optional[int],
        n_submit: Optional[int],
        question: Optional[str],
        thread: Optional[List[Dict[str, object]]],
        iteration_store: Optional[Dict[str, object]],
        focus: Optional[Dict[str, object]],
        review: Optional[Dict[str, Dict[str, str]]],
    ) -> tuple:
        """Append a new investigation block, freezing the current iteration.

        A new question never wipes the thread: the previous iteration is frozen
        (kept read-only) and the new question is appended as its own block so the
        whole history stays visible.
        """
        if not question or not question.strip():
            raise PreventUpdate
        question = question.strip()
        thread = list(thread or [])
        store = iteration_store or {}

        # Freeze the current live iteration before opening a new block. Only
        # snapshot review-store if it belongs to the live group; otherwise the
        # group already carries its persisted verdicts.
        if thread:
            live = dict(thread[-1])
            if (focus or {}).get("group_id") == live["group_id"]:
                mid = (focus or {}).get("moment_id")
                if mid and review:
                    reviews = dict(live.get("reviews") or {})
                    reviews[str(mid)] = review
                    live["reviews"] = reviews
                    thread[-1] = live

        seq = int(store.get("next_seq", 1) or 1) if thread else 1
        group_id = f"g{seq}"
        result = engine.answer(question)
        group = _serialize_group(
            result,
            group_id=group_id,
            iteration=1,
            question=question,
            mirror=mirror,
            is_refinement=False,
        )
        moments = group["moments"]  # type: ignore[index]
        if not moments:
            # No evidence moments (retrieval failure edge case); nothing to focus.
            raise PreventUpdate
        thread.append(group)
        moment = moments[0]
        new_focus = {"group_id": group_id, "moment_id": moment["moment_id"]}
        new_review = _pending_review(moment)
        iteration_store = {
            "claim": group["claim"]["text"],  # type: ignore[index]
            "iteration": 1,
            "next_seq": seq + 1,
        }
        return (
            thread,
            iteration_store,
            new_focus,
            new_review,
            _render_thread(thread, new_focus),
            *_viewer_outputs(group, moment, new_review, 1),
            True,  # compose-wrap hidden until all cameras are reviewed
            "",  # clear any prior refined query
            "",  # clear the new-question box
        )

    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        *_thread_outputs(),
        Input("send-refined-button", "n_clicks"),
        State("refined-query-input", "value"),
        State("thread-store", "data"),
        State("iteration-store", "data"),
        State("focus-store", "data"),
        State("review-store", "data"),
        prevent_initial_call=True,
    )
    def on_send_refined(
        n_clicks: int,
        refined_query: Optional[str],
        thread: Optional[List[Dict[str, object]]],
        iteration_store: Optional[Dict[str, object]],
        focus: Optional[Dict[str, object]],
        review: Optional[Dict[str, Dict[str, str]]],
    ) -> tuple:
        """Freeze the current iteration's verdicts and append the next one.

        Refinement is linear: it always continues from the latest iteration. The
        verdicts the reviewer gave for the focused moment are frozen onto that
        (now previous) iteration so it stays visible but read-only; earlier
        iterations are never removed or re-run.
        """
        thread = list(thread or [])
        store = iteration_store or {}
        claim = store.get("claim")
        if not thread or not claim:
            raise PreventUpdate
        if not refined_query or not refined_query.strip():
            raise PreventUpdate
        refined_query = refined_query.strip()

        # Linear: continue from the latest iteration only.
        current = dict(thread[-1])
        current_iteration = int(current["iteration"])  # type: ignore[index]
        if current_iteration >= _MAX_ITERATIONS:
            raise PreventUpdate

        # Freeze the reviewed moment's verdicts and the query we send onto the
        # iteration we leave, so the frozen view can show both.
        moment_id = (focus or {}).get("moment_id")
        if moment_id and review:
            reviews = dict(current.get("reviews") or {})
            reviews[str(moment_id)] = review
            current["reviews"] = reviews
        current["sent_refined_query"] = refined_query
        thread[-1] = current

        new_iteration = current_iteration + 1
        seq = int(store.get("next_seq", len(thread) + 1) or (len(thread) + 1))
        result = engine.refine(str(claim), refined_query, new_iteration)
        group = _serialize_group(
            result,
            group_id=f"g{seq}",
            iteration=new_iteration,
            question=refined_query,
            mirror=mirror,
            is_refinement=True,
            refined_query=refined_query,
        )
        moments = group["moments"]  # type: ignore[index]
        if not moments:
            # Refinement returned no evidence moments; leave the thread unchanged.
            raise PreventUpdate
        thread.append(group)
        moment = moments[0]
        new_focus = {"group_id": group["group_id"], "moment_id": moment["moment_id"]}
        new_review = _pending_review(moment)
        new_store = {"claim": claim, "iteration": new_iteration, "next_seq": seq + 1}
        return (
            thread,
            new_store,
            new_focus,
            new_review,
            _render_thread(thread, new_focus),
            *_viewer_outputs(group, moment, new_review, new_iteration),
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
        iteration = int(group["iteration"])  # type: ignore[index]
        frozen = not _is_live_group(thread, triggered["gid"])

        # Restore any saved verdicts for this moment; pending if never reviewed.
        saved = (group.get("reviews") or {}).get(triggered["mid"])  # type: ignore[union-attr]
        review = dict(saved) if saved else _pending_review(moment)

        # The compose box stays hidden until the reviewer submits; _viewer_outputs
        # decides whether the "Submit reviews" button is shown.
        return (
            focus,
            review,
            _render_thread(thread, focus),
            *_viewer_outputs(group, moment, review, iteration, frozen),
            True,  # compose hidden
            "",  # refined query cleared
        )

    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        Output("review-store", "data", allow_duplicate=True),
        Output("review-row", "children", allow_duplicate=True),
        Output("submit-wrap", "hidden", allow_duplicate=True),
        Output("compose-wrap", "hidden", allow_duplicate=True),
        Output("refined-query-input", "value", allow_duplicate=True),
        Output("converged-banner", "children", allow_duplicate=True),
        Output("converged-banner", "hidden", allow_duplicate=True),
        Output("thread-store", "data", allow_duplicate=True),
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
        """Record a per-camera verdict and auto-draft its justification.

        The reviewer clicks one verdict per camera; its justification box appears
        pre-filled with an editable AI draft. The refined query is NOT generated
        here — that waits for the explicit "Submit reviews" button, which appears
        once every camera has a verdict (see ``on_submit_reviews``).
        """
        triggered = ctx.triggered_id
        review = dict(review or {})
        thread = list(thread or [])
        if not triggered or not review:
            raise PreventUpdate
        # Newly injected verdict buttons fire this callback with n_clicks 0/None
        # even under prevent_initial_call; only act on a real click.
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            raise PreventUpdate
        # Verdicts are only editable on the latest (live) iteration; frozen ones
        # render disabled controls, but guard here in case one slips through.
        focus_gid = (focus or {}).get("group_id", "")
        if not _is_live_group(thread, focus_gid):
            raise PreventUpdate

        review = _capture_justifications(review, ctx.states_list)

        cam = triggered["cam"]
        if cam in review:
            new_state = _REVIEW_ACTION_STATE[triggered["action"]]
            prev_state = review[cam].get("state", "pending")
            cur_just = review[cam].get("justification") or ""
            prev_suggestion = review[cam].get("suggestion", "")
            review[cam] = {**review[cam], "state": new_state}

            # Auto-draft a justification on the verdict click (create-then-edit):
            # fill when the box is empty, or when the verdict changed and the
            # reviewer hasn't edited the previous AI draft. Never clobber edits.
            unedited = cur_just == "" or cur_just == prev_suggestion
            if cur_just == "" or (new_state != prev_state and unedited):
                group = _find_group(thread, focus_gid)
                moment_id = (focus or {}).get("moment_id")
                moment = _find_moment(group, str(moment_id)) if group else None
                cam_info = (
                    next(
                        (c for c in moment["cameras"]  # type: ignore[index]
                         if c["camera_id"] == cam),
                        None,
                    )
                    if moment
                    else None
                )
                meta = {
                    "clock_label": (moment or {}).get("clock_label"),
                    "place_label": (moment or {}).get("place_label"),
                    "match_score": (cam_info or {}).get("match_score"),
                }
                suggestion = engine.suggest_justification(
                    _focused_claim_text(thread, focus),
                    cam,
                    new_state,
                    (cam_info or {}).get("evidence_text"),
                    meta,
                )
                if suggestion:
                    review[cam] = {
                        **review[cam],
                        "justification": suggestion,
                        "suggestion": suggestion,
                    }

        # Persist verdicts onto the live group so they survive navigation.
        _persist_review(thread, focus, review)

        # Show the Submit button once all cameras have a verdict; clear any prior
        # compose/converged output since the verdicts just changed.
        submit_hidden = not _all_verdicts_in(review)
        return (
            review,
            _render_review_row(review),
            submit_hidden,
            True,  # compose hidden until submit
            "",  # refined query cleared
            [],  # converged banner cleared
            True,  # converged banner hidden
            thread,
        )

    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        Output("review-store", "data", allow_duplicate=True),
        Output("thread-store", "data", allow_duplicate=True),
        Output("compose-wrap", "hidden", allow_duplicate=True),
        Output("refined-query-input", "value", allow_duplicate=True),
        Output("converged-banner", "children", allow_duplicate=True),
        Output("converged-banner", "hidden", allow_duplicate=True),
        Input("submit-reviews-button", "n_clicks"),
        State({"type": "review-just", "cam": ALL}, "value"),
        State("review-store", "data"),
        State("thread-store", "data"),
        State("focus-store", "data"),
        prevent_initial_call=True,
    )
    def on_submit_reviews(
        n_clicks: Optional[int],
        justifications: List[Optional[str]],
        review: Optional[Dict[str, Dict[str, str]]],
        thread: Optional[List[Dict[str, object]]],
        focus: Optional[Dict[str, object]],
    ) -> tuple:
        """Commit the three verdicts: converge if all confirmed, else draft a query.

        Runs only on an explicit click once every camera has a verdict. Picks up
        the latest (possibly edited) justifications before composing the query.
        """
        if not n_clicks:
            raise PreventUpdate
        review = dict(review or {})
        thread = list(thread or [])
        focus_gid = (focus or {}).get("group_id", "")
        if not review or not _is_live_group(thread, focus_gid):
            raise PreventUpdate
        if not _all_verdicts_in(review):
            raise PreventUpdate

        # Pick up any edits the reviewer made to the justification boxes, persist.
        review = _capture_justifications(review, ctx.states_list)
        _persist_review(thread, focus, review)

        if _all_confirmed(review):
            return review, thread, True, "", _converged_banner(), False

        claim_text = _focused_claim_text(thread, focus)
        prefilled = engine.suggest_refined_query(claim_text, review)
        return review, thread, False, prefilled, [], True

