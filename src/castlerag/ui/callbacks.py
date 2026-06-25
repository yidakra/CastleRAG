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

import json
import re
from typing import Dict, List, Optional, Tuple

import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from castlerag.ui.chat import ChatEngine, ChatTurnResult
from castlerag.ui.figures import (
    camera_match_figure,
    empty_figure,
    pipeline_funnel_figure,
)
from castlerag.ui.youtube import YouTubeMirror

_MAX_ITERATIONS = 5

_REVIEW_ACTION_STATE = {
    "confirm": "confirmed",
    "refine": "flagged",
    "reject": "rejected",
    "ignore": "ignored",
}
# Mantine color per review action button.
_REVIEW_ACTION_COLOR = {
    "confirm": "green",
    "refine": "yellow",
    "reject": "red",
    "ignore": "gray",
}
# How each recorded verdict reads on a frozen (read-only) iteration.
_REVIEW_STATE_DISPLAY = {
    "confirmed": ("✓ Confirmed", "green"),
    "flagged": ("↻ Flagged for refine", "yellow"),
    "rejected": ("✕ Rejected", "red"),
    "ignored": ("— Ignored", "gray"),
    "pending": ("— No verdict", "gray"),
}

_INTERNAL_LINK_RE = re.compile(r"\[([^\]]+)\]\((?!https?://)([^)]*)\)")
# Inline evidence citation the placeholder/engine answer embeds:
# [[cite:{moment_id}:{camera_id}:{label}]] — rendered as a clickable link.
_CITE_MARKER_RE = re.compile(r"\[\[cite:([^:\]]+):([^:\]]+):([^\]]*)\]\]")
# Minimal inline markdown the answer prose uses (only **bold**).
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")


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
                # A synchronized angle pulled in by timestamp (not a semantic
                # match) — the tile shows "sync" rather than a 0.00 score.
                "is_context": getattr(cam, "is_context", False),
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
                # Click-out to the canonical YouTube watch page (seeked), so a
                # viewer that hits the cross-site bot gate on the iframe can
                # still see the clip in their signed-in YouTube tab.
                "watch_url": (
                    None
                    if mirror.is_placeholder(cam.day, cam.camera_id, cam.hour)
                    else mirror.watch_url(
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
                # Epoch-ms anchor so a rejected angle can be swapped in-scene.
                "absolute_start_ms": getattr(moment, "absolute_start_ms", None),
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
        "pipeline_stats": getattr(result, "pipeline_stats", {}) or {},
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


def _moment_anchor(
    group: Dict[str, object], moment_id: str
) -> Optional[tuple]:
    """Return ``(day, absolute_start_ms)`` for a moment, for in-scene refine.

    ``None`` when the moment is missing or carries no absolute time (so the
    engine falls back to a fresh global search).
    """
    moment = _find_moment(group, moment_id)
    if not moment:
        return None
    abs_ms = moment.get("absolute_start_ms")  # type: ignore[union-attr]
    if abs_ms is None:
        return None
    cameras = moment.get("cameras") or []  # type: ignore[union-attr]
    day = cameras[0]["day"] if cameras else None  # type: ignore[index]
    return (day, abs_ms)


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


def _focused_question_text(
    thread: Optional[List[Dict[str, object]]],
    focus: Optional[Dict[str, object]],
) -> str:
    """Return the question of the focused group (empty string if not found).

    This anchors the refined-query draft on what the user actually asked, not on
    the (possibly wrong) prior answer carried in the claim.
    """
    group = _find_group(thread or [], (focus or {}).get("group_id", ""))
    if group is None:
        return ""
    return str(group["question"])  # type: ignore[index]


# ---------------------------------------------------------------------------
# Renderers (pure: read the store dicts, never the mirror)
# ---------------------------------------------------------------------------


def _inline_markdown(text: str) -> List[object]:
    """Render the answer's minimal inline markdown (only ``**bold**``)."""
    nodes: List[object] = []
    pos = 0
    for match in _BOLD_RE.finditer(text):
        if match.start() > pos:
            nodes.append(text[pos:match.start()])
        nodes.append(html.Strong(match.group(1)))
        pos = match.end()
    if pos < len(text):
        nodes.append(text[pos:])
    return nodes


def _citation_chip(
    group_id: str, moment_id: str, camera_id: str, label: str
) -> html.Button:
    """Render one inline evidence citation as a clickable, seekable link.

    Clicking the chip focuses the cited moment and autoplays that camera's clip
    from the cited timestamp (see ``on_citation_click``). The moment id is
    namespaced with the group, matching the moment-button ids.
    """
    return html.Button(
        f"▶ {camera_id} · {label}",
        id={
            "type": "cite",
            "gid": group_id,
            "mid": f"{group_id}-{moment_id}",
            "cam": camera_id,
        },
        n_clicks=0,
        className="cite-chip",
        title=f"Play {camera_id} at {label}",
    )


def _render_answer(group_id: str, answer_text: str) -> dmc.Text:
    """Render an answer string with inline, clickable evidence citations.

    Internal (non-http) reference links are stripped first; ``[[cite:...]]``
    markers become clickable chips that seek the matching camera embed, and the
    surrounding prose keeps its ``**bold**`` emphasis.
    """
    text = _strip_internal_links(str(answer_text))
    nodes: List[object] = []
    pos = 0
    for match in _CITE_MARKER_RE.finditer(text):
        if match.start() > pos:
            nodes.extend(_inline_markdown(text[pos:match.start()]))
        moment_id, camera_id, label = match.group(1), match.group(2), match.group(3)
        nodes.append(_citation_chip(group_id, moment_id, camera_id, label))
        pos = match.end()
    if pos < len(text):
        nodes.extend(_inline_markdown(text[pos:]))
    return dmc.Text(nodes, className="answer-text", size="sm")


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


# Answer-confidence badge keyed by the claim's support level. Mirrors the
# evidence-dot palette (green / amber / red) so a glance tells the reviewer how
# strongly the surfaced evidence backs the answer — "Low confidence" is the
# low-support flag the prediction's is_supported signal was always meant to show.
_SUPPORT_BADGE: Dict[str, Tuple[str, str]] = {
    "supported": ("Well supported", "green"),
    "partial": ("Partial support", "yellow"),
    "unsupported": ("Low confidence", "red"),
}


def _support_badge(support: str) -> dmc.Badge:
    """Return a coloured confidence badge for a claim's support level."""
    label, color = _SUPPORT_BADGE.get(support, _SUPPORT_BADGE["partial"])
    return dmc.Badge(label, variant="light", color=color, size="xs")


def _render_group(
    group: Dict[str, object], focus: Dict[str, object], order: int = 0
) -> dmc.Card:
    """Render one query group (question, answer, ranked moments)."""
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

    funnel_section: List[object] = []
    stats = group.get("pipeline_stats") or {}
    if stats and stats.get("retrieved", 0) > 0:
        funnel_section = [
            dmc.Text("Retrieval pipeline", size="xs", c="dimmed", mt="sm"),
            dcc.Graph(
                id=f"funnel-{group['group_id']}",
                figure=pipeline_funnel_figure(stats),  # type: ignore[arg-type]
                config={"displayModeBar": False, "staticPlot": True},
                style={"marginTop": "2px"},
            ),
        ]

    return dmc.Card(
        className="group-card",
        withBorder=True,
        shadow="sm",
        radius="md",
        p="md",
        children=[
            *header,
            dmc.Text(str(group["question"]), fw=600, className="question-text"),
            dmc.Group(
                [
                    dmc.Text("Answer", size="xs", c="dimmed"),
                    _support_badge(
                        str(
                            (group.get("claim") or {}).get("support", "partial")  # type: ignore[union-attr]
                        )
                    ),
                ],
                gap="xs",
                align="center",
                mt="sm",
            ),
            _render_answer(str(group["group_id"]), str(group["answer_text"])),
            *funnel_section,
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


def _render_camera_grid(
    moment: Dict[str, object], autoplay_camera: Optional[str] = None
) -> List[dmc.Card]:
    """Render the synchronized camera tiles (live embeds or a no-footage tile).

    When ``autoplay_camera`` matches a camera id (set by a citation click), that
    tile's embed autoplays from its seeked start and is flagged as ``playing``.
    """
    tiles: List[dmc.Card] = []
    for cam in moment["cameras"]:  # type: ignore[index]
        is_best = bool(cam["is_best"])
        is_playing = autoplay_camera is not None and cam["camera_id"] == autoplay_camera
        embed_url = cam.get("embed_url")  # type: ignore[union-attr]
        if embed_url:
            src = str(embed_url)
            if is_playing:
                # The stored embed already carries ?start=…&rel=0, so append.
                src += "&autoplay=1"
            # referrerpolicy=strict-origin gives YouTube a usable Referer (the
            # bare origin) instead of the full tunnel URL, slightly nudging the
            # cross-site cookie/anti-bot heuristics in the embed's favour.
            inner: object = html.Iframe(
                src=src,
                className="camera-frame",
                allow="autoplay; encrypted-media; picture-in-picture",
                referrerPolicy="strict-origin",
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
        if is_playing:
            media_children.append(
                dmc.Badge(
                    "▶ playing", color="indigo", size="xs", className="playing-badge"
                )
            )
        # "Open on YouTube" fallback — useful when a tunneled deployment hits
        # the cross-site bot gate on the embed. None when the mirror has no
        # upload for this triple (the tile already shows a no-footage message).
        watch_url = cam.get("watch_url")  # type: ignore[union-attr]
        # Co-temporally pulled-in angles (no semantic score against this query)
        # render "sync" instead of a 0.00 score so the UI doesn't pretend they
        # were ranked.
        score_label = (
            "sync"
            if cam.get("is_context")
            else f"{float(cam['match_score']):.2f}"
        )
        header_children: List[object] = [
            dmc.Text(str(cam["camera_id"]), size="sm", fw=600),
            dmc.Text(
                score_label,
                size="sm",
                c="dimmed",
                ff="monospace",
            ),
        ]
        if watch_url:
            header_children.insert(
                1,
                html.A(
                    "open ↗",
                    href=str(watch_url),
                    target="_blank",
                    rel="noopener noreferrer",
                    className="camera-watch-link",
                ),
            )
        tile_class = "camera-tile"
        if is_best:
            tile_class += " best"
        if is_playing:
            tile_class += " playing"
        tiles.append(
            dmc.Card(
                className=tile_class,
                withBorder=True,
                radius="md",
                p=6,
                children=[
                    dmc.Box(media_children, className="camera-media"),
                    dmc.Group(
                        justify="space-between",
                        mt=6,
                        children=header_children,
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

    actions = (
        ("confirm", "✓ Confirm"),
        ("refine", "↻ Refine"),
        ("reject", "✕"),
        ("ignore", "— Ignore"),
    )
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

def _should_converge(review: Dict[str, Dict[str, str]]) -> bool:
    """Return True when every camera is confirmed or ignored (no outstanding refine)."""
    return bool(review) and all(
        info.get("state", "pending") in {"confirmed", "ignored"}
        for info in review.values()
    )


def _strip_internal_links(text: str) -> str:
    """Remove LLM-generated evidence ref links (non-http), keeping the link text."""
    return _INTERNAL_LINK_RE.sub(r"\1", text)


def _needs_refinement(review: Dict[str, Dict[str, str]]) -> bool:
    """Return True when any camera was flagged for refinement or rejected."""
    return any(
        info.get("state", "pending") in {"flagged", "rejected"}
        for info in review.values()
    )


def _rejected_cameras(
    thread: Optional[List[Dict[str, object]]],
    current_review: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[str]:
    """Camera ids the reviewer rejected anywhere on this thread.

    Accumulates across every frozen iteration plus the in-flight review, so a
    camera rejected once stays excluded from all later refine iterations.
    """
    rejected: set = set()
    for group in thread or []:
        for _moment_id, review in (group.get("reviews") or {}).items():  # type: ignore[union-attr]
            for camera_id, info in (review or {}).items():
                if isinstance(info, dict) and info.get("state") == "rejected":
                    rejected.add(camera_id)
    for camera_id, info in (current_review or {}).items():
        if isinstance(info, dict) and info.get("state") == "rejected":
            rejected.add(camera_id)
    return sorted(rejected)


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
            "All synchronized camera views were confirmed or ignored — no further "
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
    autoplay_camera: Optional[str] = None,
) -> tuple:
    """Build the seven viewer outputs for a focused (group, moment).

    Returns: title, subtitle, camera-grid, review-row, banner children, banner
    hidden, evidence figure. (The compose box and its prefilled query are managed
    separately, as they appear only once all three cameras have a verdict.) When
    ``frozen`` the review controls render read-only for a past iteration.
    ``autoplay_camera`` (set by a citation click) autoplays that camera's clip.
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
        _render_camera_grid(moment, autoplay_camera),
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

    # ---- 1. Disable Send when input is empty ----------------------------------
    app.clientside_callback(
        """
        function(value) {
            return !value || value.trim() === "";
        }
        """,
        Output("ask-new-button", "disabled"),
        Input("new-question-input", "value"),
    )

    # ---- 2. Auto-focus input after every thread update -----------------------
    app.clientside_callback(
        """
        function(data) {
            setTimeout(function() {
                var el = document.getElementById("new-question-input");
                if (el) {
                    el.focus();
                    // Reset textarea height after submission/clear.
                    if (!el.value) el.style.height = "";
                }
            }, 80);
            return "";
        }
        """,
        Output("_focus-dummy", "children"),
        Input("thread-store", "data"),
        prevent_initial_call=True,
    )

    # ---- 2b. Optimistic submit: clear the box + show the question instantly --
    # Runs in the browser the moment Send is hit, so the typed text leaves the
    # input and appears as a "searching" card immediately — instead of lingering
    # in the box until the (possibly slow) pipeline returns. The question is
    # stashed in ``pending-question`` for the server callback to pick up, so the
    # value is never lost to a clear/read race.
    app.clientside_callback(
        """
        function(n_clicks, value) {
            const dc = window.dash_clientside;
            if (!n_clicks || !value || !value.trim()) {
                return [dc.no_update, dc.no_update, dc.no_update, dc.no_update];
            }
            const q = value.trim();
            // n is included so re-asking the identical question still changes
            // the store and re-triggers the server callback.
            return ["", {q: q, n: n_clicks}, q, false];
        }
        """,
        Output("new-question-input", "value", allow_duplicate=True),
        Output("pending-question", "data"),
        Output("pending-card", "children"),
        Output("pending-card", "hidden"),
        Input("ask-new-button", "n_clicks"),
        State("new-question-input", "value"),
        prevent_initial_call=True,
    )

    # ---- 3. Clear thread -----------------------------------------------------
    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        Output("thread-store", "data", allow_duplicate=True),
        Output("iteration-store", "data", allow_duplicate=True),
        Output("focus-store", "data", allow_duplicate=True),
        Output("review-store", "data", allow_duplicate=True),
        Output("thread", "children", allow_duplicate=True),
        Output("new-question-input", "value", allow_duplicate=True),
        *_viewer_output_specs(),
        Output("compose-wrap", "hidden", allow_duplicate=True),
        Output("refined-query-input", "value", allow_duplicate=True),
        Input("clear-thread-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_clear(n_clicks: Optional[int]) -> tuple:
        """Reset all thread and viewer state to the initial empty dashboard."""
        if not n_clicks:
            raise PreventUpdate
        return (
            [],
            {"claim": None, "iteration": 0, "next_seq": 1},
            {},
            {},
            _render_thread([], {}),
            "",
            # _viewer_output_specs order: title, subtitle, camera-grid, review-row,
            # banner children, banner hidden, evidence-figure, submit-wrap hidden
            "Selected moment",
            "",
            [dmc.Text(
                "Select an evidence moment to see its synchronized cameras.",
                className="viewer-hint",
                c="dimmed",
                size="sm",
            )],
            [],
            [],
            True,
            empty_figure(),
            True,
            True,   # compose-wrap hidden
            "",     # refined-query-input
        )

    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        Output("export-download", "data"),
        Input("export-thread-button", "n_clicks"),
        State("thread-store", "data"),
        prevent_initial_call=True,
    )
    def on_export_thread(n_clicks: Optional[int], thread: Optional[list]) -> object:
        """Serialize the current thread to JSON and trigger a browser download."""
        if not n_clicks or not thread:
            raise PreventUpdate
        turns = []
        for g in thread:
            turns.append({
                "group_id": g.get("group_id"),
                "iteration": g.get("iteration"),
                "is_refinement": g.get("is_refinement"),
                "question": g.get("question"),
                "refined_query": g.get("refined_query"),
                "answer": g.get("answer_text"),
                "reviews": g.get("reviews", {}),
                "moments": [
                    {
                        "moment_id": m.get("moment_id"),
                        "clock_label": m.get("clock_label"),
                        "place_label": m.get("place_label"),
                        "cameras": [
                            {
                                "camera_id": c.get("camera_id"),
                                "match_score": c.get("match_score"),
                                "is_best": c.get("is_best"),
                                "evidence_text": c.get("evidence_text"),
                            }
                            for c in (m.get("cameras") or [])
                        ],
                    }
                    for m in (g.get("moments") or [])
                ],
            })
        payload = json.dumps({"turns": turns}, indent=2, ensure_ascii=False)
        return dcc.send_string(payload, "castlerag_session.json")

    @app.callback(  # type: ignore[attr-defined,untyped-decorator]
        *_thread_outputs(),
        Output("pending-card", "children", allow_duplicate=True),
        Output("pending-card", "hidden", allow_duplicate=True),
        Input("pending-question", "data"),
        State("thread-store", "data"),
        State("iteration-store", "data"),
        State("focus-store", "data"),
        State("review-store", "data"),
        prevent_initial_call=True,
    )
    def on_ask_new(
        pending: Optional[Dict[str, object]],
        thread: Optional[List[Dict[str, object]]],
        iteration_store: Optional[Dict[str, object]],
        focus: Optional[Dict[str, object]],
        review: Optional[Dict[str, Dict[str, str]]],
    ) -> tuple:
        """Append a new investigation block, freezing the current iteration.

        Triggered by the ``pending-question`` store (set by the optimistic
        clientside callback), which already cleared the input and showed the
        question. A new question never wipes the thread: the previous iteration
        is frozen (kept read-only) and the new question is appended as its own
        block so the whole history stays visible.
        """
        question = (pending or {}).get("q") if isinstance(pending, dict) else pending
        if not question or not str(question).strip():
            raise PreventUpdate
        question = str(question).strip()
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
            "claim": group["answer_text"],  # type: ignore[index]
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
            [],  # clear the optimistic pending-question card…
            True,  # …and hide it now the real result group is rendered
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
        # Anchor the refine on the moment under review so a rejected angle is
        # swapped in-scene (same timestamp) rather than teleporting to a new one.
        anchor = _moment_anchor(current, str(moment_id)) if moment_id else None
        result = engine.refine(
            str(claim),
            refined_query,
            new_iteration,
            exclude_cameras=_rejected_cameras(thread, review),
            anchor=anchor,
            reviews=review,
        )
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
        Output("focus-store", "data", allow_duplicate=True),
        Output("review-store", "data", allow_duplicate=True),
        Output("thread", "children", allow_duplicate=True),
        *_viewer_output_specs(),
        Output("compose-wrap", "hidden", allow_duplicate=True),
        Output("refined-query-input", "value", allow_duplicate=True),
        Input({"type": "cite", "gid": ALL, "mid": ALL, "cam": ALL}, "n_clicks"),
        State("thread-store", "data"),
        prevent_initial_call=True,
    )
    def on_citation_click(
        n_clicks: List[int],
        thread: Optional[List[Dict[str, object]]],
    ) -> tuple:
        """Focus a cited moment and autoplay the cited camera from its timestamp.

        Mirrors ``on_moment_click`` (focus + re-seek the synchronized cameras) but
        also autoplays the specific camera the answer's citation pointed at, so a
        click on an inline evidence link starts that clip in place.
        """
        triggered = ctx.triggered_id
        if not triggered or not thread:
            raise PreventUpdate
        # Freshly injected citation chips fire with n_clicks 0/None; ignore those.
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

        saved = (group.get("reviews") or {}).get(triggered["mid"])  # type: ignore[union-attr]
        review = dict(saved) if saved else _pending_review(moment)

        return (
            focus,
            review,
            _render_thread(thread, focus),
            *_viewer_outputs(
                group, moment, review, iteration, frozen,
                autoplay_camera=triggered["cam"],
            ),
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

        if _should_converge(review):
            return review, thread, True, "", _converged_banner(), False

        claim_text = _focused_claim_text(thread, focus)
        question_text = _focused_question_text(thread, focus)
        prefilled = engine.suggest_refined_query(
            claim_text, review, question=question_text
        )
        return review, thread, False, prefilled, [], True

