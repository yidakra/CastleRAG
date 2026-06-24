"""Tests for the UI backbone: YouTube mirror, placeholder engine, app factory."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from castlerag.ui.chat import (
    CameraAngle,
    ChatTurnResult,
    Claim,
    EvidenceMoment,
    PlaceholderEngine,
    QueryGroup,
    SupportLevel,
)
from castlerag.ui.youtube import PLACEHOLDER_VIDEO_ID, YouTubeMirror

# ---------------------------------------------------------------------------
# YouTube mirror
# ---------------------------------------------------------------------------


def test_mirror_from_csv_loads_real_mapping():
    mirror = YouTubeMirror.from_csv()
    assert len(mirror.mapping) == 666  # full CASTLE mirror
    # Real id from the CASTLE viewer's videos.json.
    assert mirror.video_id("day1", "Allie", 8) == "XYkq1PPZu9A"
    assert mirror.is_placeholder("day1", "Allie", 8) is False


def test_mirror_embed_url_includes_start_offset_and_video_id():
    mirror = YouTubeMirror(mapping={("day1", "Allie", 8): "ABCDEFGHIJK"})
    url = mirror.embed_url("day1", "Allie", 8, start_seconds=42.7)
    assert "/embed/ABCDEFGHIJK?" in url
    assert "start=42" in url
    assert url.startswith("https://www.youtube-nocookie.com")


def test_mirror_default_and_watch_urls():
    mirror = YouTubeMirror.from_csv()
    assert "/embed/" in mirror.default_embed_url()
    watch = mirror.watch_url("day1", "Allie", 8, 90)
    assert watch == "https://www.youtube.com/watch?v=XYkq1PPZu9A&t=90s"


def test_mirror_unknown_key_falls_back_to_placeholder():
    mirror = YouTubeMirror(mapping={})
    assert mirror.video_id("day9", "Nobody", 23) == PLACEHOLDER_VIDEO_ID
    assert mirror.is_placeholder("day9", "Nobody", 23) is True


def test_mirror_is_placeholder_tracks_resolved_id():
    mirror = YouTubeMirror(mapping={("day1", "Allie", 8): "REALVIDEOID0"})
    assert mirror.is_placeholder("Day1", "Allie", 8) is False  # day case-normalized
    assert mirror.is_placeholder("day1", "Bjorn", 8) is True


def test_mirror_from_csv_missing_file_is_empty(tmp_path):
    mirror = YouTubeMirror.from_csv(tmp_path / "absent.csv")
    assert mirror.mapping == {}


# ---------------------------------------------------------------------------
# Placeholder engine — answer()
# ---------------------------------------------------------------------------


def test_answer_populates_claim_and_moments():
    result = PlaceholderEngine().answer("Who pranks Bjorn when he returns?")
    assert isinstance(result, ChatTurnResult)
    assert result.is_placeholder is True
    assert isinstance(result.claim, Claim)
    assert result.claim.support is SupportLevel.PARTIAL
    assert result.moments
    for moment in result.moments:
        assert isinstance(moment, EvidenceMoment)
        # Exactly three synchronized cameras, all at the same (day, hour, start).
        assert len(moment.cameras) == 3
        sync = {(c.day, c.hour, c.start_seconds) for c in moment.cameras}
        assert len(sync) == 1  # all three cameras share day/hour/start
        # Exactly one camera flagged best.
        assert sum(1 for c in moment.cameras if c.is_best) == 1


def test_answer_keeps_legacy_contract():
    result = PlaceholderEngine().answer("How many cups are on the table?")
    assert result.predicted_choice in {"a", "b", "c", "d"}
    assert result.route in {"static_visual", "speech_text", "temporal", "mixed"}
    assert set(result.support_priors) == {"a", "b", "c", "d"}
    assert abs(sum(result.support_priors.values()) - 1.0) < 1e-6
    assert result.evidence  # legacy EvidenceRef rows still populated


def test_answer_is_deterministic():
    engine = PlaceholderEngine()
    first = engine.answer("Where did Bjorn go after lunch?")
    second = engine.answer("Where did Bjorn go after lunch?")
    assert first.claim.text == second.claim.text
    assert [m.moment_id for m in first.moments] == [m.moment_id for m in second.moments]
    assert [
        c.match_score for m in first.moments for c in m.cameras
    ] == [c.match_score for m in second.moments for c in m.cameras]


def test_moments_resolve_to_real_embeds():
    mirror = YouTubeMirror.from_csv()
    engine = PlaceholderEngine.from_mirror(mirror)
    result = engine.answer("What was happening when the timer went off?")
    for moment in result.moments:
        for cam in moment.cameras:
            url = mirror.embed_url(cam.day, cam.camera_id, cam.hour, cam.start_seconds)
            assert "/embed/" in url
            assert mirror.is_placeholder(cam.day, cam.camera_id, cam.hour) is False


# ---------------------------------------------------------------------------
# Placeholder engine — refine()
# ---------------------------------------------------------------------------


def test_refine_strengthens_support_and_converges():
    engine = PlaceholderEngine()
    claim = "Luca is the one who sets up the prank."
    early = engine.refine(claim, "Show the doorway more clearly.", 2)
    late = engine.refine(claim, "Confirm Luca's hand on the frame.", 3)
    assert early.claim.support is SupportLevel.PARTIAL
    assert late.claim.support is SupportLevel.SUPPORTED
    # The claim text is preserved across refinements.
    assert early.claim.text == claim == late.claim.text


def test_refine_is_deterministic():
    engine = PlaceholderEngine()
    a = engine.refine("c", "sharper angle", 2)
    b = engine.refine("c", "sharper angle", 2)
    assert [m.moment_id for m in a.moments] == [m.moment_id for m in b.moments]
    assert a.moments[0].aggregate_score == b.moments[0].aggregate_score


# ---------------------------------------------------------------------------
# Serialization (str-Enum round-trips into a dcc.Store)
# ---------------------------------------------------------------------------


def test_query_group_json_round_trips():
    import json

    cam = CameraAngle("Klaus", "day1", 12, 749.0, 0.91, is_best=True)
    moment = EvidenceMoment(
        "m0", "12:29", "Doorway", 3, 0.74, "match 0.74", "#d97706", [cam]
    )
    group = QueryGroup(
        group_id="g1",
        iteration=1,
        question="q",
        answer_text="a",
        claim=Claim("c", SupportLevel.PARTIAL),
        moments=[moment],
    )
    payload = json.loads(json.dumps(asdict(group)))
    assert payload["claim"]["support"] == "partial"
    assert payload["moments"][0]["cameras"][0]["is_best"] is True


# ---------------------------------------------------------------------------
# App factory (requires dash)
# ---------------------------------------------------------------------------


def _walk(node, acc):
    """Collect (type_name, className) for every Dash component in a subtree."""
    if hasattr(node, "_type") or hasattr(node, "children"):
        acc.append((type(node).__name__, getattr(node, "className", None)))
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _walk(child, acc)
    elif children is not None and hasattr(children, "children"):
        _walk(children, acc)
    return acc


def test_serialize_group_marks_unmapped_cameras_as_no_embed():
    from castlerag.ui.callbacks import _serialize_group

    mirror = YouTubeMirror(mapping={})  # empty -> every triple is a placeholder
    moment = EvidenceMoment(
        moment_id="m0",
        clock_label="12:29",
        place_label="Kitchen",
        camera_count=1,
        aggregate_score=0.5,
        score_caption="x",
        dot_color="#000",
        cameras=[
            CameraAngle(
                camera_id="Luca",
                day="day1",
                hour=12,
                start_seconds=10.0,
                match_score=0.5,
                is_best=True,
            )
        ],
    )
    result = ChatTurnResult(
        answer_text="a",
        route="mixed",
        support_priors={},
        claim=Claim(text="c", support=SupportLevel.PARTIAL),
        moments=[moment],
    )
    group = _serialize_group(
        result,
        group_id="g1",
        iteration=1,
        question="q",
        mirror=mirror,
        is_refinement=False,
    )
    cam = group["moments"][0]["cameras"][0]
    assert cam["embed_url"] is None  # no placeholder video embedded


def test_camera_grid_renders_missing_tile_without_iframe():
    from castlerag.ui.callbacks import _render_camera_grid

    moment = {
        "clock_label": "12:29",
        "cameras": [
            {"camera_id": "Luca", "is_best": True, "match_score": 0.8,
             "embed_url": "https://example.test/embed"},
            {"camera_id": "Klaus", "is_best": False, "match_score": 0.0,
             "embed_url": None},
        ],
    }
    tiles = _render_camera_grid(moment)
    nodes = [n for tile in tiles for n in _walk(tile, [])]
    classnames = [cn for _, cn in nodes]
    type_names = [t for t, _ in nodes]
    # Exactly one real embed (the mapped camera), and a missing tile for the other.
    assert type_names.count("Iframe") == 1
    assert "camera-missing" in classnames


def test_build_app_assembles_layout_and_callbacks():
    pytest.importorskip("dash")
    from castlerag.ui.app import build_app

    app = build_app()
    assert app.layout is not None
    assert app.callback_map  # callbacks registered
    # The Plotly evidence chart is wired as a callback output in the viewer.
    assert any("evidence-figure" in key for key in app.callback_map)


def test_viewer_outputs_match_their_output_specs():
    """`_viewer_outputs` must return exactly one value per declared Output."""
    import plotly.graph_objects as go

    from castlerag.ui.callbacks import _viewer_output_specs, _viewer_outputs

    moment = {
        "moment_id": "m0",
        "clock_label": "12:29",
        "camera_count": 3,
        "cameras": [
            {
                "camera_id": "Luca",
                "match_score": 0.8,
                "is_best": True,
                "embed_url": "https://example.test/embed",
            },
        ],
    }
    group = {
        "is_refinement": False,
        "claim": {"support": "partial"},
        "moments": [moment],
    }
    out = _viewer_outputs(group, moment, {"Luca": {"state": "pending"}}, 1)
    assert len(out) == len(_viewer_output_specs())
    assert any(isinstance(o, go.Figure) for o in out)  # evidence figure present
    # All cameras pending -> the Submit button stays hidden (last output).
    assert out[-1] is True


def test_rejected_cameras_accumulates_across_thread():
    from castlerag.ui.callbacks import _rejected_cameras

    # Two frozen iterations: Kitchen rejected in iter 1, Living2 in iter 2.
    thread = [
        {"reviews": {"m1": {"Kitchen": {"state": "rejected"},
                            "Allie": {"state": "confirmed"}}}},
        {"reviews": {"m2": {"Living2": {"state": "rejected"},
                            "Bjorn": {"state": "flagged"}}}},
    ]
    # In-flight review on the current moment rejects Reading.
    current = {"Reading": {"state": "rejected"}, "Cathal": {"state": "pending"}}
    assert _rejected_cameras(thread, current) == ["Kitchen", "Living2", "Reading"]


def test_rejected_cameras_empty_when_none_rejected():
    from castlerag.ui.callbacks import _rejected_cameras

    thread = [{"reviews": {"m1": {"Allie": {"state": "confirmed"}}}}]
    assert _rejected_cameras(thread, {"Bjorn": {"state": "flagged"}}) == []
    assert _rejected_cameras(None, None) == []


def test_basic_auth_gates_requests_when_env_set(monkeypatch):
    monkeypatch.setenv("CASTLERAG_UI_BASIC_AUTH", "demo:secret")
    from dash import Dash, html

    from castlerag.ui.app import _install_basic_auth

    app = Dash(__name__)
    app.layout = html.Div("ok")
    _install_basic_auth(app)
    client = app.server.test_client()
    assert client.get("/").status_code == 401  # no creds -> challenged
    import base64

    token = base64.b64encode(b"demo:secret").decode()
    ok = client.get("/", headers={"Authorization": f"Basic {token}"})
    assert ok.status_code != 401  # correct creds pass the gate
    bad = base64.b64encode(b"demo:wrong").decode()
    assert client.get(
        "/", headers={"Authorization": f"Basic {bad}"}
    ).status_code == 401


def test_basic_auth_absent_when_env_unset(monkeypatch):
    monkeypatch.delenv("CASTLERAG_UI_BASIC_AUTH", raising=False)
    from dash import Dash, html

    from castlerag.ui.app import _install_basic_auth

    app = Dash(__name__)
    app.layout = html.Div("ok")
    _install_basic_auth(app)
    assert app.server.test_client().get("/").status_code != 401  # no gate


def test_should_converge_true_when_all_confirmed_or_ignored():
    from castlerag.ui.callbacks import _should_converge

    assert _should_converge(
        {"Luca": {"state": "confirmed"}, "Klaus": {"state": "ignored"}}
    )
    assert _should_converge(
        {"Luca": {"state": "confirmed"}, "Klaus": {"state": "confirmed"}}
    )


def test_should_converge_false_when_any_flagged_or_rejected():
    from castlerag.ui.callbacks import _should_converge

    assert not _should_converge(
        {"Luca": {"state": "confirmed"}, "Klaus": {"state": "flagged"}}
    )
    assert not _should_converge(
        {"Luca": {"state": "ignored"}, "Klaus": {"state": "rejected"}}
    )
    assert not _should_converge({})


def test_strip_internal_links_removes_non_http_refs():
    from castlerag.ui.callbacks import _strip_internal_links

    text = (
        "The answer is clear from "
        "[camera=Luca time=day1 11:39:53-11:42:30](camera=Luca time=day1 "
        "11:39:53-11:42:30) and also from "
        "[transcript window camera=Cathal day=day1 11:18:24-11:19:26]"
        "(transcript window camera=Cathal day=day1 11:18:24-11:19:26)."
    )
    stripped = _strip_internal_links(text)
    assert "camera=Luca" in stripped  # text kept
    assert "11:39:53" in stripped     # timestamp kept
    assert "](camera=" not in stripped  # link markup gone
    assert "https://" not in stripped   # no links remain


def test_strip_internal_links_preserves_real_urls():
    from castlerag.ui.callbacks import _strip_internal_links

    text = "See [YouTube](https://www.youtube.com/watch?v=abc) for details."
    assert _strip_internal_links(text) == text
