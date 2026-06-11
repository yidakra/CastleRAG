"""Tests for src/castlerag/preprocess/visual_summary.py"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from castlerag.preprocess.visual_summary import generate_visual_summary

# ---------------------------------------------------------------------------
# generate_visual_summary — edge cases
# ---------------------------------------------------------------------------


def test_empty_frame_paths_returns_empty_string():
    result = generate_visual_summary(
        frame_paths=[],
        transcript_text="Some text",
        model_name="llava",
        vllm_base_url="http://localhost:8000/v1",
    )
    assert result == ""


def test_missing_vllm_base_url_raises_value_error():
    with pytest.raises(ValueError, match="vllm_base_url"):
        generate_visual_summary(
            frame_paths=[],
            transcript_text=None,
            model_name="llava",
            vllm_base_url=None,
        )


def test_none_vllm_base_url_raises_before_empty_check():
    """vllm_base_url validation must fire even when frame_paths is empty."""
    with pytest.raises(ValueError, match="vllm_base_url"):
        generate_visual_summary(
            frame_paths=[],
            transcript_text=None,
            model_name="llava",
            vllm_base_url=None,
        )


# ---------------------------------------------------------------------------
# generate_visual_summary — normal call
# ---------------------------------------------------------------------------


def _make_frame(path: Path) -> Path:
    """Write minimal JPEG-like bytes so read_bytes() returns something."""
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    return path


def test_single_frame_passes_correct_content_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frame = _make_frame(tmp_path / "frame_0001.jpg")
    captured = {}

    def fake_vllm_chat(vllm_base_url, model_name, messages, **kwargs):
        captured["messages"] = messages
        return "A person walks through a kitchen."

    monkeypatch.setattr(
        "castlerag.preprocess.visual_summary._vllm_chat", fake_vllm_chat
    )

    result = generate_visual_summary(
        frame_paths=[frame],
        transcript_text=None,
        model_name="llava",
        vllm_base_url="http://localhost:8000/v1",
    )

    assert result == "A person walks through a kitchen."
    messages = captured["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    # Last item must be the text prompt
    assert content[-1]["type"] == "text"
    # First item must be an image_url with base64 data
    assert content[0]["type"] == "image_url"
    expected_b64 = base64.b64encode(frame.read_bytes()).decode()
    assert content[0]["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"


def test_transcript_text_appended_to_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frame = _make_frame(tmp_path / "frame_0001.jpg")
    captured = {}

    def fake_vllm_chat(vllm_base_url, model_name, messages, **kwargs):
        captured["messages"] = messages
        return "Summary."

    monkeypatch.setattr(
        "castlerag.preprocess.visual_summary._vllm_chat", fake_vllm_chat
    )

    generate_visual_summary(
        frame_paths=[frame],
        transcript_text="Hello everyone",
        model_name="llava",
        vllm_base_url="http://localhost:8000/v1",
    )

    text_item = captured["messages"][0]["content"][-1]
    assert "Hello everyone" in text_item["text"]


def test_more_than_8_frames_samples_only_8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frames = [_make_frame(tmp_path / f"frame_{i:04d}.jpg") for i in range(20)]
    captured = {}

    def fake_vllm_chat(vllm_base_url, model_name, messages, **kwargs):
        captured["messages"] = messages
        return "Summary."

    monkeypatch.setattr(
        "castlerag.preprocess.visual_summary._vllm_chat", fake_vllm_chat
    )

    generate_visual_summary(
        frame_paths=frames,
        transcript_text=None,
        model_name="llava",
        vllm_base_url="http://localhost:8000/v1",
    )

    content = captured["messages"][0]["content"]
    # Last element is the text prompt; all others are images
    image_items = [c for c in content if c["type"] == "image_url"]
    assert len(image_items) == 8


def test_exactly_8_frames_all_included(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frames = [_make_frame(tmp_path / f"frame_{i:04d}.jpg") for i in range(8)]
    captured = {}

    def fake_vllm_chat(vllm_base_url, model_name, messages, **kwargs):
        captured["messages"] = messages
        return "Summary."

    monkeypatch.setattr(
        "castlerag.preprocess.visual_summary._vllm_chat", fake_vllm_chat
    )

    generate_visual_summary(
        frame_paths=frames,
        transcript_text=None,
        model_name="llava",
        vllm_base_url="http://localhost:8000/v1",
    )

    content = captured["messages"][0]["content"]
    image_items = [c for c in content if c["type"] == "image_url"]
    assert len(image_items) == 8


def test_vllm_chat_receives_correct_url_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frame = _make_frame(tmp_path / "frame_0001.jpg")
    captured = {}

    def fake_vllm_chat(vllm_base_url, model_name, messages, **kwargs):
        captured["url"] = vllm_base_url
        captured["model"] = model_name
        return "Done."

    monkeypatch.setattr(
        "castlerag.preprocess.visual_summary._vllm_chat", fake_vllm_chat
    )

    generate_visual_summary(
        frame_paths=[frame],
        transcript_text=None,
        model_name="my-llava-model",
        vllm_base_url="http://gpu-host:8080/v1",
    )

    assert captured["url"] == "http://gpu-host:8080/v1"
    assert captured["model"] == "my-llava-model"
