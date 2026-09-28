"""Tests for the shared frame sampling helper."""

from __future__ import annotations

import pytest

from castlerag.frame_encoding import sample_frames_evenly


def test_sample_frames_evenly_spreads_over_a_30s_clip():
    frames = [f"f{i}" for i in range(30)]
    assert sample_frames_evenly(frames, 4) == ["f3", "f11", "f18", "f26"]


def test_sample_frames_evenly_preserves_count_and_order():
    frames = [f"f{i}" for i in range(30)]
    picked = sample_frames_evenly(frames, 8)
    assert len(picked) == 8
    assert picked == sorted(picked, key=lambda s: int(s[1:]))
    assert len(set(picked)) == 8


def test_sample_frames_evenly_returns_all_when_within_cap():
    assert sample_frames_evenly(["a", "b"], 4) == ["a", "b"]
    assert sample_frames_evenly(["a", "b", "c", "d"], 4) == ["a", "b", "c", "d"]


@pytest.mark.parametrize("cap", [0, -1])
def test_sample_frames_evenly_non_positive_cap_is_empty(cap):
    assert sample_frames_evenly(["a", "b"], cap) == []


def test_sample_frames_evenly_single_frame_picks_middle():
    frames = [f"f{i}" for i in range(30)]
    assert sample_frames_evenly(frames, 1) == ["f15"]
