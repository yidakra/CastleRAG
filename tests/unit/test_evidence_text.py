"""Tests for the shared caption/scene-graph truncation helper."""

from __future__ import annotations

from castlerag.evidence_text import truncate_text


def test_truncate_text_none_and_blank_return_none():
    assert truncate_text(None, 10) is None
    assert truncate_text("   \n ", 10) is None


def test_truncate_text_collapses_whitespace_and_keeps_short_text():
    assert truncate_text("  a   b\n c ", 10) == "a b c"


def test_truncate_text_cuts_with_ellipsis_at_limit():
    out = truncate_text("abcdefghijklmnop", 10)
    assert out == "abcdefg..."
    assert len(out) == 10


def test_truncate_text_tiny_limit_has_no_ellipsis():
    assert truncate_text("abcdef", 2) == "ab"
