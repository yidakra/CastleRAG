"""Token-budgeted batching for the OmniEmbed server."""

from __future__ import annotations

import pytest

from castlerag.embed.batching import plan_batches


def _check(lengths, max_batch, budget):
    batches = plan_batches(lengths, max_batch, budget)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(len(lengths)))  # every item exactly once
    for b in batches:
        assert len(b) <= max_batch
        if len(b) > 1:
            assert len(b) * max(lengths[i] for i in b) <= budget
    return batches


def test_short_texts_fill_batches_up_to_max_batch():
    batches = _check([40] * 40, max_batch=16, budget=8192)
    assert [len(b) for b in batches] == [16, 16, 8]


def test_a_long_outlier_gets_its_own_small_batch():
    lengths = [60] * 20 + [3500, 1100]
    batches = _check(lengths, max_batch=16, budget=8192)
    assert [20] in batches  # the 3500-token text is not padded with 15 others
    assert all(len(b) <= 7 for b in batches if 21 in b)


def test_item_longer_than_budget_still_runs_alone():
    batches = _check([10000, 20], max_batch=16, budget=8192)
    assert [0] in batches


def test_rejects_bad_limits():
    with pytest.raises(ValueError):
        plan_batches([1], 0, 10)
