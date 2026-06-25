"""Tests for the optional W&B eval logger (no network / wandb import required)."""

from __future__ import annotations

from castlerag.eval.wandb_logger import _stringify_keys


def test_stringify_keys_coerces_int_keys():
    # The diversity histogram is int-keyed (camera_count_distribution); wandb's
    # summary encoder concatenates nested keys as strings and crashes on ints.
    assert _stringify_keys({2: 2, 5: 1, 1: 2}) == {"2": 2, "5": 1, "1": 2}


def test_stringify_keys_recurses_into_nested_dicts_and_lists():
    nested = {"d": {1: {2: 3}}, "xs": [{4: 5}]}
    assert _stringify_keys(nested) == {"d": {"1": {"2": 3}}, "xs": [{"4": 5}]}


def test_stringify_keys_passes_scalars_through():
    assert _stringify_keys(0.6) == 0.6
    assert _stringify_keys("ok") == "ok"
    assert _stringify_keys(None) is None


def test_stringify_keys_preserves_real_diversity_payload():
    diversity = {
        "mean_cameras_per_question": 2.2,
        "pct_multi_camera": 0.6,
        "camera_count_distribution": {2: 2, 5: 1, 1: 2},
    }
    out = _stringify_keys(diversity)
    assert out["mean_cameras_per_question"] == 2.2
    assert out["camera_count_distribution"] == {"2": 2, "5": 1, "1": 2}
    # All nested keys are now strings (the property wandb's encoder requires).
    assert all(isinstance(k, str) for k in out["camera_count_distribution"])
