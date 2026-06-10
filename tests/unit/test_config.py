"""Tests for src/castlerag/config.py"""

from pathlib import Path

import pytest

from castlerag.config import CastleRAGConfig, _deep_merge, load_config


def test_default_config_is_valid():
    cfg = CastleRAGConfig()
    assert cfg.preprocessing.clip_seconds == 30
    assert cfg.preprocessing.stride_seconds == 30
    assert cfg.preprocessing.fps == 1
    assert cfg.dataset.camera_scope == "ego"
    assert cfg.generation.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg.embedding.model == "Tevatron/OmniEmbed-v0.1-multivent"
    assert cfg.qdrant.collection == "castle_multimodal_v1"
    assert cfg.retrieval.max_evidence_rows == 50
    assert cfg.lora.enabled is False


def test_default_ego_cameras():
    cfg = CastleRAGConfig()
    assert len(cfg.dataset.ego_cameras) == 10
    assert "Allie" in cfg.dataset.ego_cameras
    assert len(cfg.dataset.exo_cameras) == 5


def test_evidence_budget_defaults():
    cfg = CastleRAGConfig()
    assert cfg.retrieval.transcript_top_k == 30
    assert cfg.retrieval.max_candidate_videos == 4
    assert cfg.retrieval.frames_per_candidate == 32
    assert cfg.retrieval.max_aux_images == 16
    assert cfg.retrieval.max_evidence_rows == 50


def test_reranking_weights_sum_to_one():
    cfg = CastleRAGConfig()
    total = cfg.reranking.relevance_weight + cfg.reranking.support_weight
    assert abs(total - 1.0) < 1e-9


def test_generation_and_reranking_use_same_model():
    cfg = CastleRAGConfig()
    assert cfg.generation.model == cfg.reranking.model


def test_ablation_model_set():
    cfg = CastleRAGConfig()
    assert "InternVL3" in cfg.generation.ablation_model


def test_lora_blocked_by_default():
    cfg = CastleRAGConfig()
    assert cfg.lora.enabled is False, "LoRA must stay blocked until QA split is found"


def test_deep_merge_basic():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99}, "c": 4}
    merged = _deep_merge(base, override)
    assert merged["a"]["x"] == 1
    assert merged["a"]["y"] == 99
    assert merged["b"] == 3
    assert merged["c"] == 4


def test_deep_merge_nested():
    base = {"a": {"b": {"c": 1, "d": 2}}}
    override = {"a": {"b": {"d": 99}}}
    merged = _deep_merge(base, override)
    assert merged["a"]["b"]["c"] == 1
    assert merged["a"]["b"]["d"] == 99


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    _deep_merge(base, override)
    assert "y" not in base["a"]


def test_load_config_from_base_yaml():
    base_path = Path(__file__).parent.parent.parent / "configs" / "base.yaml"
    if not base_path.exists():
        pytest.skip("configs/base.yaml not found")
    cfg = load_config(base_path=base_path)
    assert cfg.preprocessing.clip_seconds == 30
    assert cfg.dataset.camera_scope == "ego"
    assert cfg.qdrant.collection == "castle_multimodal_v1"


def test_load_config_with_nonexistent_override():
    """Missing override file should not raise — just use base."""
    base_path = Path(__file__).parent.parent.parent / "configs" / "base.yaml"
    if not base_path.exists():
        pytest.skip("configs/base.yaml not found")
    cfg = load_config(base_path=base_path, override_path="/nonexistent/path.yaml")
    assert cfg.preprocessing.clip_seconds == 30


def test_load_config_explicit_missing_base_raises():
    """Explicit base_path that does not exist must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(base_path="/nonexistent/config.yaml")


def test_expand_env_vars(tmp_path: Path):
    """$VAR references in YAML values must be expanded after load."""
    import os

    from castlerag.config import _expand_env

    os.environ["_CR_TEST_VAR"] = "expanded_value"
    result = _expand_env({"path": "/scratch/$_CR_TEST_VAR/data"})
    assert "expanded_value" in result["path"]
