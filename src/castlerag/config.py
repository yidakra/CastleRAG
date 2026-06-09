"""Pydantic config models and YAML loader for CastleRAG."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    root: str = "/data/castle2024"
    hf_repo: str = "castle-challenge/castle2024"
    days: List[int] = Field(default=[1, 2, 3, 4])
    # 10 egocentric participant cameras (TAHAKOM-validated baseline scope)
    ego_cameras: List[str] = Field(default=[
        "Allie", "Bjorn", "Celine", "Deon", "Estella",
        "Finn", "Greta", "Harvey", "Isla", "Jian",
    ])
    # 5 fixed room cameras — extension only, not in baseline
    exo_cameras: List[str] = Field(default=[
        "Kitchen", "Living1", "Living2", "Office", "Hallway",
    ])
    camera_scope: Literal["ego", "all"] = "ego"
    hours: List[int] = Field(default=list(range(8, 21)))


class PreprocessingConfig(BaseModel):
    clip_seconds: int = 30
    stride_seconds: int = 30
    fps: int = 1
    placeholder_frame_threshold: float = 0.80
    max_transcript_window_seconds: int = 15
    max_transcript_tokens: int = 96
    frames_dir: str = "data/derived/frames_1fps"
    clips_dir: str = "data/derived/clips"
    manifests_dir: str = "data/manifests"
    chunks_dir: str = "data/derived/chunks"


class EmbeddingBatchSizes(BaseModel):
    transcript: int = 128
    event_summary: int = 64
    image: int = 16
    video: int = 4


class EmbeddingConfig(BaseModel):
    model: str = "Tevatron/OmniEmbed-v0.1-multivent"
    backend: Literal["vllm", "transformers"] = "vllm"
    batch_sizes: EmbeddingBatchSizes = Field(default_factory=EmbeddingBatchSizes)
    cache_dir: str = "data/derived/embeddings"
    vllm_tensor_parallel: int = 1
    vllm_gpu_memory_utilization: float = 0.90


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection: str = "castle_multimodal_v1"
    vector_size: Optional[int] = None  # discovered from first batch
    distance: str = "Cosine"
    on_disk_payload: bool = True


class RetrievalConfig(BaseModel):
    transcript_top_k: int = 30
    event_summary_top_k: int = 20
    video_top_k: int = 20
    photo_top_k: int = 16
    aux_video_top_k: int = 8
    heartrate_top_k: int = 8
    gaze_top_k: int = 8
    thermal_top_k: int = 8
    rrf_k: int = 60
    max_candidate_videos: int = 4
    frames_per_candidate: int = 32
    max_aux_images: int = 16
    max_evidence_rows: int = 50


class GenerationConfig(BaseModel):
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    ablation_model: str = "OpenGVLab/InternVL3-8B"
    backend: Literal["vllm", "transformers"] = "vllm"
    max_new_tokens: int = 512
    temperature: float = 0.0
    vllm_tensor_parallel: int = 1
    vllm_gpu_memory_utilization: float = 0.90


class RerankingConfig(BaseModel):
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    top_k: int = 4
    relevance_weight: float = 0.7
    support_weight: float = 0.3
    min_relevance: int = 1


class OutputsConfig(BaseModel):
    dir: str = "outputs"
    predictions: str = "outputs/predictions.json"
    evidence_traces: str = "outputs/evidence_traces.jsonl"
    submissions: str = "outputs/submissions.json"
    metrics: str = "outputs/metrics.json"


class LoRAConfig(BaseModel):
    # Blocked until a CASTLE QA train/val split is explicitly confirmed to exist
    enabled: bool = False
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    rank: int = 16
    alpha: int = 32
    target_modules: List[str] = Field(default=["q_proj", "v_proj"])
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2.0e-4
    output_dir: str = "data/lora_checkpoints"


class SlurmConfig(BaseModel):
    partition: str = "gpu_a100"
    account: str = ""
    time: str = "04:00:00"
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 18
    mem: str = "120G"
    gpus: int = 1
    mail_type: str = "FAIL"
    mail_user: str = ""


class CastleRAGConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    version: str = "0.1.0"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base, returning a new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _expand_env(obj: Any) -> Any:
    """Recursively expand $VAR / ${VAR} environment variables in string values."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def _default_base_path() -> Path:
    """Resolve the default base.yaml location.

    Checks the installed-package location first (wheel install), then falls
    back to the project-root configs/ directory (editable / dev install).
    """
    pkg_relative = Path(__file__).parent / "configs" / "base.yaml"
    if pkg_relative.exists():
        return pkg_relative
    return Path(__file__).parent.parent.parent / "configs" / "base.yaml"


def load_config(
    base_path: str | Path | None = None,
    override_path: str | Path | None = None,
) -> CastleRAGConfig:
    """Load config from YAML, merging override on top of base.

    Raises FileNotFoundError if an explicit base_path is given but does not
    exist (fail-fast for typos).  A missing override_path is silently skipped
    (it is optional by contract).
    """
    explicit_base = base_path is not None
    if base_path is None:
        base_path = _default_base_path()

    data: Dict[str, Any] = {}
    base_path = Path(base_path)
    if not base_path.exists():
        if explicit_base:
            raise FileNotFoundError(f"Config file not found: {base_path}")
        # Default path missing — proceed with Pydantic defaults
    else:
        with base_path.open() as f:
            data = yaml.safe_load(f) or {}

    if override_path is not None:
        override_path = Path(override_path)
        if override_path.exists():
            with override_path.open() as f:
                override_data = yaml.safe_load(f) or {}
            data = _deep_merge(data, override_data)

    data = _expand_env(data)
    return CastleRAGConfig.model_validate(data)
