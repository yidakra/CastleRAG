"""Pydantic config models and YAML loader for CastleRAG."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, get_args

import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    root: str = "/data/castle2024"
    hf_repo: str = "CASTLE-Dataset/CASTLE2024"
    days: List[int] = Field(default=[1, 2, 3, 4])
    # 11 egocentric participant cameras (real CASTLE stream names)
    ego_cameras: List[str] = Field(
        default=[
            "Allie",
            "Bao",
            "Bjorn",
            "Cathal",
            "Florian",
            "Klaus",
            "Luca",
            "Onanong",
            "Stevan",
            "Tien",
            "Werner",
        ]
    )
    # 5 fixed room cameras — extension only, not in baseline
    exo_cameras: List[str] = Field(
        default=[
            "Kitchen",
            "Living1",
            "Living2",
            "Meeting",
            "Reading",
        ]
    )
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
    # Old frame-path prefix -> new prefix. Chunks and Qdrant payloads store
    # absolute frame paths; after the frames move (e.g. scratch -> project
    # space) readers try the new prefix when a stored path no longer exists.
    frame_path_aliases: Dict[str, str] = Field(default_factory=dict)


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
    # Base URL of the OmniEmbed embeddings server. Set this in two-endpoint
    # deployments (UI / eval / answer) where embeddings are served separately
    # from generation — e.g. OmniEmbed on :8200 and the Qwen3-VL chat model on
    # :8201 — so query embeddings never get POSTed to the generation server
    # (which 404s). When None, the runtime falls back to OMNIEMBED_BASE_URL then
    # VLLM_BASE_URL, preserving single-endpoint setups. See issue #54.
    base_url: Optional[str] = None


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection: str = "castle_multimodal_v1"
    vector_size: Optional[int] = None  # discovered from first batch
    distance: str = "Cosine"
    on_disk_payload: bool = True
    # Where the Qdrant server keeps its data; read by the Slurm scripts via
    # `castlerag paths` (the server itself is configured by environment).
    storage_path: Optional[str] = None


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
    # Post-fusion evidence budgets applied in retrieval.search._collapse_hits.
    # Clips (main_clip) and event summaries (main_event_summary) are budgeted
    # separately; min_clip_hits reserves rows for the top clips on every route
    # so frames always reach the reranker/generator when clips were retrieved.
    # All budgets are non-negative: a negative value would turn into a
    # Python drop-last-N slice in _collapse_hits and silently starve a lane.
    max_candidate_videos: int = Field(default=4, ge=0)
    max_event_summaries: int = Field(default=4, ge=0)
    min_clip_hits: int = Field(default=2, ge=0)
    frames_per_candidate: int = Field(default=32, ge=0)
    max_aux_images: int = Field(default=16, ge=0)
    max_evidence_rows: int = Field(default=50, ge=0)
    modality_score_thresholds: Dict[str, float] = Field(default_factory=dict)
    # Visual-text lexical lane (issue #50, modality gap): BM25 over per-clip
    # captions + OCR + scene-graph text and per-event summaries + aggregated
    # OCR, built by `castlerag index` as visual_text.json. `visual_text_top_k`
    # is the lane size; `visual_text_route_weights` is its RRF weight in the
    # multimodal fusion pass per question route (dense lanes weigh 1.0/0.7/0.9
    # per query variant). Only used when visual_text.json exists.
    visual_text_top_k: int = 20
    visual_text_route_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "static_visual": 2.0,
            "mixed": 1.5,
            "temporal": 1.0,
            "speech_text": 0.5,
        }
    )

    @field_validator("visual_text_route_weights")
    @classmethod
    def _check_visual_route_weights(cls, value: Dict[str, float]) -> Dict[str, float]:
        from castlerag.schemas import QuestionRoute

        routes = set(get_args(QuestionRoute))
        for route, weight in value.items():
            if route not in routes:
                raise ValueError(
                    f"visual_text_route_weights: unknown route {route!r} "
                    f"(expected one of {sorted(routes)})"
                )
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"visual_text_route_weights[{route!r}] must be a finite "
                    f"float >= 0, got {weight}"
                )
        return value


class GenerationConfig(BaseModel):
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    ablation_model: str = "OpenGVLab/InternVL3-8B"
    backend: Literal["vllm", "transformers"] = "vllm"
    # The MCQ generator answers first (FINAL_ANSWER line) then justifies; this
    # budget must cover that justification. 512 truncated long chain-of-thought
    # before the answer line was ever emitted, so 13/19 questions were graded on
    # a fallback guess instead of the model's real answer.
    max_new_tokens: int = 1024
    temperature: float = 0.0
    vllm_tensor_parallel: int = 1
    vllm_gpu_memory_utilization: float = 0.90
    # Multimodal prompt budgeting. Frames are downscaled to `frame_max_pixels` on
    # their longest edge before encoding, then packed into the generation prompt
    # only while the running token estimate stays under `prompt_token_budget`
    # (which must leave headroom below the served --max-model-len for the
    # `max_new_tokens` completion). Without this, full-resolution frames produced
    # 66k-72k-token prompts that the server rejected (>49k context).
    max_frames: int = 8
    frame_max_pixels: int = 768
    prompt_token_budget: int = 40000
    # When True, generate_answer presents the four answer choices in a
    # deterministic per-question permutation (sha1(question_id)) and maps the
    # model's predicted letter back to the original letter.  Counteracts the
    # late-position bias that small multiple-choice models exhibit on weak
    # evidence (Qwen3-VL-4B clusters predictions on 'd' otherwise).
    shuffle_choices: bool = False


class RerankingConfig(BaseModel):
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    top_k: int = 4
    relevance_weight: float = 0.7
    support_weight: float = 0.3
    # Packs with relevance <= min_relevance are pruned. Per-route overrides
    # (e.g. {"static_visual": 0}) win over the global value for that route.
    min_relevance: int = Field(default=1, ge=0, le=4)
    min_relevance_by_route: Dict[str, int] = Field(default_factory=dict)
    # keep=false from the reranker only discards packs with relevance at or
    # below this value; higher-rated packs survive a stray keep=false. 4
    # restores the old behaviour where keep was always decisive.
    keep_gate_max_relevance: int = Field(default=1, ge=0, le=4)

    @field_validator("min_relevance_by_route")
    @classmethod
    def _check_route_overrides(cls, value: Dict[str, int]) -> Dict[str, int]:
        from castlerag.schemas import QuestionRoute

        routes = set(get_args(QuestionRoute))
        for route, threshold in value.items():
            if route not in routes:
                raise ValueError(
                    f"min_relevance_by_route: unknown route {route!r} "
                    f"(expected one of {sorted(routes)})"
                )
            if not 0 <= threshold <= 4:
                raise ValueError(
                    f"min_relevance_by_route[{route!r}] must be within 0-4, "
                    f"got {threshold}"
                )
        return value

    def min_relevance_for(self, route: str) -> int:
        """Return the min_relevance threshold in force for ``route``."""
        return self.min_relevance_by_route.get(route, self.min_relevance)


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


class RoutingConfig(BaseModel):
    use_llm_hints: bool = False
    model: str = "Qwen/Qwen3-VL-8B-Instruct"


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "castlerag"
    entity: str = ""
    run_name: str = ""


class UIConfig(BaseModel):
    # Controls which score is shown in the dashboard's per-camera bar chart.
    # rrf_normalized: RRF score normalised to [0,1] relative to the top moment.
    # cosine:         Raw cosine similarity from Qdrant before RRF (always in [0,1]).
    # reranker:       VLM-assessed relevance from the reranker, normalised to [0,1].
    #                 Falls back to rrf_normalized when the reranker did not run.
    score_mode: Literal["rrf_normalized", "cosine", "reranker"] = "rrf_normalized"


class CastleRAGConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
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
        # Keys too: frame_path_aliases maps path prefixes that may use $USER.
        return {
            (os.path.expandvars(k) if isinstance(k, str) else k): _expand_env(v)
            for k, v in obj.items()
        }
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
    cfg = CastleRAGConfig.model_validate(data)
    # Frame readers (retrieval, generation) have no config handle; register the
    # aliases process-wide so stored scratch paths resolve after a move.
    from castlerag.frame_encoding import set_frame_path_aliases

    set_frame_path_aliases(cfg.preprocessing.frame_path_aliases)
    return cfg
