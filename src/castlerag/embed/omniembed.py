"""OmniEmbed (Tevatron/OmniEmbed-v0.1-multivent) batch inference wrappers.

Backend: vLLM (default) or HuggingFace Transformers.

Modality batching strategy (SPEC §3.2):
  transcript windows : 128 per batch
  event summaries    : 64 per batch
  images             : 16 per batch
  video clips        : 4 per batch

Query format per OmniEmbed model card:
  text  → "Query: {text}"
  media → raw frames/video via Qwen2.5-Omni processor

Point ids are deterministic: sha1(model_version + source_type + record_id + modality)
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import numpy as np


def make_point_id(
    model_version: str,
    source_type: str,
    record_id: str,
    modality: str,
) -> str:
    """Return a deterministic hex point id (SHA-1)."""
    key = f"{model_version}|{source_type}|{record_id}|{modality}"
    return hashlib.sha1(key.encode()).hexdigest()


class OmniEmbedClient:
    """Thin wrapper around a vLLM or Transformers OmniEmbed backend.

    Call embed_texts() or embed_images() or embed_videos() to get
    float32 embedding arrays.  Vector dimensionality is discovered from
    the first successful batch and stored in self.dim.
    """

    def __init__(
        self,
        model: str = "Tevatron/OmniEmbed-v0.1-multivent",
        backend: str = "vllm",
        vllm_base_url: Optional[str] = None,
        vllm_tensor_parallel: int = 1,
        vllm_gpu_memory_utilization: float = 0.90,
    ) -> None:
        self.model = model
        self.backend = backend
        self.vllm_base_url = vllm_base_url
        self.vllm_tensor_parallel = vllm_tensor_parallel
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.dim: Optional[int] = None  # set after first batch
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self.backend == "vllm":
            self._client = self._init_vllm()
        elif self.backend == "transformers":
            self._client = self._init_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def _init_vllm(self) -> Any:
        raise NotImplementedError("vLLM backend initialised in issue #5")

    def _init_transformers(self) -> Any:
        raise NotImplementedError("Transformers backend initialised in issue #5")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed text strings.  Automatically prepends 'Query: ' prefix."""
        raise NotImplementedError("Implemented in issue #5")

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """Embed image files (JPEG/PNG).  Preserves source resolution."""
        raise NotImplementedError("Implemented in issue #5")

    def embed_videos(self, frame_path_lists: List[List[str]]) -> np.ndarray:
        """Embed video clips represented as lists of 1 fps frame paths."""
        raise NotImplementedError("Implemented in issue #5")
