"""Build a retrieval pipeline from a plain configuration.

One place that knows how the pieces fit, used by the experiment runner and by
the leaderboard adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PIPELINES = ("text",)


@dataclass
class PipelineSpec:
    pipeline: str = "text"
    top_k: int = 10
    # text route (LightRAG naive)
    chunk_top_k: int = 60
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100

    def to_dict(self) -> dict:
        return asdict(self)


def build_text_route(spec: PipelineSpec, work_dir: Path, embedder: Any | None = None):
    from mmrag.models import OpenAICompatEmbedder
    from mmrag.pipelines.text_baseline import TextBaselinePipeline

    embedder = embedder or OpenAICompatEmbedder.from_env()
    return TextBaselinePipeline(
        working_dir=str(work_dir / "text"),
        embedder=embedder,
        embedding_dim=getattr(embedder, "dim", None) or 1024,
        embedding_max_tokens=getattr(embedder, "max_tokens", 8192),
        top_k_pages=spec.top_k,
        chunk_top_k=spec.chunk_top_k,
        chunk_token_size=spec.chunk_token_size,
        chunk_overlap_token_size=spec.chunk_overlap_token_size,
    )


def build_pipeline(spec: PipelineSpec, work_dir: str | Path, embedder: Any | None = None, **_):
    """Return an object with index / search / storage_bytes (see retrieval.base.Retriever)."""
    work_dir = Path(work_dir)
    if spec.pipeline == "text":
        return build_text_route(spec, work_dir, embedder)
    raise ValueError(f"Unknown pipeline '{spec.pipeline}'. Choose from {PIPELINES}")
