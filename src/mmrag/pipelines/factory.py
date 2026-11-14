"""Build a retrieval pipeline from a plain configuration.

One place that knows how the pieces fit, used by the experiment runner and by
the leaderboard adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PIPELINES = ("text", "visual")


@dataclass
class PipelineSpec:
    pipeline: str = "text"
    top_k: int = 10
    # text route (LightRAG naive)
    chunk_top_k: int = 60
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    # visual route
    visual_model: str = "vidore/colSmol-256M"
    # Directory with pages.npz / queries.npz from mmrag.visual.encode_cli. Empty: encode
    # here with the model (needs torch + colpali-engine in the current environment).
    visual_store: str = ""
    quantization: str = "float16"
    pool_factor: int = 1

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


def load_precomputed(visual_store: str):
    """(page store, query encoder) from an encode_cli output directory."""
    from mmrag.visual.encoder import PrecomputedQueryEncoder
    from mmrag.visual.store import MultiVectorStore

    root = Path(visual_store)
    pages, queries = root / "pages.npz", root / "queries.npz"
    if not pages.exists() or not queries.exists():
        raise FileNotFoundError(
            f"{root} must contain pages.npz and queries.npz (run mmrag.visual.encode_cli)"
        )
    return MultiVectorStore.load(pages), PrecomputedQueryEncoder(MultiVectorStore.load(queries))


def build_visual_route(
    spec: PipelineSpec,
    work_dir: Path,
    data: Any | None = None,
    visual_encoder: Any | None = None,
    visual_pages: Any | None = None,
    quantization: str | None = None,
    pool_factor: int | None = None,
    store_name: str = "visual/pages.npz",
):
    from mmrag.visual.retriever import VisualRetriever

    precomputed = visual_pages
    encoder = visual_encoder
    if encoder is None and spec.visual_store:
        precomputed, encoder = load_precomputed(spec.visual_store)
    elif encoder is None:
        from mmrag.visual.encoder import ColpaliEncoder

        encoder = ColpaliEncoder(spec.visual_model)
    return VisualRetriever(
        encoder=encoder,
        store_path=work_dir / store_name,
        precomputed=precomputed,
        quantization=quantization or spec.quantization,
        pool_factor=pool_factor or spec.pool_factor,
        image_provider=data.iter_page_images if data is not None else None,
    )


def build_pipeline(
    spec: PipelineSpec,
    work_dir: str | Path,
    embedder: Any | None = None,
    data: Any | None = None,
    visual_encoder: Any | None = None,
    visual_pages: Any | None = None,
    **_,
):
    """Return an object with index / search / storage_bytes (see retrieval.base.Retriever).

    ``embedder``, ``visual_encoder`` and ``visual_pages`` override the models that would
    otherwise be created from the environment; tests pass offline stand-ins. ``data`` is
    the loaded dataset, used to read page images on demand.
    """
    work_dir = Path(work_dir)
    if spec.pipeline == "text":
        return build_text_route(spec, work_dir, embedder)
    if spec.pipeline == "visual":
        return build_visual_route(spec, work_dir, data, visual_encoder, visual_pages)
    raise ValueError(f"Unknown pipeline '{spec.pipeline}'. Choose from {PIPELINES}")
