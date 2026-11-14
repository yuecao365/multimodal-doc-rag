"""Build a retrieval pipeline from a plain configuration.

One place that knows how the pieces fit, used by the experiment runner and by
the leaderboard adapter.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PIPELINES = ("text", "bm25", "visual", "hybrid")


@dataclass
class PipelineSpec:
    pipeline: str = "text"
    top_k: int = 10
    # text route (LightRAG naive)
    chunk_top_k: int = 60
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    # visual route
    visual_model: str = "vidore/colqwen2.5-v0.2"
    # Directory with pages.npz / queries.npz from mmrag.visual.encode_cli. Empty: encode
    # here with the model (needs torch + colpali-engine in the current environment).
    visual_store: str = ""
    quantization: str = "float16"
    pool_factor: int = 1
    # Encode queries that are not in queries.npz (rewrites, agent searches) with the
    # local model, in the visual environment named by VISUAL_PYTHON.
    local_query_encoder: bool = False
    # keyword route: analyzer language (stemmer, stopwords); "french" for finance_fr
    bm25_language: str = "english"
    # hybrid
    routes: str = "text,bm25,visual"  # comma-separated; drop one for an ablation
    routing: str = "analyzer"  # "analyzer": route each page; "all": every route gets every page
    rrf_k: int = 60
    candidates: int = 50  # pages taken from each route before fusion
    analyzer_images: bool = True  # let the page analyzer look at page images
    min_text_chars: int = 300
    max_table_ratio: float = 0.4
    max_color_ratio: float = 0.06
    # reranking of the fused candidates (hybrid only)
    rerank: str = "none"  # "none", "maxsim" (full-precision late interaction) or "llm"
    rerank_top_n: int = 20

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


def build_query_worker(model: str):
    """Query encoder process: the stored vectors' model, on CPU in bfloat16."""
    from mmrag.visual.encoder import WorkerQueryEncoder

    python = os.getenv("VISUAL_PYTHON")
    if not python:
        raise RuntimeError("Set VISUAL_PYTHON in .env to the Python of the visual environment")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    if os.getenv("VISUAL_HF_HOME"):
        env["HF_HOME"] = os.environ["VISUAL_HF_HOME"]
    return WorkerQueryEncoder([python, "-m", "mmrag.visual.query_worker", "--model", model], env)


def load_precomputed(visual_store: str, local_query_encoder: bool = False):
    """(page store, query encoder) from an encode_cli output directory."""
    from mmrag.visual.encoder import PrecomputedQueryEncoder
    from mmrag.visual.store import MultiVectorStore

    root = Path(visual_store)
    pages, queries = root / "pages.npz", root / "queries.npz"
    if not pages.exists() or not queries.exists():
        raise FileNotFoundError(
            f"{root} must contain pages.npz and queries.npz (run mmrag.visual.encode_cli)"
        )
    page_store = MultiVectorStore.load(pages)
    fallback = build_query_worker(page_store.meta["model"]) if local_query_encoder else None
    return page_store, PrecomputedQueryEncoder(MultiVectorStore.load(queries), fallback)


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
        precomputed, encoder = load_precomputed(spec.visual_store, spec.local_query_encoder)
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


def build_bm25_route(spec: PipelineSpec, work_dir: Path):
    from mmrag.retrieval.bm25 import BM25Retriever

    return BM25Retriever(store_path=work_dir / "bm25" / "index.json", language=spec.bm25_language)


def build_reranker(
    spec: PipelineSpec,
    visual_encoder: Any | None = None,
    visual_pages: Any | None = None,
    rerank_client: Any | None = None,
):
    """The second-stage reranker named by ``spec.rerank``, or None."""
    from mmrag.rerank.rerankers import LLMReranker, MaxSimReranker

    if spec.rerank == "none":
        return None
    if spec.rerank == "llm":
        if rerank_client is None:
            from mmrag.models import ChatClient

            rerank_client = ChatClient.from_env("RERANK_MODEL", max_tokens=400)
        return LLMReranker(rerank_client)
    if spec.rerank == "maxsim":
        if visual_pages is None or visual_encoder is None:
            if not spec.visual_store:
                raise ValueError(
                    "--rerank maxsim needs --visual-store: full-precision vectors of every page"
                )
            visual_pages, visual_encoder = load_precomputed(
                spec.visual_store, spec.local_query_encoder
            )
        return MaxSimReranker(visual_encoder, visual_pages)
    raise ValueError(f"Unknown reranker '{spec.rerank}'. Choose none, maxsim or llm")


def build_hybrid(
    spec: PipelineSpec,
    work_dir: Path,
    embedder: Any | None = None,
    data: Any | None = None,
    visual_encoder: Any | None = None,
    visual_pages: Any | None = None,
    rerank_client: Any | None = None,
):
    from mmrag.analysis.page_analyzer import AnalyzerConfig, PageAnalyzer
    from mmrag.pipelines.hybrid import ROUTES, HybridPipeline

    routes = [r.strip() for r in spec.routes.split(",") if r.strip()]
    unknown = set(routes) - set(ROUTES)
    if unknown:
        raise ValueError(f"Unknown routes {sorted(unknown)}; choose from {ROUTES}")
    analyzer = PageAnalyzer(
        AnalyzerConfig(
            min_text_chars=spec.min_text_chars,
            max_table_ratio=spec.max_table_ratio,
            max_color_ratio=spec.max_color_ratio,
        )
    )
    return HybridPipeline(
        text=build_text_route(spec, work_dir, embedder) if "text" in routes else None,
        bm25=build_bm25_route(spec, work_dir) if "bm25" in routes else None,
        visual=(
            build_visual_route(spec, work_dir, data, visual_encoder, visual_pages)
            if "visual" in routes
            else None
        ),
        analyzer=analyzer,
        routing=spec.routing,
        rrf_k=spec.rrf_k,
        candidates=spec.candidates,
        reranker=build_reranker(spec, visual_encoder, visual_pages, rerank_client),
        rerank_top_n=spec.rerank_top_n,
        image_provider=data.iter_page_images if data is not None else None,
        analyzer_uses_images=spec.analyzer_images,
        work_dir=work_dir,
    )


def build_pipeline(
    spec: PipelineSpec,
    work_dir: str | Path,
    embedder: Any | None = None,
    data: Any | None = None,
    visual_encoder: Any | None = None,
    visual_pages: Any | None = None,
    rerank_client: Any | None = None,
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
    if spec.pipeline == "bm25":
        return build_bm25_route(spec, work_dir)
    if spec.pipeline == "hybrid":
        return build_hybrid(
            spec, work_dir, embedder, data, visual_encoder, visual_pages, rerank_client
        )
    raise ValueError(f"Unknown pipeline '{spec.pipeline}'. Choose from {PIPELINES}")
