"""Hybrid pipeline: per-page routing, three retrieval routes, RRF fusion.

Indexing
  page analyzer -> each page is a text page or a visual page
  text route    <- text pages      (parsed text, dense retrieval)
  visual route  <- visual pages    (page image, late interaction)
  keyword route <- every page      (BM25 over the parsed text)

Search
  each route returns its candidates -> RRF over page ids -> optional rerank

Any route can be left out (ablation). With ``routing="all"`` the analyzer is
skipped and the text and visual routes both index every page, which is the
no-savings upper bound the routed variant is compared against.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from mmrag.analysis.page_analyzer import PageAnalysis, PageAnalyzer, route_counts
from mmrag.retrieval.base import Run, top_k_of
from mmrag.retrieval.fusion import rrf_fuse

logger = logging.getLogger(__name__)

ROUTES = ("text", "bm25", "visual")
ImageProvider = Callable[[list[str]], Iterable[tuple[str, Any]]]


class HybridPipeline:
    name = "hybrid"

    def __init__(
        self,
        text: Any | None = None,
        bm25: Any | None = None,
        visual: Any | None = None,
        analyzer: PageAnalyzer | None = None,
        routing: str = "analyzer",
        rrf_k: int = 60,
        candidates: int = 50,
        weights: dict[str, float] | None = None,
        reranker: Any | None = None,
        rerank_top_n: int = 20,
        image_provider: ImageProvider | None = None,
        analyzer_uses_images: bool = True,
        work_dir: str | Path | None = None,
    ) -> None:
        if routing not in ("analyzer", "all"):
            raise ValueError("routing must be 'analyzer' or 'all'")
        self.routes = {"text": text, "bm25": bm25, "visual": visual}
        self.routes = {name: r for name, r in self.routes.items() if r is not None}
        if not self.routes:
            raise ValueError("At least one route is required")
        self.analyzer = analyzer or PageAnalyzer()
        self.routing = routing
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.weights = weights
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self.image_provider = image_provider
        # The visual route reads the images of its own pages through the same provider.
        if visual is not None and getattr(visual, "image_provider", None) is None:
            visual.image_provider = image_provider
        self.analyzer_uses_images = analyzer_uses_images
        self.work_dir = Path(work_dir) if work_dir else None
        self.analyses: dict[str, PageAnalysis] = {}
        self.page_routes: dict[str, str] = {}
        self.route_runs: dict[str, Run] = {}
        self.fused_run: Run = {}
        self.timings: dict[str, float] = {}
        # Exposed so the runner can report text-embedding usage.
        self.embedder = getattr(text, "embedder", None)
        self.llm = getattr(text, "llm", None)

    # -- indexing ------------------------------------------------------------

    def _images(self, ids: list[str], corpus_ids: list[str], corpus_images: list[Any] | None):
        if corpus_images is not None:
            by_id = dict(zip(corpus_ids, corpus_images))
            return [(pid, by_id[pid]) for pid in ids]
        if self.image_provider is not None:
            return self.image_provider(ids)
        return None

    def _route_pages(self, corpus_ids, corpus_images, corpus_texts) -> None:
        if self.routing == "all":
            self.page_routes = dict.fromkeys(corpus_ids, "both")
            return
        images = None
        if self.analyzer_uses_images:
            images = self._images(corpus_ids, corpus_ids, corpus_images)
        t0 = time.perf_counter()
        self.analyses = self.analyzer.analyze(corpus_ids, corpus_texts, images)
        self.timings["analyze_seconds"] = round(time.perf_counter() - t0, 2)
        self.page_routes = {pid: a.route for pid, a in self.analyses.items()}
        # Routing does not depend on which routes are enabled: with a route switched off
        # (ablation) its pages stay where they were sent and are reachable only by the
        # routes that remain, which is what measures the missing route's contribution.
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            (self.work_dir / "page_analysis.json").write_text(
                json.dumps({p: a.to_dict() for p, a in self.analyses.items()}, ensure_ascii=False),
                encoding="utf-8",
            )

    def index(
        self,
        corpus_ids: list[str],
        corpus_images: list[Any] | None,
        corpus_texts: list[str],
        dataset_name: str | None = None,
    ) -> None:
        corpus_ids = [str(c) for c in corpus_ids]
        corpus_texts = [t or "" for t in corpus_texts]
        self._route_pages(corpus_ids, corpus_images, corpus_texts)
        if self.reranker is not None and hasattr(self.reranker, "prepare"):
            self.reranker.prepare(corpus_ids, corpus_texts, self.image_provider)
        text_of = dict(zip(corpus_ids, corpus_texts))
        text_ids = [p for p in corpus_ids if self.page_routes[p] in ("text", "both")]
        visual_ids = [p for p in corpus_ids if self.page_routes[p] in ("visual", "both")]
        logger.info(
            "routing: %d text pages, %d visual pages, %d total",
            len(text_ids), len(visual_ids), len(corpus_ids),
        )  # fmt: skip

        if "bm25" in self.routes:
            self._timed("bm25", lambda: self.routes["bm25"].index(corpus_ids, None, corpus_texts))
        if "text" in self.routes:
            texts = [text_of[p] for p in text_ids]
            self._timed("text", lambda: self.routes["text"].index(text_ids, None, texts))
        if "visual" in self.routes:
            images = None
            if corpus_images is not None:
                by_id = dict(zip(corpus_ids, corpus_images))
                images = [by_id[p] for p in visual_ids]
            self._timed(
                "visual",
                lambda: self.routes["visual"].index(visual_ids, images, None, dataset_name),
            )

    def _timed(self, name: str, fn: Callable[[], None]) -> None:
        t0 = time.perf_counter()
        fn()
        self.timings[f"index_{name}_seconds"] = round(time.perf_counter() - t0, 2)

    # -- search --------------------------------------------------------------

    def search(self, query_ids: list[str], queries: list[str], top_k: int = 10) -> Run:
        query_ids = [str(q) for q in query_ids]
        depth = max(self.candidates, top_k)
        self.route_runs = {}
        for name, route in self.routes.items():
            t0 = time.perf_counter()
            self.route_runs[name] = route.search(query_ids, list(queries), top_k=depth)
            self.timings[f"search_{name}_seconds"] = round(time.perf_counter() - t0, 2)

        fused: Run = {}
        for qid in query_ids:
            per_route = {name: run.get(qid, {}) for name, run in self.route_runs.items()}
            fused[qid] = rrf_fuse(per_route, k=self.rrf_k, weights=self.weights, top_k=depth)
        self.fused_run = fused

        if self.reranker is None:
            return {qid: top_k_of(scores, top_k) for qid, scores in fused.items()}

        t0 = time.perf_counter()
        final: Run = {}
        for qid, query in zip(query_ids, queries):
            final[qid] = top_k_of(
                self.reranker.rerank(query, fused[qid], top_n=self.rerank_top_n), top_k
            )
        self.timings["rerank_seconds"] = round(time.perf_counter() - t0, 2)
        return final

    # -- reporting -----------------------------------------------------------

    def storage_bytes(self) -> int:
        """First-stage index size. A reranker's own storage is reported in ``info()``."""
        return sum(route.storage_bytes() for route in self.routes.values())

    def info(self) -> dict:
        info: dict[str, Any] = {
            "routes": list(self.routes),
            "routing": self.routing,
            "timings": dict(self.timings),
            "storage_bytes_by_route": {n: r.storage_bytes() for n, r in self.routes.items()},
        }
        if self.analyses:
            info["routing_summary"] = route_counts(self.analyses)
        if self.page_routes:
            info["page_routes"] = dict(self.page_routes)
        for route in self.routes.values():
            if hasattr(route, "info"):
                info.update(route.info())
        if self.route_runs:
            info["route_runs"] = {**self.route_runs, "fused": self.fused_run}
        if self.reranker is not None and hasattr(self.reranker, "info"):
            info["rerank"] = self.reranker.info()
        return info

    def close(self) -> None:
        for route in self.routes.values():
            if hasattr(route, "close"):
                route.close()
