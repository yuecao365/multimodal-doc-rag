"""Leaderboard adapter: our pipelines as a vidore-benchmark ``BasePipeline``.

    vidore-benchmark pipeline evaluate \\
        --dataset-name vidore/vidore_v3_hr --language english \\
        --module-path src/mmrag/adapters/vidore_pipeline.py --class-name MMRAGPipeline \\
        --pipeline-args '{"pipeline": "hybrid", "work_dir": "rag_storage/leaderboard_hr"}'

Constructor arguments are the fields of ``mmrag.pipelines.factory.PipelineSpec``
plus ``work_dir``. vidore-benchmark is not a dependency of this project; when it
is not installed the class still works as a plain object with the same methods.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

from mmrag.pipelines.factory import PipelineSpec, build_pipeline

try:  # pragma: no cover - depends on the optional vidore-benchmark install
    from vidore_benchmark.pipeline_evaluation import BasePipeline
except ImportError:  # pragma: no cover

    class BasePipeline:  # type: ignore[no-redef]
        pass


class MMRAGPipeline(BasePipeline):
    def __init__(self, work_dir: str = "rag_storage/leaderboard", embedder: Any = None, **spec):
        from dotenv import load_dotenv

        load_dotenv()
        known = {f.name for f in dataclasses.fields(PipelineSpec)}
        unknown = set(spec) - known
        if unknown:
            raise TypeError(f"Unknown pipeline arguments: {sorted(unknown)}")
        self.spec = PipelineSpec(**spec)
        self.work_dir = Path(work_dir)
        self._embedder = embedder
        self._pipeline = None
        self._index_seconds = 0.0

    def index(self, corpus_ids, corpus_images, corpus_texts, dataset_name: str | None = None):
        work_dir = self.work_dir
        if dataset_name:
            work_dir = work_dir / dataset_name.replace("/", "_")
        models = {"embedder": self._embedder} if self._embedder is not None else {}
        self._pipeline = build_pipeline(self.spec, work_dir, **models)
        t0 = time.perf_counter()
        self._pipeline.index(
            [str(c) for c in corpus_ids], corpus_images, list(corpus_texts), dataset_name
        )
        self._index_seconds = time.perf_counter() - t0

    def search(self, query_ids, queries):
        if self._pipeline is None:
            raise RuntimeError("index() must be called before search()")
        t0 = time.perf_counter()
        run = self._pipeline.search(
            [str(q) for q in query_ids], list(queries), top_k=self.spec.top_k
        )
        infos = {
            "total_indexing_time_seconds": round(self._index_seconds, 2),
            "total_retrieval_time_seconds": round(time.perf_counter() - t0, 2),
            "index_storage_bytes": self._pipeline.storage_bytes(),
            "pipeline": self.spec.to_dict(),
        }
        return run, infos
