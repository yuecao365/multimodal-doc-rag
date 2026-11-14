"""Shared retriever interface.

Every retrieval route (text, keyword, visual) and every combined pipeline has
the same shape, which is also the shape of vidore-benchmark's ``BasePipeline``:

    index(corpus_ids, corpus_images, corpus_texts)
    search(query_ids, queries) -> {query_id: {corpus_id: score}}

Scores are "higher is better"; within one query they only need to rank.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

Run = dict[str, dict[str, float]]


@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(
        self,
        corpus_ids: list[str],
        corpus_images: list[Any] | None,
        corpus_texts: list[str] | None,
        dataset_name: str | None = None,
    ) -> None: ...

    def search(self, query_ids: list[str], queries: list[str], top_k: int = 10) -> Run: ...

    def storage_bytes(self) -> int: ...


def dir_size(path: str | Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def top_k_of(scores: dict[str, float], k: int) -> dict[str, float]:
    """Keep the k best entries; ties broken by id so results are reproducible."""
    best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return dict(best)
