"""Collapse ranked text chunks into a ranked list of pages."""

from __future__ import annotations

from collections.abc import Iterable


def chunks_to_page_scores(
    ranked_chunks: Iterable[tuple[str, float | None]], top_k: int
) -> dict[str, float]:
    """Turn chunks (best first) into {page_id: score} for the top_k pages.

    ``ranked_chunks`` yields (page_id, score) in retrieval order. A page takes
    the score of its best chunk. When the retriever exposes no score (None), a
    rank-based score 1/(rank+1) is used so the order is preserved.
    """
    pages: dict[str, float] = {}
    for rank, (page_id, score) in enumerate(ranked_chunks):
        if not page_id:
            continue
        value = float(score) if score is not None else 1.0 / (rank + 1)
        if page_id not in pages:
            if len(pages) >= top_k:
                continue
            pages[page_id] = value
        elif value > pages[page_id]:
            pages[page_id] = value
    return pages
