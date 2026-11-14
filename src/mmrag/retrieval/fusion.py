"""Reciprocal rank fusion of several routes' page rankings."""

from __future__ import annotations

from mmrag.retrieval.base import top_k_of


def ranked_ids(scores: dict[str, float]) -> list[str]:
    """Page ids best-first; ties broken by id so the order is reproducible."""
    return [pid for pid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def rrf_fuse(
    route_scores: dict[str, dict[str, float]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> dict[str, float]:
    """Fuse one query's results from several routes.

    score(page) = sum over routes of  weight / (k + rank of page in that route)

    Only ranks matter, so routes with incomparable scores (cosine, BM25, MaxSim)
    combine without calibration. A page missing from a route simply gets nothing
    from it, which is what lets routes cover different sets of pages.
    """
    weights = weights or {}
    fused: dict[str, float] = {}
    for route, scores in route_scores.items():
        weight = weights.get(route, 1.0)
        if weight == 0:
            continue
        for rank, pid in enumerate(ranked_ids(scores), start=1):
            fused[pid] = fused.get(pid, 0.0) + weight / (k + rank)
    return top_k_of(fused, top_k) if top_k is not None else fused
