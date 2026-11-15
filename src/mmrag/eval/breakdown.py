"""Break a per-query score down by query type and by number of relevant pages."""

from __future__ import annotations

from mmrag.data.vidore import VidoreSubset

RELEVANT_BINS = (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10+", 10, 10**9))


def relevant_bin(n_relevant: int) -> str:
    return next(name for name, low, high in RELEVANT_BINS if low <= n_relevant <= high)


def n_relevant(data: VidoreSubset, qid: str) -> int:
    return sum(1 for score in data.qrels[qid].values() if score > 0)


def breakdown(scores: dict[str, float], data: VidoreSubset) -> dict:
    """Mean of ``scores`` ({query_id: value}) per group, with group sizes.

    A query with several types counts in each of them.
    """
    groups: dict[str, dict[str, list[float]]] = {"query_type": {}, "relevant_pages": {}}
    for qid, value in scores.items():
        for kind in data.query_types.get(qid) or ["(none)"]:
            groups["query_type"].setdefault(kind, []).append(value)
        groups["relevant_pages"].setdefault(relevant_bin(n_relevant(data, qid)), []).append(value)
    return {
        by: {
            name: {"n": len(values), "mean": round(sum(values) / len(values), 4)}
            for name, values in sorted(named.items())
        }
        for by, named in groups.items()
    }
