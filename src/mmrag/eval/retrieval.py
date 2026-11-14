"""Retrieval metrics via pytrec_eval, the same scorer vidore-benchmark uses."""

from __future__ import annotations

import pytrec_eval

Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]

# pytrec_eval measure spec -> key it reports per query
MEASURES = {"ndcg_cut.10": "ndcg_cut_10", "recall.10": "recall_10"}


def evaluate_run(qrels: Qrels, run: Run) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Score a run against qrels.

    Returns (mean over all judged queries, per-query scores). A query with no
    retrieved pages scores 0 rather than being dropped from the mean.
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(MEASURES))
    non_empty = {qid: docs for qid, docs in run.items() if docs and qid in qrels}
    scored = evaluator.evaluate(non_empty) if non_empty else {}

    per_query = {
        qid: {key: float(scored.get(qid, {}).get(key, 0.0)) for key in MEASURES.values()}
        for qid in qrels
    }
    n = len(per_query)
    mean = {
        key: (sum(q[key] for q in per_query.values()) / n if n else 0.0)
        for key in MEASURES.values()
    }
    return mean, per_query
