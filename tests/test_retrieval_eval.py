import math

import pytest

from mmrag.eval.retrieval import evaluate_run


def test_perfect_ranking_scores_one():
    qrels = {"q1": {"a": 1, "b": 1}}
    run = {"q1": {"a": 2.0, "b": 1.0, "c": 0.5}}
    mean, per_query = evaluate_run(qrels, run)
    assert mean["ndcg_cut_10"] == pytest.approx(1.0)
    assert mean["recall_10"] == pytest.approx(1.0)
    assert per_query["q1"]["recall_10"] == pytest.approx(1.0)


def test_known_values():
    # One relevant page at rank 2, one relevant page never retrieved.
    qrels = {"q1": {"a": 1, "b": 1}}
    run = {"q1": {"x": 3.0, "a": 2.0, "y": 1.0}}
    mean, _ = evaluate_run(qrels, run)
    dcg = 1 / math.log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert mean["ndcg_cut_10"] == pytest.approx(dcg / idcg)
    assert mean["recall_10"] == pytest.approx(0.5)


def test_recall_is_cut_at_ten():
    qrels = {"q1": {"hit": 1}}
    run = {"q1": {f"d{i}": 100.0 - i for i in range(10)} | {"hit": 1.0}}  # rank 11
    mean, _ = evaluate_run(qrels, run)
    assert mean["recall_10"] == 0.0


def test_query_without_results_counts_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": {"a": 1.0}, "q2": {}}
    mean, per_query = evaluate_run(qrels, run)
    assert per_query["q2"] == {"ndcg_cut_10": 0.0, "recall_10": 0.0}
    assert mean["recall_10"] == pytest.approx(0.5)
