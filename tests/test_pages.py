from mmrag.retrieval.pages import chunks_to_page_scores


def test_dedupes_and_keeps_best_chunk_score():
    chunks = [("p1", 0.9), ("p2", 0.8), ("p1", 0.7), ("p3", 0.6)]
    assert chunks_to_page_scores(chunks, top_k=10) == {"p1": 0.9, "p2": 0.8, "p3": 0.6}


def test_truncates_to_top_k_pages_not_chunks():
    chunks = [("p1", 0.9), ("p1", 0.85), ("p2", 0.8), ("p3", 0.7)]
    assert list(chunks_to_page_scores(chunks, top_k=2)) == ["p1", "p2"]


def test_rank_fallback_preserves_order_without_scores():
    chunks = [("p1", None), ("p2", None), ("p1", None), ("p3", None)]
    scores = chunks_to_page_scores(chunks, top_k=10)
    assert sorted(scores, key=scores.get, reverse=True) == ["p1", "p2", "p3"]


def test_skips_chunks_without_page():
    assert chunks_to_page_scores([("", 0.9), ("p1", 0.5)], top_k=10) == {"p1": 0.5}
