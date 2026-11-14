from mmrag.data.vidore import build_qrels, sample_query_ids


def test_build_qrels_filters_and_casts():
    rows = [
        {"query_id": 1, "corpus_id": 10, "score": 2},
        {"query_id": 1, "corpus_id": 11, "score": 1},
        {"query_id": 2, "corpus_id": 10, "score": 1},
    ]
    assert build_qrels(rows) == {"1": {"10": 2, "11": 1}, "2": {"10": 1}}
    assert build_qrels(rows, keep_query_ids={"2"}) == {"2": {"10": 1}}


def test_sample_filters_language_and_unjudged():
    qids = ["a", "b", "c", "d"]
    langs = ["english", "french", "english", "english"]
    picked = sample_query_ids(qids, langs, {"a", "b", "c"}, n=None, seed=0, language="english")
    assert picked == ["a", "c"]  # b is french, d has no judgments


def test_sample_is_deterministic_and_order_independent():
    qids = [str(i) for i in range(200)]
    langs = ["english"] * 200
    judged = set(qids)
    first = sample_query_ids(qids, langs, judged, n=50, seed=0, language="english")
    again = sample_query_ids(qids[::-1], langs, judged, n=50, seed=0, language="english")
    other = sample_query_ids(qids, langs, judged, n=50, seed=1, language="english")
    assert len(first) == 50 and len(set(first)) == 50
    assert first == again
    assert first != other
