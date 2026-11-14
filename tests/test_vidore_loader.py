from mmrag.data.vidore import build_qrels, load_vidore_subset, sample_query_ids


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


def test_loads_subset_from_local_parquet(dataset_dir):
    data = load_vidore_subset(str(dataset_dir), n_queries=2, language="english", seed=0)
    assert len(data.query_ids) == 2 and set(data.query_ids) <= {"10", "11", "12"}
    assert data.corpus_ids == ["0", "1", "2", "3", "4", "5"]
    assert data.corpus_texts[3] == ""
    assert set(data.qrels) == set(data.query_ids)
    assert all(data.answers[q] for q in data.query_ids)
    assert data.doc_ids[0] == "handbook" and data.page_numbers[5] == 5
    qid = data.query_ids[0]
    assert data.bboxes[qid][next(iter(data.qrels[qid]))][0]["x2"] == 56


def test_all_queries_and_page_images(dataset_dir):
    data = load_vidore_subset(str(dataset_dir), n_queries=None, language=None)
    assert len(data.query_ids) == 4
    images = data.load_page_images(["1", "4"])
    assert set(images) == {"1", "4"}
    assert images["1"].size == (64, 80)
    assert [cid for cid, _ in data.iter_page_images()] == data.corpus_ids
