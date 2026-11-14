import pytest

from mmrag.retrieval.bm25 import BM25Retriever, tokenize
from mmrag.retrieval.fusion import ranked_ids, rrf_fuse

DOCS = {
    "a": "The TX-900 projector must be booked through the travel desk.",
    "b": "A projector and a screen are available in every meeting room.",
    "c": "Employees may claim travel expenses within thirty days of returning.",
    "d": "Invoice INV-2024/0031 was paid on 3 March. Total: 1,250.00 EUR.",
    "e": "",
}


@pytest.fixture
def bm25():
    retriever = BM25Retriever()
    retriever.index(list(DOCS), None, list(DOCS.values()))
    return retriever


def test_tokenize_keeps_compound_codes_and_their_parts():
    assert tokenize("Model TX-900, v1.2") == ["model", "tx-900", "tx", "900", "v1.2", "v1", "2"]


def test_tokenize_stems_words_and_drops_stopwords_but_not_numbers():
    assert tokenize("The employees are claiming expenses for 2024") == [
        "employe",
        "claim",
        "expens",
        "2024",
    ]


def test_tokenize_uses_the_stemmer_of_the_corpus_language():
    assert tokenize("les politiques", "french") == ["le", "polit"]
    with pytest.raises(ValueError):
        tokenize("text", "klingon")


def test_exact_identifier_beats_the_generic_page(bm25):
    run = bm25.search(["q"], ["who books the TX-900 projector"])
    assert ranked_ids(run["q"])[:2] == ["a", "b"]
    assert run["q"]["a"] > 2 * run["q"]["b"]


def test_invoice_number_and_inflected_queries(bm25):
    run = bm25.search(
        ["inv", "stem", "none"], ["INV-2024/0031", "employee expense claim", "zebra crossing"]
    )
    assert ranked_ids(run["inv"]) == ["d"]
    assert ranked_ids(run["stem"]) == ["c"]  # matches "Employees ... claim ... expenses"
    assert run["none"] == {}  # no page shares a term: nothing is returned


def test_top_k_and_persistence(bm25, tmp_path):
    assert len(bm25.search(["q"], ["projector"], top_k=1)["q"]) == 1
    path = bm25.save(tmp_path / "bm25" / "index.json")
    loaded = BM25Retriever.load(path)
    query = ["who books the TX-900 projector"]
    assert loaded.search(["q"], query) == bm25.search(["q"], query)
    assert loaded.storage_bytes() == path.stat().st_size
    assert (
        BM25Retriever.load(BM25Retriever(language="french").save(tmp_path / "fr.json")).language
        == "french"
    )


def test_rrf_rewards_agreement_between_routes():
    fused = rrf_fuse(
        {
            "text": {"x": 0.9, "y": 0.8, "z": 0.1},
            "bm25": {"y": 12.0, "w": 3.0},
            "visual": {"y": 30.0, "x": 20.0},
        },
        k=60,
    )
    assert ranked_ids(fused) == ["y", "x", "w", "z"]
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)
    assert fused["z"] == pytest.approx(1 / 63)


def test_rrf_ignores_score_scales_and_supports_weights_and_top_k():
    routes = {"text": {"x": 0.002, "y": 0.001}, "bm25": {"y": 9000.0, "x": 1.0}}
    fused = rrf_fuse(routes)
    assert fused["x"] == pytest.approx(fused["y"])  # rank 1 + rank 2 on both sides

    weighted = rrf_fuse(routes, weights={"bm25": 2.0})
    assert ranked_ids(weighted) == ["y", "x"]
    assert list(rrf_fuse(routes, weights={"bm25": 0.0}, top_k=1)) == ["x"]
    assert rrf_fuse({"text": {}, "bm25": {}}) == {}
