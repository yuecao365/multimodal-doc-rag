import numpy as np
import pytest

from mmrag.visual.store import MultiVectorStore, dequantize, pool_tokens, quantize, text_key


def unit(rng, n, dim=32):
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def brute_maxsim(query, page):
    return float((query @ page.T).max(axis=1).sum()) if len(page) else 0.0


@pytest.fixture
def pages():
    rng = np.random.default_rng(0)
    return {f"p{i}": unit(rng, n) for i, n in enumerate([7, 1, 12, 5, 9])}


def test_score_matches_brute_force_maxsim(pages):
    rng = np.random.default_rng(1)
    query = unit(rng, 4)
    store = MultiVectorStore.from_embeddings(list(pages), list(pages.values()), "float32")
    expected = [brute_maxsim(query, p) for p in pages.values()]
    np.testing.assert_allclose(store.score(query), expected, rtol=1e-5)
    # Same result when pages are scored in several small chunks.
    np.testing.assert_allclose(store.score(query, chunk_vectors=8), expected, rtol=1e-5)


def test_pages_without_vectors_score_zero_and_do_not_shift_others(pages):
    rng = np.random.default_rng(2)
    query = unit(rng, 3)
    ids = ["a", "empty", "b"]
    embs = [pages["p0"], np.zeros((0, 32), dtype=np.float32), pages["p2"]]
    store = MultiVectorStore.from_embeddings(ids, embs, "float32")
    scores = store.score(query)
    assert scores[1] == 0.0
    assert scores[0] == pytest.approx(brute_maxsim(query, pages["p0"]), rel=1e-5)
    assert scores[2] == pytest.approx(brute_maxsim(query, pages["p2"]), rel=1e-5)


@pytest.mark.parametrize(
    ("quantization", "bytes_per_number", "tolerance"),
    [("float32", 4, 1e-6), ("float16", 2, 1e-3), ("int8", 1, 2e-2)],
)
def test_quantization_trades_bytes_for_precision(pages, quantization, bytes_per_number, tolerance):
    page = pages["p2"]
    stored = quantize(page, quantization)
    assert stored.nbytes == page.size * bytes_per_number
    np.testing.assert_allclose(dequantize(stored, quantization, 32), page, atol=tolerance)


def test_binary_keeps_one_bit_per_dimension_and_the_ranking_of_clear_cases():
    rng = np.random.default_rng(3)
    target = unit(rng, 6, dim=128)
    other = unit(rng, 6, dim=128)
    store = MultiVectorStore.from_embeddings(["target", "other"], [target, other], "binary")
    assert store.vectors.shape == (12, 16)  # 128 bits -> 16 bytes per vector
    scores = store.score(target[:3])  # query made of the target's own vectors
    assert scores[0] > scores[1]


def test_pooling_reduces_vectors_and_merges_near_duplicates():
    rng = np.random.default_rng(4)
    centers = unit(rng, 4)
    # 4 tight clusters of 5 vectors each.
    page = np.concatenate([c + 0.01 * rng.standard_normal((5, 32)) for c in centers])
    page = (page / np.linalg.norm(page, axis=1, keepdims=True)).astype(np.float32)
    pooled = pool_tokens(page, factor=5)
    assert pooled.shape == (4, 32)
    np.testing.assert_allclose(np.linalg.norm(pooled, axis=1), 1.0, atol=1e-5)
    # Every original cluster center is still represented.
    assert (centers @ pooled.T).max(axis=1).min() > 0.99
    assert pool_tokens(page, factor=1) is page


def test_recompress_from_a_full_precision_store(pages):
    full = MultiVectorStore.from_embeddings(list(pages), list(pages.values()), "float16")
    small = full.recompress("int8", pool_factor=3)
    assert small.quantization == "int8" and small.pool_factor == 3
    assert small.n_vectors < full.n_vectors
    assert small.nbytes < full.nbytes / 3
    assert small.ids == full.ids
    assert full.recompress("float16", 1) is full


def test_subset_keeps_order_and_rejects_unknown_ids(pages):
    store = MultiVectorStore.from_embeddings(list(pages), list(pages.values()), "float32")
    sub = store.subset(["p3", "p0"])
    assert sub.ids == ["p3", "p0"]
    np.testing.assert_allclose(sub.get("p0"), pages["p0"])
    with pytest.raises(KeyError):
        store.subset(["p0", "nope"])


def test_save_and_load_roundtrip(pages, tmp_path):
    rng = np.random.default_rng(5)
    query = unit(rng, 4)
    store = MultiVectorStore.from_embeddings(
        list(pages), list(pages.values()), "int8", pool_factor=2, meta={"model": "m"}
    )
    path = store.save(tmp_path / "idx" / "pages.npz")
    loaded = MultiVectorStore.load(path)
    assert loaded.ids == store.ids and loaded.meta == {"model": "m"}
    assert loaded.stats() == store.stats()
    np.testing.assert_allclose(loaded.score(query), store.score(query))
    assert loaded.score_pages(query, ["p1", "missing"]).keys() == {"p1"}


def test_text_key_is_stable():
    assert text_key("what is x?") == text_key("what is x?") != text_key("what is y?")
