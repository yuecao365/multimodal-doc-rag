import pytest

from helpers import PAGES, FakeLateEncoder, page_image
from mmrag.visual.encoder import PrecomputedQueryEncoder, _family
from mmrag.visual.retriever import VisualRetriever
from mmrag.visual.store import MultiVectorStore, text_key

IDS = list(PAGES)
IMAGES = [page_image(p) for p in IDS]


def best(run, qid):
    return max(run[qid], key=run[qid].get)


def test_finds_text_pages_and_the_chart_page_from_images(tmp_path):
    retriever = VisualRetriever(FakeLateEncoder(), store_path=tmp_path / "v" / "pages.npz")
    retriever.index(IDS, IMAGES, None)
    run = retriever.search(
        ["leave", "chart"], ["paid parental leave weeks", "headcount growth by region chart"]
    )
    assert best(run, "leave") == "0"
    assert best(run, "chart") == "3"  # the page with no extracted text
    assert len(run["leave"]) == len(IDS)
    assert retriever.storage_bytes() == (tmp_path / "v" / "pages.npz").stat().st_size
    assert retriever.info()["visual_index"]["n_pages"] == len(IDS)


def test_top_k_and_image_provider_batches():
    encoder = FakeLateEncoder()

    def provider(ids):
        return ((pid, page_image(pid)) for pid in ids)

    retriever = VisualRetriever(encoder, image_provider=provider, encode_batch=4)
    retriever.index(IDS, None, None)
    assert encoder.image_calls == 2  # 6 pages in batches of 4
    run = retriever.search(["q"], ["salary bands reviewed"], top_k=2)
    assert len(run["q"]) == 2 and best(run, "q") == "1"


def test_precomputed_vectors_need_no_model(tmp_path):
    encoder = FakeLateEncoder()
    pages = MultiVectorStore.from_embeddings(IDS, encoder.encode_images(IMAGES), "float16")
    query = "who books the tx 900 projector"
    queries = MultiVectorStore.from_embeddings(
        [text_key(query)], encoder.encode_queries([query]), "float16"
    )
    retriever = VisualRetriever(
        PrecomputedQueryEncoder(queries), precomputed=pages, quantization="int8", pool_factor=2
    )
    # Only a subset of the encoded pages is indexed, as the hybrid pipeline does.
    retriever.index(["5", "1", "3"], None, None)
    assert retriever.store.ids == ["5", "1", "3"]
    assert retriever.store.quantization == "int8"
    assert best(retriever.search(["q"], [query]), "q") == "5"

    with pytest.raises(KeyError, match="no precomputed vectors"):
        retriever.search(["x"], ["a query that was never encoded"])


def test_compression_shrinks_the_index():
    encoder = FakeLateEncoder()
    sizes = {}
    for name, quantization, pool in [("full", "float32", 1), ("small", "binary", 2)]:
        retriever = VisualRetriever(encoder, quantization=quantization, pool_factor=pool)
        retriever.index(IDS, IMAGES, None)
        sizes[name] = retriever.store.vectors.nbytes
    assert sizes["small"] < sizes["full"] / 30


def test_indexing_without_any_image_source_is_an_error():
    with pytest.raises(ValueError, match="needs page images"):
        VisualRetriever(FakeLateEncoder()).index(IDS, None, None)


def test_model_family_lookup():
    assert _family("vidore/colSmol-256M") == ("ColIdefics3", "ColIdefics3Processor")
    assert _family("vidore/colqwen2.5-v0.2")[0] == "ColQwen2_5"
    assert _family("vidore/colqwen2-v1.0")[0] == "ColQwen2"
    with pytest.raises(ValueError):
        _family("some/unknown-model")
