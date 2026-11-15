"""Encoding a subset to files: format, resume, and the run-time query encoder."""

import io
import json
import sys

import numpy as np
import pytest

from helpers import PAGES, QUERIES, FakeLateEncoder
from mmrag.data.vidore import load_vidore_subset
from mmrag.pipelines.factory import load_precomputed
from mmrag.visual.check_vectors import check_vectors, token_cosines
from mmrag.visual.encode_cli import encode_dataset, encode_pages
from mmrag.visual.encoder import PrecomputedQueryEncoder, WorkerQueryEncoder
from mmrag.visual.query_worker import serve
from mmrag.visual.store import MultiVectorStore, text_key

ENGLISH = [q[0] for q in QUERIES.values() if q[1] == "english"]


class FailingEncoder(FakeLateEncoder):
    """Stops after ``calls`` batches, like a session that ran out of time."""

    def __init__(self, calls: int) -> None:
        super().__init__()
        self.calls = calls

    def encode_images(self, images):
        if self.image_calls == self.calls:
            raise RuntimeError("session ended")
        return super().encode_images(images)


class OverflowingEncoder(FakeLateEncoder):
    """fp16 overflow stand-in: ``batch_only`` pages fail only together with other pages
    (padding), ``always`` pages fail even alone."""

    def __init__(self, batch_only=(), always=()) -> None:
        super().__init__()
        self.batch_only, self.always = set(batch_only), set(always)

    def encode_images(self, images):
        ids = {str(np.asarray(im)[0, 0, 0]) for im in images}
        if ids & self.always or (len(images) > 1 and ids & self.batch_only):
            raise FloatingPointError("non-finite")
        return super().encode_images(images)


def test_overflowing_batches_are_retried_page_by_page(dataset_dir, tmp_path):
    data = load_vidore_subset(str(dataset_dir), n_queries=None)
    fallback = FakeLateEncoder()
    stats = encode_pages(
        OverflowingEncoder(batch_only={"1"}, always={"2"}), data, tmp_path, "fake",
        batch_size=4, fallback=lambda: fallback,
    )  # fmt: skip
    assert stats["fallback_pages"] == ["2"] and fallback.image_calls == 1
    clean = tmp_path / "clean"
    encode_pages(FakeLateEncoder(), data, clean, "fake", batch_size=4)
    got, want = (
        MultiVectorStore.load(tmp_path / "pages.npz"),
        MultiVectorStore.load(clean / "pages.npz"),
    )
    assert got.ids == want.ids
    assert all(np.array_equal(got.get(p), want.get(p)) for p in got.ids)


def test_an_overflowing_page_without_fallback_is_named(dataset_dir, tmp_path):
    data = load_vidore_subset(str(dataset_dir), n_queries=None)
    with pytest.raises(FloatingPointError, match="page 2"):
        encode_pages(OverflowingEncoder(always={"2"}), data, tmp_path, "fake", batch_size=4)


def test_output_is_what_the_retriever_loads(dataset_dir, tmp_path):
    out = tmp_path / "vectors" / "toy"
    meta = encode_dataset(FakeLateEncoder(), str(dataset_dir), out, model="fake", batch_size=4)
    assert sorted(p.name for p in out.iterdir()) == ["meta.json", "pages.npz", "queries.npz"]
    assert meta["pages"]["n_pages"] == len(PAGES) and meta["pages"]["encoded_now"] == len(PAGES)
    assert meta["queries"]["n_pages"] == len(ENGLISH)
    assert json.loads((out / "meta.json").read_text())["model"] == "fake"

    pages, query_encoder = load_precomputed(str(out))
    assert pages.ids == list(PAGES) and pages.quantization == "float16"
    assert pages.meta == {"model": "fake"}
    assert len(query_encoder.encode_queries(ENGLISH)) == len(ENGLISH)


def test_rerun_resumes_after_an_interruption(dataset_dir, tmp_path):
    data = load_vidore_subset(str(dataset_dir), n_queries=None)
    out = tmp_path / "toy"
    with pytest.raises(RuntimeError, match="session ended"):
        encode_pages(FailingEncoder(2), data, out, "fake", batch_size=2, part_pages=2)
    assert len(list((out / "parts").glob("*.npz"))) == 2 and not (out / "pages.npz").exists()

    encoder = FakeLateEncoder()
    stats = encode_pages(encoder, data, out, "fake", batch_size=2, part_pages=2)
    assert encoder.image_calls == 1 and stats["encoded_now"] == 2  # only the missing pages
    assert MultiVectorStore.load(out / "pages.npz").ids == list(PAGES)
    assert not (out / "parts").exists()

    again = FakeLateEncoder()
    assert encode_pages(again, data, out, "fake")["encoded_now"] == 0 and again.image_calls == 0


def test_a_limited_run_is_extended_not_repeated(dataset_dir, tmp_path):
    out = tmp_path / "toy"
    encode_dataset(FakeLateEncoder(), str(dataset_dir), out, model="fake", limit=2)
    assert MultiVectorStore.load(out / "pages.npz").ids == ["0", "1"]
    meta = encode_dataset(FakeLateEncoder(), str(dataset_dir), out, model="fake")
    assert meta["pages"]["encoded_now"] == 4 and meta["queries"]["encoded_now"] == 0
    assert MultiVectorStore.load(out / "pages.npz").ids == list(PAGES)


def test_vectors_of_another_model_are_not_mixed_in(dataset_dir, tmp_path):
    encode_dataset(FakeLateEncoder(), str(dataset_dir), tmp_path, model="fake")
    with pytest.raises(ValueError, match="holds vectors of 'fake'"):
        encode_dataset(FakeLateEncoder(), str(dataset_dir), tmp_path, model="other")


def test_concat_rejects_duplicates_and_mixed_compression():
    encoder = FakeLateEncoder()
    a = MultiVectorStore.from_embeddings(["a"], encoder.encode_queries(["one two"]), "float16")
    b = MultiVectorStore.from_embeddings(["b"], encoder.encode_queries(["three"]), "float16")
    both = MultiVectorStore.concat([a, b])
    assert both.ids == ["a", "b"] and np.array_equal(both.get("b"), b.get("b"))
    with pytest.raises(ValueError, match="share page ids"):
        MultiVectorStore.concat([a, a])
    with pytest.raises(ValueError, match="differ"):
        MultiVectorStore.concat([a, b.recompress("int8")])


def test_worker_protocol_writes_vectors(tmp_path):
    out = tmp_path / "q.npz"
    requests = io.StringIO(
        json.dumps({"queries": ["paid leave", "paid leave"], "out": str(out)}) + "\n"
    )
    replies = io.StringIO()
    serve(FakeLateEncoder(), requests, replies)
    kinds = [json.loads(line)["mmrag"] for line in replies.getvalue().splitlines()]
    assert kinds == ["ready", "done"]
    assert MultiVectorStore.load(out).ids == [text_key("paid leave")]


WORKER = """
import sys
sys.path[:0] = {paths!r}
from helpers import FakeLateEncoder
from mmrag.visual.query_worker import serve
print("a library printing to stdout")
serve(FakeLateEncoder(), sys.stdin, sys.stdout)
"""


def test_unknown_queries_go_to_the_worker_process(tmp_path):
    encoder = FakeLateEncoder()
    known, new = "salary bands reviewed", "when is pay reviewed"
    store = MultiVectorStore.from_embeddings(
        [text_key(known)], encoder.encode_queries([known]), "float16"
    )
    script = tmp_path / "worker.py"
    script.write_text(WORKER.format(paths=sys.path), encoding="utf-8")
    worker = WorkerQueryEncoder([sys.executable, str(script)])
    queries = PrecomputedQueryEncoder(store, fallback=worker)
    try:
        assert len(queries.encode_queries([known])) == 1 and worker.seconds == []  # not started
        vectors = queries.encode_queries([new, known, new])
        assert np.allclose(vectors[0], encoder.encode_queries([new])[0], atol=1e-3)
        assert np.array_equal(vectors[0], vectors[2])
        queries.encode_queries([new])
        assert len(worker.seconds) == 1  # the second time it is served from memory
    finally:
        worker.close()


def test_worker_that_dies_is_an_error():
    worker = WorkerQueryEncoder([sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(RuntimeError, match=r"exited \(code 3\)"):
        worker.encode_queries(["q"])
    assert worker.restarts == 4  # it was started again before giving up


def test_check_vectors_passes_for_the_same_model_and_fails_for_another(dataset_dir, tmp_path):
    encode_dataset(FakeLateEncoder(), str(dataset_dir), tmp_path, model="fake")
    report = check_vectors(FakeLateEncoder(), str(dataset_dir), tmp_path, n_pages=3, n_queries=2)
    assert report["agree"] == {"page": True, "query": True}
    assert report["worst_mean_cosine"]["page"] > 0.999 and report["mean_query_top10_same"] == 1
    assert [i["kind"] for i in report["items"]] == ["page"] * 3 + ["query"] * 2

    class Other(FakeLateEncoder):
        def _vectors(self, text):
            return super()._vectors(text)[:, ::-1].copy()

    assert check_vectors(Other(), str(dataset_dir), tmp_path)["agree"]["query"] is False
    with pytest.raises(ValueError, match="differ in shape"):
        token_cosines(np.ones((2, 4)), np.ones((3, 4)))
