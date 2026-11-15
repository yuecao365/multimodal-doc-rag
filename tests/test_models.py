"""Embedding wrappers: disk cache and Matryoshka truncation."""

import asyncio

import numpy as np

from helpers import HashingEmbedder
from mmrag.models import CachedEmbedder, TruncatedEmbedder


class Counting(HashingEmbedder):
    model = "hashing"

    def __init__(self) -> None:
        super().__init__()
        self.texts = 0

    async def __call__(self, texts, **kwargs):
        self.texts += len(texts)
        return await super().__call__(texts)


def test_cached_embedder_sends_each_text_once(tmp_path):
    inner = Counting()
    cached = CachedEmbedder(inner, tmp_path / "cache" / "e.sqlite")
    first = asyncio.run(cached(["salary bands", "parental leave", "salary bands"]))
    assert inner.texts == 2 and np.array_equal(first[0], first[2])

    again = CachedEmbedder(Counting(), tmp_path / "cache" / "e.sqlite")  # a later process
    second = asyncio.run(again(["parental leave", "remote work"]))
    assert again.inner.texts == 1 and again.usage()["cache_hits"] == 1
    assert np.array_equal(second[0], first[1])


def test_truncated_embedder_keeps_a_normalized_prefix():
    inner = Counting()
    small = TruncatedEmbedder(inner, 200)
    full = asyncio.run(inner(["sixteen weeks of paid parental leave"]))
    out = asyncio.run(small(["sixteen weeks of paid parental leave"]))
    assert out.shape == (1, 200) and abs(np.linalg.norm(out[0]) - 1) < 1e-6
    prefix = full[0, :200] / np.linalg.norm(full[0, :200])
    assert np.allclose(out[0], prefix)
