"""Multi-vector page index: storage, compression and MaxSim scoring.

A page is a matrix of token vectors (one row per image patch). A query is also
a matrix; the late-interaction score is

    MaxSim(Q, P) = sum over query tokens q of  max over page tokens p of  q . p

Two ways to shrink the index, usable together:
  pooling       merge similar token vectors of a page (fewer vectors per page)
  quantization  store each number in fewer bits (float16, int8 or 1 bit)

This module needs only numpy (scipy for pooling), so it runs in any environment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

QUANTIZATIONS = ("float32", "float16", "int8", "binary")


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def pool_tokens(embedding: np.ndarray, factor: int) -> np.ndarray:
    """Merge a page's token vectors into about ``len / factor`` cluster means.

    Hierarchical (Ward) clustering groups similar vectors, as in ColPali's token
    pooling; each cluster is replaced by its re-normalised mean.
    """
    n = len(embedding)
    if factor <= 1 or n <= 1:
        return embedding.astype(np.float32, copy=False)
    target = max(1, n // factor)
    if target >= n:
        return embedding.astype(np.float32, copy=False)
    from scipy.cluster.hierarchy import fcluster, linkage

    x = _normalize(embedding.astype(np.float32))
    labels = fcluster(linkage(x, method="ward"), t=target, criterion="maxclust")
    pooled = np.zeros((labels.max(), x.shape[1]), dtype=np.float32)
    np.add.at(pooled, labels - 1, x)
    return _normalize(pooled)


def quantize(x: np.ndarray, quantization: str) -> np.ndarray:
    """Encode float vectors (assumed L2-normalised) in the given storage format."""
    if quantization == "float32":
        return x.astype(np.float32)
    if quantization == "float16":
        return x.astype(np.float16)
    if quantization == "int8":
        return np.clip(np.rint(x * 127.0), -127, 127).astype(np.int8)
    if quantization == "binary":
        return np.packbits(x > 0, axis=-1)
    raise ValueError(f"Unknown quantization '{quantization}'. Choose from {QUANTIZATIONS}")


def dequantize(x: np.ndarray, quantization: str, dim: int) -> np.ndarray:
    """Back to float32 vectors usable for dot products."""
    if quantization in ("float32", "float16"):
        return x.astype(np.float32)
    if quantization == "int8":
        return x.astype(np.float32) / 127.0
    if quantization == "binary":
        bits = np.unpackbits(x, axis=-1, count=dim).astype(np.float32)
        return (bits * 2.0 - 1.0) / np.sqrt(dim)
    raise ValueError(f"Unknown quantization '{quantization}'")


def text_key(text: str) -> str:
    """Stable id for a query string, used to look up precomputed query vectors."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


@dataclass
class MultiVectorStore:
    """Token vectors of many pages in one array, with per-page offsets."""

    ids: list[str]
    offsets: np.ndarray  # shape (n_pages + 1,), row range of each page in ``vectors``
    vectors: np.ndarray  # shape (n_vectors, dim) or (n_vectors, ceil(dim / 8)) for binary
    dim: int
    quantization: str = "float16"
    pool_factor: int = 1
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._row = {pid: i for i, pid in enumerate(self.ids)}

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_embeddings(
        cls,
        ids: list[str],
        embeddings: list[np.ndarray],
        quantization: str = "float16",
        pool_factor: int = 1,
        meta: dict | None = None,
    ) -> MultiVectorStore:
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings differ in length")
        if quantization not in QUANTIZATIONS:
            raise ValueError(f"Unknown quantization '{quantization}'")
        dim = int(embeddings[0].shape[1]) if embeddings else 0
        pooled = [pool_tokens(np.asarray(e, dtype=np.float32), pool_factor) for e in embeddings]
        offsets = np.zeros(len(pooled) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(e) for e in pooled])
        if pooled:
            vectors = quantize(np.concatenate(pooled, axis=0), quantization)
        else:
            vectors = np.zeros((0, 0), dtype=np.float32)
        return cls(list(ids), offsets, vectors, dim, quantization, pool_factor, dict(meta or {}))

    def recompress(self, quantization: str, pool_factor: int = 1) -> MultiVectorStore:
        """A new store with different compression, built from this one's vectors."""
        if quantization == self.quantization and pool_factor == 1:
            return self
        embeddings = [self.get(pid) for pid in self.ids]
        store = MultiVectorStore.from_embeddings(
            self.ids, embeddings, quantization=quantization, pool_factor=pool_factor, meta=self.meta
        )
        store.pool_factor = pool_factor * self.pool_factor
        return store

    def subset(self, ids: list[str]) -> MultiVectorStore:
        missing = [pid for pid in ids if pid not in self._row]
        if missing:
            raise KeyError(f"{len(missing)} ids not in the store, e.g. {missing[:5]}")
        parts = [self.vectors[self.offsets[self._row[p]] : self.offsets[self._row[p] + 1]]
                 for p in ids]  # fmt: skip
        offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(p) for p in parts])
        vectors = np.concatenate(parts, axis=0) if parts else self.vectors[:0]
        return MultiVectorStore(
            list(ids), offsets, vectors, self.dim, self.quantization, self.pool_factor, self.meta
        )

    # -- access ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, page_id: str) -> bool:
        return page_id in self._row

    def get(self, page_id: str) -> np.ndarray:
        """Float32 token vectors of one page."""
        i = self._row[page_id]
        raw = self.vectors[self.offsets[i] : self.offsets[i + 1]]
        return dequantize(raw, self.quantization, self.dim)

    @property
    def n_vectors(self) -> int:
        return int(self.offsets[-1]) if len(self.offsets) else 0

    @property
    def nbytes(self) -> int:
        return int(self.vectors.nbytes + self.offsets.nbytes)

    def stats(self) -> dict:
        n = max(1, len(self.ids))
        return {
            "n_pages": len(self.ids),
            "n_vectors": self.n_vectors,
            "vectors_per_page": round(self.n_vectors / n, 1),
            "dim": self.dim,
            "quantization": self.quantization,
            "pool_factor": self.pool_factor,
            "bytes": self.nbytes,
            "bytes_per_page": round(self.nbytes / n),
        }

    # -- scoring ---------------------------------------------------------------

    def score(self, query: np.ndarray, chunk_vectors: int = 200_000) -> np.ndarray:
        """MaxSim of one query (n_tokens, dim) against every page; shape (n_pages,)."""
        scores = np.zeros(len(self.ids), dtype=np.float32)
        if not len(self.ids) or not len(query):
            return scores
        query = np.asarray(query, dtype=np.float32)
        counts = np.diff(self.offsets)
        start = 0
        while start < len(self.ids):
            # Take whole pages until the chunk holds about ``chunk_vectors`` rows.
            end = start + 1
            while (
                end < len(self.ids) and self.offsets[end + 1] - self.offsets[start] <= chunk_vectors
            ):
                end += 1
            lo, hi = self.offsets[start], self.offsets[end]
            nonempty = np.flatnonzero(counts[start:end] > 0)
            if hi > lo and len(nonempty):
                block = dequantize(self.vectors[lo:hi], self.quantization, self.dim)
                sims = query @ block.T  # (n_query_tokens, n_block_vectors)
                starts = (self.offsets[start:end] - lo)[nonempty]
                best = np.maximum.reduceat(sims, starts, axis=1)  # per query token, per page
                scores[start + nonempty] = best.sum(axis=0)
            start = end
        return scores

    def score_pages(self, query: np.ndarray, page_ids: list[str]) -> dict[str, float]:
        """MaxSim against selected pages only (used for reranking)."""
        query = np.asarray(query, dtype=np.float32)
        out = {}
        for pid in page_ids:
            if pid in self._row:
                page = self.get(pid)
                out[pid] = float((query @ page.T).max(axis=1).sum()) if len(page) else 0.0
        return out

    # -- persistence -----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "dim": self.dim,
            "quantization": self.quantization,
            "pool_factor": self.pool_factor,
            "meta": self.meta,
        }
        with path.open("wb") as f:
            np.savez(
                f,
                ids=np.array(self.ids, dtype=str),
                offsets=self.offsets,
                vectors=self.vectors,
                header=np.array(json.dumps(header)),
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> MultiVectorStore:
        with np.load(Path(path), allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            return cls(
                ids=[str(i) for i in data["ids"]],
                offsets=data["offsets"],
                vectors=data["vectors"],
                dim=int(header["dim"]),
                quantization=header["quantization"],
                pool_factor=int(header.get("pool_factor", 1)),
                meta=header.get("meta", {}),
            )
