"""Visual route: whole-page late-interaction retrieval."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from mmrag.retrieval.base import Run, top_k_of
from mmrag.visual.encoder import LateInteractionEncoder
from mmrag.visual.store import MultiVectorStore

logger = logging.getLogger(__name__)

ImageProvider = Callable[[list[str]], Iterable[tuple[str, Any]]]


class VisualRetriever:
    """Index page images as multi-vectors and rank pages by MaxSim.

    Page vectors come from one of two places:
      * ``precomputed``: a store encoded earlier (the GPU step); ``index`` takes
        the requested pages from it and applies pooling / quantization.
      * otherwise ``encoder.encode_images`` over the images passed to ``index``
        or yielded by ``image_provider``.
    """

    name = "visual"

    def __init__(
        self,
        encoder: LateInteractionEncoder,
        store_path: str | Path | None = None,
        precomputed: MultiVectorStore | None = None,
        quantization: str = "float16",
        pool_factor: int = 1,
        image_provider: ImageProvider | None = None,
        encode_batch: int = 16,
    ) -> None:
        self.encoder = encoder
        self.store_path = Path(store_path) if store_path else None
        self.precomputed = precomputed
        self.quantization = quantization
        self.pool_factor = pool_factor
        self.image_provider = image_provider
        self.encode_batch = encode_batch
        self.store: MultiVectorStore | None = None
        self.pages_encoded = 0

    def index(
        self,
        corpus_ids: list[str],
        corpus_images: list[Any] | None = None,
        corpus_texts: list[str] | None = None,
        dataset_name: str | None = None,
    ) -> None:
        corpus_ids = [str(c) for c in corpus_ids]
        if self.precomputed is not None:
            base = self.precomputed.subset(corpus_ids)
            self.store = base.recompress(self.quantization, self.pool_factor)
        else:
            self.store = self._encode(corpus_ids, corpus_images)
        if self.store_path is not None:
            self.store.save(self.store_path)
        logger.info("visual route: %s", self.store.stats())

    def _encode(self, corpus_ids: list[str], corpus_images: list[Any] | None) -> MultiVectorStore:
        if corpus_images is not None:
            pairs: Iterable[tuple[str, Any]] = zip(corpus_ids, corpus_images)
        elif self.image_provider is not None:
            pairs = self.image_provider(corpus_ids)
        else:
            raise ValueError("Visual indexing needs page images, an image_provider or a "
                             "precomputed store")  # fmt: skip
        embeddings: dict[str, Any] = {}
        batch_ids: list[str] = []
        batch_images: list[Any] = []

        def flush() -> None:
            if batch_ids:
                for pid, emb in zip(batch_ids, self.encoder.encode_images(batch_images)):
                    embeddings[pid] = emb
                self.pages_encoded += len(batch_ids)
                batch_ids.clear()
                batch_images.clear()

        for pid, image in pairs:
            batch_ids.append(str(pid))
            batch_images.append(image)
            if len(batch_ids) >= self.encode_batch:
                flush()
        flush()
        missing = [c for c in corpus_ids if c not in embeddings]
        if missing:
            raise KeyError(f"No image for {len(missing)} pages, e.g. {missing[:5]}")
        return MultiVectorStore.from_embeddings(
            corpus_ids,
            [embeddings[c] for c in corpus_ids],
            quantization=self.quantization,
            pool_factor=self.pool_factor,
        )

    def search(self, query_ids: list[str], queries: list[str], top_k: int = 10) -> Run:
        if self.store is None:
            raise RuntimeError("index() must be called before search()")
        run: Run = {}
        scores = self.store.score_many(self.encoder.encode_queries(list(queries)))
        for qid, row in zip(query_ids, scores):
            run[str(qid)] = top_k_of(dict(zip(self.store.ids, map(float, row))), top_k)
        return run

    def storage_bytes(self) -> int:
        if self.store_path is not None and self.store_path.exists():
            return self.store_path.stat().st_size
        return self.store.nbytes if self.store is not None else 0

    def info(self) -> dict:
        info = {"pages_encoded_now": self.pages_encoded}
        if self.store is not None:
            info["visual_index"] = self.store.stats()
        return info
