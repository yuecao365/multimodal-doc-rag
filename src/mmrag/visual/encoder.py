"""Late-interaction encoders: page images and queries -> token-vector matrices.

``ColpaliEncoder`` wraps a ColPali-family model from colpali-engine (ColSmol,
ColQwen2, ColPali...). It needs torch and colpali-engine, which live in the GPU
environment, so those imports happen only when the encoder is created.

``PrecomputedQueryEncoder`` serves query vectors that were encoded earlier and
saved to disk, so retrieval can run in an environment without the model.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np

from mmrag.visual.store import MultiVectorStore, text_key


class LateInteractionEncoder(Protocol):
    def encode_images(self, images: list[Any]) -> list[np.ndarray]: ...

    def encode_queries(self, texts: list[str]) -> list[np.ndarray]: ...


# Substring of the model name -> (model class, processor class) in colpali_engine.models.
_MODEL_FAMILIES = [
    ("colsmol", "ColIdefics3", "ColIdefics3Processor"),
    ("idefics3", "ColIdefics3", "ColIdefics3Processor"),
    ("colqwen2.5", "ColQwen2_5", "ColQwen2_5_Processor"),
    ("colqwen2", "ColQwen2", "ColQwen2Processor"),
    ("colpali", "ColPali", "ColPaliProcessor"),
]


def _family(model_name: str) -> tuple[str, str]:
    lowered = model_name.lower()
    for key, model_cls, processor_cls in _MODEL_FAMILIES:
        if key in lowered:
            return model_cls, processor_cls
    raise ValueError(f"Don't know which colpali-engine class loads '{model_name}'")


def _batches(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class ColpaliEncoder:
    def __init__(
        self,
        model_name: str = "vidore/colSmol-256M",
        device: str | None = None,
        batch_size: int = 4,
        half: bool | None = None,
    ) -> None:
        import torch
        from colpali_engine import models

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_half = half if half is not None else self.device.startswith("cuda")
        model_cls, processor_cls = _family(model_name)
        self._torch = torch
        self.model = getattr(models, model_cls).from_pretrained(
            model_name, torch_dtype=torch.float16 if use_half else torch.float32
        )
        self.model = self.model.to(self.device).eval()
        self.processor = getattr(models, processor_cls).from_pretrained(model_name)
        self.image_calls = 0
        self.query_calls = 0

    def _embed(self, batch) -> list[np.ndarray]:
        torch = self._torch
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            out = self.model(**batch)  # (batch, tokens, dim), padded
        mask = batch["attention_mask"].bool()
        # Drop padding rows so each page keeps only its real token vectors.
        return [out[i][mask[i]].float().cpu().numpy() for i in range(out.shape[0])]

    def encode_images(self, images: list[Any]) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for chunk in _batches(list(images), self.batch_size):
            result.extend(self._embed(self.processor.process_images(chunk)))
            self.image_calls += 1
        return result

    def encode_queries(self, texts: list[str]) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for chunk in _batches(list(texts), max(1, self.batch_size * 4)):
            result.extend(self._embed(self.processor.process_queries(chunk)))
            self.query_calls += 1
        return result


class PrecomputedQueryEncoder:
    """Looks query vectors up in a store written by ``mmrag.visual.encode_cli``."""

    def __init__(self, store: MultiVectorStore) -> None:
        self.store = store

    def encode_images(self, images: list[Any]) -> list[np.ndarray]:
        raise RuntimeError(
            "Page images cannot be encoded here: only precomputed vectors are available. "
            "Encode pages with mmrag.visual.encode_cli in the GPU environment."
        )

    def encode_queries(self, texts: list[str]) -> list[np.ndarray]:
        keys = [text_key(t) for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self.store]
        if missing:
            raise KeyError(
                f"{len(missing)} queries have no precomputed vectors (e.g. {missing[0]!r}). "
                "Re-run mmrag.visual.encode_cli with these queries."
            )
        return [self.store.get(k) for k in keys]
