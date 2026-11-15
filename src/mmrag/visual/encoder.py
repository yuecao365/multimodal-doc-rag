"""Late-interaction encoders: page images and queries -> token-vector matrices.

``ColpaliEncoder`` wraps a ColPali-family model from colpali-engine (ColSmol,
ColQwen2, ColPali...). It needs torch and colpali-engine, which live in the GPU
environment, so those imports happen only when the encoder is created.

``PrecomputedQueryEncoder`` serves query vectors that were encoded earlier and
saved to disk, so retrieval can run in an environment without the model.

``WorkerQueryEncoder`` encodes queries that only exist at run time (rewrites,
agent searches). The model cannot be imported in the project environment
(colpali-engine and RAG-Anything need incompatible transformers versions), so it
runs as a child process in the visual environment: ``mmrag.visual.query_worker``.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
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
        model_name: str = "vidore/colqwen2.5-v0.2",
        device: str | None = None,
        batch_size: int = 4,
        dtype: str | None = None,
    ) -> None:
        import torch
        from colpali_engine import models

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 on GPU; on CPU pass "bfloat16" to halve memory (fp16 is not usable there).
        self.dtype = dtype or ("float16" if self.device.startswith("cuda") else "float32")
        model_cls, processor_cls = _family(model_name)
        self._torch = torch
        self.model = getattr(models, model_cls).from_pretrained(
            model_name, torch_dtype=getattr(torch, self.dtype)
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
        # Drop padding rows so each item keeps only its real token vectors. Only those are
        # checked: padding rows are discarded anyway.
        vectors = [out[i][mask[i]] for i in range(out.shape[0])]
        if not all(torch.isfinite(v).all() for v in vectors):
            raise FloatingPointError(
                f"{self.model_name} produced NaN or inf in {self.dtype}; these vectors are unusable"
            )
        return [v.float().cpu().numpy() for v in vectors]

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
    """Looks query vectors up in a store written by ``mmrag.visual.encode_cli``.

    Queries that are not in the store go to ``fallback`` (a ``WorkerQueryEncoder``)
    when one is given; without it an unknown query is an error.
    """

    def __init__(self, store: MultiVectorStore, fallback: Any | None = None) -> None:
        self.store = store
        self.fallback = fallback
        # Without a fallback only the stored queries can be searched, not new wording.
        self.encodes_new_queries = fallback is not None
        self._new: dict[str, np.ndarray] = {}

    def encode_images(self, images: list[Any]) -> list[np.ndarray]:
        raise RuntimeError(
            "Page images cannot be encoded here: only precomputed vectors are available. "
            "Encode pages with mmrag.visual.encode_cli in the visual environment."
        )

    def encode_queries(self, texts: list[str]) -> list[np.ndarray]:
        keys = [text_key(t) for t in texts]
        missing = list(
            dict.fromkeys(
                t for t, k in zip(texts, keys) if k not in self.store and k not in self._new
            )
        )
        if missing and self.fallback is None:
            raise KeyError(
                f"{len(missing)} queries have no precomputed vectors (e.g. {missing[0]!r}). "
                "Re-run mmrag.visual.encode_cli, or turn on --local-query-encoder."
            )
        if missing:
            for text, vectors in zip(missing, self.fallback.encode_queries(missing)):
                self._new[text_key(text)] = vectors
        return [self.store.get(k) if k in self.store else self._new[k] for k in keys]


class WorkerExited(RuntimeError):
    """The query worker process ended before it replied."""


class WorkerQueryEncoder:
    """Query encoder backed by a ``mmrag.visual.query_worker`` child process.

    The process starts on the first query and keeps the model loaded. ``seconds``
    holds the latency of every call, model loading excluded.
    """

    def __init__(
        self, command: list[str], env: dict[str, str] | None = None, attempts: int = 4
    ) -> None:
        self.command = command
        self.env = env
        self.attempts = attempts
        self.restarts = 0
        self.seconds: list[float] = []
        self.load_seconds = 0.0
        self._process: subprocess.Popen | None = None
        self._tmp = tempfile.TemporaryDirectory(prefix="mmrag_queries_")

    def _reply(self) -> dict:
        """Next protocol line from the worker; libraries may print other lines."""
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            if line.startswith('{"mmrag"'):
                return json.loads(line)
        raise WorkerExited(
            f"Query worker exited (code {self._process.wait()}), see its output above; "
            f"command: {self.command}"
        )

    def _start(self) -> None:
        started = time.perf_counter()
        self._process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            encoding="utf-8", env=self.env,
        )  # fmt: skip
        self._reply()  # "ready", after the model has loaded
        self.load_seconds = time.perf_counter() - started

    def encode_images(self, images: list[Any]) -> list[np.ndarray]:
        raise RuntimeError("The query worker encodes queries only")

    def encode_queries(self, texts: list[str]) -> list[np.ndarray]:
        out = Path(self._tmp.name) / f"{len(self.seconds)}.npz"
        started = time.perf_counter()
        # The model barely fits in memory on a small machine and the process can be
        # killed while loading or encoding; a fresh process usually succeeds.
        for attempt in range(self.attempts):
            try:
                if self._process is None:
                    self._start()
                    started = time.perf_counter()
                assert self._process is not None and self._process.stdin is not None
                request = json.dumps({"queries": texts, "out": str(out)})
                self._process.stdin.write(request + "\n")
                self._process.stdin.flush()
                self._reply()
                break
            except (WorkerExited, OSError):
                self._process = None
                self.restarts += 1
                if attempt == self.attempts - 1:
                    raise
        self.seconds.append(time.perf_counter() - started)
        store = MultiVectorStore.load(out)
        out.unlink()
        return [store.get(text_key(t)) for t in texts]

    def close(self) -> None:
        if self._process is not None:
            self._process.stdin.close()
            self._process.wait(timeout=60)
            self._process = None
        self._tmp.cleanup()
