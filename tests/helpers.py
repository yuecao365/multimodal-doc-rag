"""Offline stand-ins for remote models and a tiny on-disk ViDoRe-shaped dataset."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

DIM = 256

PAGES = {
    "0": "# Parental leave\nEmployees receive sixteen weeks of paid parental leave.",
    "1": "# Salary bands\nEngineering salary bands are reviewed every April.",
    "2": "# Remote work\nRemote work requires manager approval and a home office audit.",
    "3": "",  # a chart page with no extracted text
    "4": "| Year | Headcount |\n|---|---|\n| 2022 | 410 |\n| 2023 | 455 |\n| 2024 | 502 |",
    "5": "# Travel policy\nModel TX-900 projector must be booked through the travel desk.",
}
QUERIES = {
    "10": ("how many weeks of paid parental leave", "english", "Sixteen weeks.", ["0"]),
    "11": ("when are salary bands reviewed", "english", "Every April.", ["1"]),
    "12": ("who books the TX-900 projector", "english", "The travel desk.", ["5"]),
    "13": ("combien de semaines de congé parental", "french", "Seize semaines.", ["0"]),
}


def _hash(token: str, dim: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % dim


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HashingEmbedder:
    """Deterministic bag-of-words text embedder (async, like the real one)."""

    dim = DIM
    max_tokens = 8192

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, texts, **kwargs):
        self.calls += 1
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokens(text):
                out[i, _hash(token, DIM)] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out

    def usage(self) -> dict:
        return {"calls": self.calls}


class FakeChat:
    """Chat client that replays canned replies and records the prompts it saw."""

    def __init__(self, replies):
        self.replies = list(replies) if isinstance(replies, (list, tuple)) else None
        self.fn = replies if callable(replies) else None
        self.prompts: list[tuple[str, object]] = []
        self.calls = 0

    def complete(self, system, parts, json_mode=False):
        self.prompts.append((system, parts))
        self.calls += 1
        if self.fn is not None:
            return self.fn(system, parts)
        return self.replies.pop(0)

    def usage(self) -> dict:
        return {"model": "fake", "calls": self.calls, "prompt_tokens": 0, "completion_tokens": 0}


def page_image(page_id: str, size: tuple[int, int] = (64, 80)):
    """A small synthetic page image whose content depends on the page id."""
    from PIL import Image

    rng = np.random.default_rng(int(page_id) if page_id.isdigit() else 0)
    arr = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    arr[10:40, 8:56] = rng.integers(0, 255, size=(30, 48, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def make_dataset(root: Path) -> Path:
    """Write a ViDoRe-v3-shaped dataset (queries/corpus/qrels parquet) under ``root``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for part in ("queries", "corpus", "qrels"):
        (root / part).mkdir(parents=True, exist_ok=True)

    pq.write_table(
        pa.table(
            {
                "query_id": [int(q) for q in QUERIES],
                "query": [v[0] for v in QUERIES.values()],
                "language": [v[1] for v in QUERIES.values()],
                "answer": [v[2] for v in QUERIES.values()],
            }
        ),
        root / "queries" / "test-00000-of-00001.parquet",
    )

    images = []
    for pid in PAGES:
        buffer = io.BytesIO()
        page_image(pid).save(buffer, format="PNG")
        images.append({"bytes": buffer.getvalue(), "path": f"{pid}.png"})
    pq.write_table(
        pa.table(
            {
                "corpus_id": [int(p) for p in PAGES],
                "image": images,
                "doc_id": ["handbook"] * len(PAGES),
                "markdown": list(PAGES.values()),
                "page_number_in_doc": list(range(len(PAGES))),
            }
        ),
        root / "corpus" / "test-00000-of-00001.parquet",
    )

    rows = [(int(q), int(c)) for q, v in QUERIES.items() for c in v[3]]
    box = pa.list_(pa.struct([(k, pa.int64()) for k in ("annotator", "x1", "x2", "y1", "y2")]))
    pq.write_table(
        pa.table(
            {
                "query_id": [r[0] for r in rows],
                "corpus_id": [r[1] for r in rows],
                "score": [1] * len(rows),
                "bounding_boxes": pa.array(
                    [[{"annotator": 0, "x1": 8, "x2": 56, "y1": 10, "y2": 40}] for _ in rows],
                    type=box,
                ),
            }
        ),
        root / "qrels" / "test-00000-of-00001.parquet",
    )
    return root


def answer_json(answer: str, citations: list[tuple[str, str]]) -> str:
    return json.dumps(
        {"answer": answer, "citations": [{"page": p, "quote": q} for p, q in citations]}
    )
