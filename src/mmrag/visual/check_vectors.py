"""Check that locally encoded vectors agree with the ones encoded in the cloud.

Dataset queries are encoded on a cloud GPU in fp16; queries created at run time
are encoded locally on CPU in bfloat16. Mixing the two is only sound if both
produce the same vectors, so this re-encodes a few pages and queries locally and
compares them token by token. For queries, which are what is encoded locally at run
time, it also compares the ten best pages each version retrieves. Run it once, in
the visual environment:

    python -m mmrag.visual.check_vectors --dataset vidore/vidore_v3_hr \
        --vectors data/vectors/vidore_v3_hr

The model is read from the stored vectors. The result is printed and written to
``results/<date>_vector_check_<subset>/check.json``, pages and queries separately;
the exit code is 1 when any page or query falls below the threshold.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mmrag.data.vidore import load_vidore_subset
from mmrag.visual.store import MultiVectorStore, text_key


def token_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine of each token vector with its counterpart; both are (tokens, dim)."""
    if a.shape != b.shape:
        raise ValueError(f"Token matrices differ in shape: {a.shape} vs {b.shape}")
    a, b = a.astype(np.float64), b.astype(np.float64)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return (a * b).sum(axis=1) / np.maximum(norms, 1e-12)


def compare(stored: np.ndarray, fresh: np.ndarray) -> dict:
    if stored.shape != fresh.shape:
        return {"tokens": [len(stored), len(fresh)], "mean_cosine": 0.0, "min_cosine": 0.0}
    cosines = token_cosines(stored, fresh)
    return {
        "tokens": len(stored),
        "mean_cosine": round(float(cosines.mean()), 5),
        "min_cosine": round(float(cosines.min()), 5),
    }


def check_vectors(
    encoder: Any,
    dataset: str,
    vectors_dir: str | Path,
    n_pages: int = 3,
    n_queries: int = 5,
    threshold: float = 0.99,
) -> dict:
    vectors_dir = Path(vectors_dir)
    pages = MultiVectorStore.load(vectors_dir / "pages.npz")
    queries = MultiVectorStore.load(vectors_dir / "queries.npz")
    data = load_vidore_subset(dataset, n_queries=None)

    # Evenly spaced, so the sample is not just the first pages of one document.
    picks = np.linspace(0, len(pages) - 1, min(n_pages, len(pages))).astype(int)
    page_ids = [pages.ids[i] for i in picks]
    images = data.load_page_images(page_ids)
    texts = [q for q in data.queries if text_key(q) in queries][:n_queries]

    items = []
    for pid in page_ids:  # one at a time: CPU memory, and no padding from batch mates
        fresh = encoder.encode_images([images[pid]])[0]
        items.append({"kind": "page", "id": pid, **compare(pages.get(pid), fresh)})
        print(items[-1], flush=True)
    for text, fresh in zip(texts, encoder.encode_queries(texts)):
        stored = queries.get(text_key(text))
        k = min(10, len(pages))
        top = [set(np.argsort(-pages.score(v))[:k]) for v in (stored, fresh)]
        overlap = len(top[0] & top[1]) / k
        items.append({"kind": "query", "id": text, **compare(stored, fresh), "top10_same": overlap})
        print(items[-1], flush=True)

    worst = {
        kind: min((i["mean_cosine"] for i in items if i["kind"] == kind), default=None)
        for kind in ("page", "query")
    }
    return {
        "dataset": dataset,
        "vectors": str(vectors_dir),
        "stored": pages.meta,
        "local": {
            "device": getattr(encoder, "device", None),
            "dtype": getattr(encoder, "dtype", None),
        },
        "threshold": threshold,
        "worst_mean_cosine": worst,
        "agree": {kind: value is None or value > threshold for kind, value in worst.items()},
        "mean_query_top10_same": float(
            np.mean([i["top10_same"] for i in items if i["kind"] == "query"] or [1.0])
        ),
        "items": items,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--vectors", required=True, help="directory with pages.npz and queries.npz")
    p.add_argument("--pages", type=int, default=3)
    p.add_argument("--queries", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.99)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=None, help="default: results/<date>_vector_check_<subset>")
    args = p.parse_args(argv)

    from mmrag.visual.encoder import ColpaliEncoder

    model = MultiVectorStore.load(Path(args.vectors) / "queries.npz").meta["model"]
    encoder = ColpaliEncoder(model, args.device, batch_size=1, dtype=args.dtype)
    report = check_vectors(
        encoder, args.dataset, args.vectors, args.pages, args.queries, args.threshold
    )
    subset = args.dataset.replace("\\", "/").rstrip("/").split("/")[-1]
    out = Path(args.out or f"results/{datetime.date.today()}_vector_check_{subset}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"agree: {report['agree']}, worst mean cosine {report['worst_mean_cosine']}")
    print(f"top-10 pages shared by stored and local queries: {report['mean_query_top10_same']}")
    print(f"written to {out / 'check.json'}")
    sys.exit(0 if all(report["agree"].values()) else 1)


if __name__ == "__main__":
    main()
