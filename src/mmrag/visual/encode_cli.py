"""Encode a ViDoRe subset's pages and queries into multi-vectors.

Runs wherever the visual model runs: the Kaggle notebook
(``notebooks/encode_colqwen.ipynb``) or the local visual environment. It needs
torch, colpali-engine, pyarrow, huggingface-hub, pillow and numpy.

    python -m mmrag.visual.encode_cli --dataset vidore/vidore_v3_hr \
        --model vidore/colqwen2.5-v0.2 --out data/vectors/vidore_v3_hr

Output:

    <out>/pages.npz     page vectors, float16, not pooled (ids = corpus ids)
    <out>/queries.npz   query vectors, float16 (ids = hash of the query text)
    <out>/meta.json     model, counts and timing

Pooling and coarser quantization are applied later, when a retriever builds its
index from pages.npz, so one encoding run serves every compression setting.

Pages are saved in parts while encoding. A re-run skips every page and query
that is already on disk, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from mmrag.data.vidore import VidoreSubset, load_vidore_subset
from mmrag.visual.store import MultiVectorStore, text_key


def _load_same_model(path: Path, model: str) -> MultiVectorStore:
    store = MultiVectorStore.load(path)
    found = store.meta.get("model")
    if found != model:
        raise ValueError(
            f"{path} holds vectors of '{found}', not '{model}'. Use another --out or delete it."
        )
    return store


def encode_pages(
    encoder: Any,
    data: VidoreSubset,
    out_dir: Path,
    model: str,
    limit: int | None = None,
    batch_size: int = 4,
    part_pages: int = 64,
) -> dict:
    """Write ``pages.npz``; returns counts and timing of the pages encoded in this call."""
    final, parts_dir = out_dir / "pages.npz", out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    saved = sorted(parts_dir.glob("*.npz")) + ([final] if final.exists() else [])
    stores = [_load_same_model(p, model) for p in saved]
    done = {pid for s in stores for pid in s.ids}
    wanted = data.corpus_ids[:limit]
    todo = [c for c in wanted if c not in done]
    print(f"pages: {len(wanted)} wanted, {len(wanted) - len(todo)} already encoded", flush=True)

    ids: list[str] = []
    vectors: list = []
    n_parts, encoded, seconds = len(saved), 0, 0.0

    def save_part() -> None:
        nonlocal n_parts
        part = MultiVectorStore.from_embeddings(ids, vectors, "float16", meta={"model": model})
        part.save(parts_dir / f"{n_parts:05d}.npz")
        stores.append(part)
        n_parts += 1
        ids.clear()
        vectors.clear()

    def encode(batch: list[tuple[str, Any]]) -> None:
        nonlocal encoded, seconds
        started = time.perf_counter()
        embeddings = encoder.encode_images([image for _, image in batch])
        took = time.perf_counter() - started
        ids.extend(cid for cid, _ in batch)
        vectors.extend(embeddings)
        encoded += len(batch)
        seconds += took
        left = (len(todo) - encoded) * seconds / encoded / 60
        print(
            f"pages {encoded}/{len(todo)}  {took / len(batch):.2f} s/page"
            f"  (mean {seconds / encoded:.2f})  about {left:.0f} min left",
            flush=True,
        )
        if len(ids) >= part_pages:
            save_part()

    batch: list[tuple[str, Any]] = []
    for item in data.iter_page_images(todo, batch_size=batch_size):
        batch.append(item)
        if len(batch) == batch_size:
            encode(batch)
            batch = []
    if batch:
        encode(batch)
    if ids:
        save_part()

    # One file in corpus order; pages from an earlier, larger run are kept.
    merged = MultiVectorStore.concat(stores)
    order = [c for c in data.corpus_ids if c in merged]
    merged.subset(order).save(final)
    shutil.rmtree(parts_dir)
    return {
        **MultiVectorStore.load(final).stats(),
        "encoded_now": encoded,
        "encode_seconds": round(seconds, 1),
        "seconds_per_page": round(seconds / encoded, 3) if encoded else None,
    }


def encode_queries(encoder: Any, texts: list[str], out_dir: Path, model: str) -> dict:
    """Write ``queries.npz`` with a vector matrix for every text; known texts are kept."""
    final = out_dir / "queries.npz"
    stores = [_load_same_model(final, model)] if final.exists() else []
    todo = [t for t in sorted(set(texts)) if not (stores and text_key(t) in stores[0])]
    print(f"queries: {len(set(texts))} wanted, {len(todo)} to encode", flush=True)
    seconds = 0.0
    if todo:
        started = time.perf_counter()
        vectors = encoder.encode_queries(todo)
        seconds = time.perf_counter() - started
        stores.append(
            MultiVectorStore.from_embeddings(
                [text_key(t) for t in todo], vectors, "float16", meta={"model": model}
            )
        )
        MultiVectorStore.concat(stores).save(final)
    return {
        **MultiVectorStore.load(final).stats(),
        "encoded_now": len(todo),
        "encode_seconds": round(seconds, 2),
        "seconds_per_query": round(seconds / len(todo), 4) if todo else None,
    }


def encode_dataset(
    encoder: Any,
    dataset: str,
    out_dir: str | Path,
    model: str,
    language: str | None = "english",
    limit: int | None = None,
    batch_size: int = 4,
) -> dict:
    """Encode all pages (or the first ``limit``) and all queries of one subset."""
    out_dir = Path(out_dir)
    data = load_vidore_subset(dataset, n_queries=None, language=language)
    meta = {
        "dataset": dataset,
        "model": model,
        "language": language,
        "device": getattr(encoder, "device", None),
        "dtype": getattr(encoder, "dtype", None),
        "batch_size": batch_size,
        "pages": encode_pages(encoder, data, out_dir, model, limit, batch_size),
        "queries": encode_queries(encoder, data.queries, out_dir, model),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="Hub id or local directory")
    p.add_argument("--out", required=True, help="e.g. data/vectors/vidore_v3_hr")
    p.add_argument("--model", default="vidore/colqwen2.5-v0.2")
    p.add_argument("--language", default="english")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, help="float16 (GPU default), bfloat16 or float32")
    p.add_argument("--limit", type=int, default=None, help="encode only the first N pages")
    args = p.parse_args(argv)

    from mmrag.visual.encoder import ColpaliEncoder

    encoder = ColpaliEncoder(args.model, args.device, args.batch_size, args.dtype)
    encode_dataset(
        encoder, args.dataset, args.out, args.model, args.language, args.limit, args.batch_size
    )


if __name__ == "__main__":
    main()
