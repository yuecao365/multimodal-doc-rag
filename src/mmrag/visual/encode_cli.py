"""Encode exported pages and queries into multi-vectors (the GPU step).

Runs in the GPU environment; it needs only torch, colpali-engine, pillow and
numpy. From the project root (PowerShell):

    $env:HF_HOME = "D:\\rag_cache\\hf"; $env:PYTHONPATH = "src"
    D:\\rag_cache\\venv_gpu\\Scripts\\python -m mmrag.visual.encode_cli `
        --pages data/pages/vidore_v3_hr --out data/visual/vidore_v3_hr_colsmol

Input is the directory written by ``mmrag.experiments.export_pages``. Output:

    <out>/pages.npz     page vectors, float16, not pooled (ids = corpus ids)
    <out>/queries.npz   query vectors, float16 (ids = hash of the query text)
    <out>/meta.json     model, counts and timing

Pooling and coarser quantization are applied later, when a retriever builds its
index from pages.npz, so one encoding run serves every compression setting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mmrag.visual.store import MultiVectorStore, text_key


def _image_files(image_dir: Path) -> list[Path]:
    files = [p for p in image_dir.iterdir() if p.is_file()]
    return sorted(
        files, key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.stem)
    )


def encode_pages(encoder, image_dir: Path, batch_size: int, limit: int | None = None):
    from PIL import Image

    files = _image_files(image_dir)[:limit]
    ids: list[str] = []
    embeddings: list[np.ndarray] = []
    started = time.perf_counter()
    for i in range(0, len(files), batch_size):
        chunk = files[i : i + batch_size]
        images = [Image.open(f).convert("RGB") for f in chunk]
        for f, emb in zip(chunk, encoder.encode_images(images)):
            ids.append(f.stem)
            embeddings.append(emb.astype(np.float16))
        done = i + len(chunk)
        if done % (batch_size * 10) == 0 or done == len(files):
            rate = done / max(1e-9, time.perf_counter() - started)
            print(f"pages {done}/{len(files)}  {rate:.2f} pages/s", flush=True)
    return ids, embeddings


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pages", required=True, help="directory from export_pages")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="vidore/colSmol-256M")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="encode only the first N pages")
    p.add_argument("--skip-pages", action="store_true", help="only (re-)encode the queries")
    args = p.parse_args(argv)

    from mmrag.visual.encoder import ColpaliEncoder

    pages_dir, out_dir = Path(args.pages), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = ColpaliEncoder(args.model, device=args.device, batch_size=args.batch_size)
    meta: dict = {"model": args.model, "device": encoder.device, "batch_size": args.batch_size}
    store_meta = {"model": args.model}

    if not args.skip_pages:
        t0 = time.perf_counter()
        ids, embeddings = encode_pages(encoder, pages_dir / "images", args.batch_size, args.limit)
        seconds = time.perf_counter() - t0
        store = MultiVectorStore.from_embeddings(ids, embeddings, "float16", meta=store_meta)
        store.save(out_dir / "pages.npz")
        meta["pages"] = {**store.stats(), "encode_seconds": round(seconds, 1),
                         "seconds_per_page": round(seconds / max(1, len(ids)), 3)}  # fmt: skip

    queries_file = pages_dir / "queries.jsonl"
    if queries_file.exists():
        with queries_file.open(encoding="utf-8") as f:
            texts = sorted({json.loads(line)["query"] for line in f if line.strip()})
        t0 = time.perf_counter()
        vectors = encoder.encode_queries(texts)
        seconds = time.perf_counter() - t0
        store = MultiVectorStore.from_embeddings(
            [text_key(t) for t in texts], vectors, "float16", meta=store_meta
        )
        store.save(out_dir / "queries.npz")
        meta["queries"] = {**store.stats(), "encode_seconds": round(seconds, 2),
                           "seconds_per_query": round(seconds / max(1, len(texts)), 4)}  # fmt: skip

    meta_path = out_dir / "meta.json"
    if meta_path.exists():  # keep figures from an earlier pages run when only queries re-ran
        meta = {**json.loads(meta_path.read_text(encoding="utf-8")), **meta}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
