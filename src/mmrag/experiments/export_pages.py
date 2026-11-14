"""Export page images and queries of a ViDoRe subset to plain files.

    uv run python -m mmrag.experiments.export_pages --dataset vidore/vidore_v3_hr

The GPU environment that runs the visual encoder has no parquet reader, so the
pages are written once as image files and the queries as JSON lines:

    data/pages/<subset>/images/<corpus_id>.<ext>
    data/pages/<subset>/queries.jsonl        {"query_id", "query"}

``data/`` is git-ignored. Existing image files are kept, so the command can be re-run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mmrag.data.vidore import load_vidore_subset
from mmrag.experiments.common import ROOT, write_jsonl


def _extension(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:2] == b"\xff\xd8":
        return "jpg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return "img"


def export_pages(
    dataset: str,
    out_dir: Path,
    language: str | None = "english",
    n_queries: int | None = None,
    seed: int = 0,
) -> dict:
    data = load_vidore_subset(dataset, n_queries, language, seed)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in image_dir.iterdir()}
    todo = [c for c in data.corpus_ids if c not in existing]
    written = 0
    for cid, raw in data.iter_page_image_bytes(todo):
        (image_dir / f"{cid}.{_extension(raw)}").write_bytes(raw)
        written += 1
    write_jsonl(
        out_dir / "queries.jsonl",
        [{"query_id": q, "query": t} for q, t in zip(data.query_ids, data.queries)],
    )
    return {"pages": len(data.corpus_ids), "images_written": written,
            "queries": len(data.query_ids)}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="vidore/vidore_v3_hr")
    p.add_argument("--out", default=None, help="default: data/pages/<subset>")
    p.add_argument("--language", default="english")
    p.add_argument("--n-queries", type=int, default=0, help="0 = all queries of the language")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    slug = args.dataset.replace("\\", "/").rstrip("/").split("/")[-1]
    out_dir = Path(args.out) if args.out else ROOT / "data" / "pages" / slug
    stats = export_pages(args.dataset, out_dir, args.language, args.n_queries or None, args.seed)
    print(f"{stats} -> {out_dir}")


if __name__ == "__main__":
    main()
