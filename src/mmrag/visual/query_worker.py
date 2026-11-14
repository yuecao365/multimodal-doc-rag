"""Query encoding service for queries created at run time.

Started by ``mmrag.visual.encoder.WorkerQueryEncoder`` with the Python of the
visual environment. The model is loaded once, on CPU in bfloat16 by default, and
then serves requests over stdin/stdout, one JSON object per line:

    request   {"queries": ["..."], "out": "<file>.npz"}
    reply     {"mmrag": "done", "seconds": 1.9}

Vectors are written to ``out`` as a ``MultiVectorStore`` keyed by ``text_key``.
Any failure ends the process with a traceback on stderr; the parent reports it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import IO, Any

from mmrag.visual.store import MultiVectorStore, text_key


def serve(encoder: Any, requests: IO[str], replies: IO[str]) -> None:
    def send(**reply: Any) -> None:
        replies.write(json.dumps(reply) + "\n")
        replies.flush()

    send(mmrag="ready")
    for line in requests:
        if not line.strip():
            continue
        request = json.loads(line)
        texts = list(dict.fromkeys(request["queries"]))
        started = time.perf_counter()
        vectors = encoder.encode_queries(texts)
        seconds = time.perf_counter() - started
        keys = [text_key(t) for t in texts]
        MultiVectorStore.from_embeddings(keys, vectors, "float16").save(request["out"])
        send(mmrag="done", seconds=round(seconds, 3))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args(argv)

    from mmrag.visual.encoder import ColpaliEncoder

    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    serve(ColpaliEncoder(args.model, args.device, dtype=args.dtype), sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
