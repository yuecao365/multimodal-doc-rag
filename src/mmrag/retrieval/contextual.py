"""Contextual chunk headers (Anthropic's Contextual Retrieval).

A chunk may not say which report or section it came from. Before indexing, a
model writes one short sentence that places each chunk in its document, and the
sentence is put in front of the chunk for both the dense and the BM25 index.

The model sees the chunk's page and its neighbouring pages, not the whole
document: a document's pages are taken three at a time, and every chunk of those
three pages gets the same three pages as context. The pages come first in the
prompt and the chunk last, so the calls for one group share a prefix that the
API caches. Sentences are kept on disk and written once.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

CONTEXT_SYSTEM = """You prepare chunks of reports for a search index.
You are given consecutive pages of one report and then one chunk taken from them.
Write one short sentence (at most 30 words) that situates the chunk within the report:
name the report and the section or topic the chunk belongs to, so that the chunk can be
found by someone who has not seen the surrounding pages. Reply with the sentence only."""

GROUP_PAGES = 3


def split_tokens(text: str, size: int, overlap: int, encoding: Any) -> list[str]:
    """The text in pieces of ``size`` tokens, the way LightRAG chunks it."""
    tokens = encoding.encode(text)
    if len(tokens) <= size:
        return [text]
    step = size - overlap
    return [encoding.decode(tokens[i : i + size]) for i in range(0, len(tokens), step)]


class ContextualHeaders:
    """transform(corpus_ids, corpus_texts) -> texts with a header before each chunk."""

    def __init__(
        self,
        client: Any,
        doc_ids: dict[str, str],
        page_numbers: dict[str, int],
        cache_path: str | Path | None = None,
        chunk_tokens: int = 1200,
        overlap_tokens: int = 100,
        workers: int = 8,
    ) -> None:
        import tiktoken

        self.client = client
        self.doc_ids = doc_ids
        self.page_numbers = page_numbers
        self.cache_path = Path(cache_path) if cache_path else None
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.workers = workers
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")  # LightRAG's tokenizer
        self.headers: dict[str, str] = {}
        if self.cache_path is not None and self.cache_path.exists():
            self.headers = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.written = 0
        self.chunks = 0

    def _groups(self, corpus_ids: list[str], texts: dict[str, str]) -> list[list[str]]:
        """Pages with text, grouped three consecutive pages of a document at a time."""
        by_doc: dict[str, list[str]] = {}
        for pid in corpus_ids:
            if texts[pid].strip():
                by_doc.setdefault(self.doc_ids.get(pid, ""), []).append(pid)
        groups = []
        for pages in by_doc.values():
            pages.sort(key=lambda p: self.page_numbers.get(p, 0))
            groups += [pages[i : i + GROUP_PAGES] for i in range(0, len(pages), GROUP_PAGES)]
        return groups

    def _write_group(self, group: list[str], texts: dict[str, str]) -> dict[str, list[str]]:
        """Contextualized chunks of every page in the group."""
        shared = f"Report: {self.doc_ids.get(group[0], '')}\n\n" + "\n\n".join(
            f'<page number="{self.page_numbers.get(p, 0) + 1}">\n{texts[p]}\n</page>' for p in group
        )
        out: dict[str, list[str]] = {}
        for pid in group:
            pieces = []
            chunks = split_tokens(texts[pid], self.chunk_tokens, self.overlap_tokens, self.encoding)
            for chunk in chunks:
                key = hashlib.sha1(f"{pid}|{chunk}".encode()).hexdigest()
                if key not in self.headers:
                    prompt = f"{shared}\n\nChunk to situate:\n<chunk>\n{chunk}\n</chunk>"
                    self.headers[key] = self.client.complete(CONTEXT_SYSTEM, prompt).strip()
                pieces.append(f"{self.headers[key]}\n\n{chunk}")
            out[pid] = pieces
        return out

    def __call__(self, corpus_ids: list[str], corpus_texts: list[str]) -> list[str]:
        texts = dict(zip(corpus_ids, corpus_texts))
        groups = self._groups(corpus_ids, texts)
        known = len(self.headers)
        # Groups run in parallel; the calls of one group run in order, so the later
        # ones find the shared pages in the prompt cache.
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            done = list(pool.map(lambda g: self._write_group(g, texts), groups))
        self.written += len(self.headers) - known
        self.chunks += sum(len(chunks) for group in done for chunks in group.values())
        if self.cache_path is not None and self.written:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self.headers, ensure_ascii=False), encoding="utf-8"
            )
        pieces = {pid: chunks for group in done for pid, chunks in group.items()}
        return ["\n\n".join(pieces[pid]) if pid in pieces else texts[pid] for pid in corpus_ids]

    def info(self) -> dict:
        info: dict[str, Any] = {"chunks": self.chunks, "headers_written": self.written}
        if hasattr(self.client, "usage"):
            info["usage"] = self.client.usage()
        return info
