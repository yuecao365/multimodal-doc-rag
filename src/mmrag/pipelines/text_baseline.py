"""Text-only baseline: RAG-Anything indexing + LightRAG naive (vector-only) retrieval.

Each page's Markdown is inserted as its own document whose ``file_path`` is the
page's corpus_id, so every retrieved chunk maps back to exactly one page.

The class has the same ``index`` / ``search`` shape as
``vidore_benchmark.pipeline_evaluation.BasePipeline``.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
from typing import Any

from mmrag.models import NoKGLLM
from mmrag.retrieval.pages import chunks_to_page_scores

logger = logging.getLogger(__name__)


class TextBaselinePipeline:
    def __init__(
        self,
        working_dir: str,
        embedder: Any,
        embedding_dim: int,
        embedding_max_tokens: int = 8192,
        top_k_pages: int = 10,
        chunk_top_k: int = 60,
        chunk_token_size: int = 1200,
        chunk_overlap_token_size: int = 100,
        llm: Any | None = None,
    ) -> None:
        self.working_dir = working_dir
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self.embedding_max_tokens = embedding_max_tokens
        self.top_k_pages = top_k_pages
        self.chunk_top_k = chunk_top_k
        self.chunk_token_size = chunk_token_size
        self.chunk_overlap_token_size = chunk_overlap_token_size
        self.llm = llm or NoKGLLM()
        self.skipped_empty: list[str] = []
        self._rag = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- setup ---------------------------------------------------------------

    async def _ensure_rag(self):
        if self._rag is not None:
            return self._rag
        from lightrag.utils import EmbeddingFunc
        from raganything import RAGAnything, RAGAnythingConfig

        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            enable_image_processing=False,
            enable_table_processing=False,
            enable_equation_processing=False,
            display_content_stats=False,
        )
        rag = RAGAnything(
            config=config,
            llm_model_func=self.llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=self.embedding_max_tokens,
                func=self.embedder,
            ),
            lightrag_kwargs={
                "chunk_token_size": self.chunk_token_size,
                "chunk_overlap_token_size": self.chunk_overlap_token_size,
                "entity_extract_max_gleaning": 0,
                "enable_llm_cache": False,
                "enable_llm_cache_for_entity_extract": False,
            },
        )
        # We only insert pre-parsed text, so no document parser (MinerU/Docling) is needed.
        # Without this, RAG-Anything refuses to initialise when its parser is not installed.
        rag._parser_installation_checked = True
        result = await rag._ensure_lightrag_initialized()
        if not result or not result.get("success"):
            raise RuntimeError(f"RAG-Anything init failed: {(result or {}).get('error')}")
        self._rag = rag
        return rag

    # -- indexing ------------------------------------------------------------

    async def aindex(
        self,
        corpus_ids: list[str],
        corpus_images: list[Any] | None,
        corpus_texts: list[str],
        dataset_name: str | None = None,
    ) -> None:
        rag = await self._ensure_rag()
        total = len(corpus_ids)
        for i, (cid, text) in enumerate(zip(corpus_ids, corpus_texts), start=1):
            if not text or not text.strip():
                self.skipped_empty.append(cid)
                continue
            doc_id = f"doc-{cid}"
            existing = await rag.lightrag.doc_status.get_by_id(doc_id)
            if existing and _status_name(existing) == "processed":
                continue
            await rag.insert_content_list(
                content_list=[{"type": "text", "text": text, "page_idx": 0}],
                file_path=cid,
                doc_id=doc_id,
                display_stats=False,
            )
            if i % 50 == 0 or i == total:
                logger.info("indexed %d/%d pages", i, total)

    # -- search --------------------------------------------------------------

    async def asearch(
        self, query_ids: list[str], queries: list[str]
    ) -> dict[str, dict[str, float]]:
        from lightrag import QueryParam

        rag = await self._ensure_rag()
        param = QueryParam(
            mode="naive",
            chunk_top_k=self.chunk_top_k,
            enable_rerank=False,
            # Do not let the answer-context token budget drop retrieved chunks.
            max_total_tokens=self.chunk_top_k * (self.chunk_token_size + 200) + 10_000,
        )
        run: dict[str, dict[str, float]] = {}
        for qid, query in zip(query_ids, queries):
            result = await rag.lightrag.aquery_data(query, param=param)
            chunks = (result.get("data") or {}).get("chunks") or []
            ranked = [(str(c.get("file_path") or ""), None) for c in chunks]
            run[qid] = chunks_to_page_scores(ranked, self.top_k_pages)
        return run

    # -- sync wrappers (BasePipeline shape) ------------------------------------

    def _run(self, coro):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def index(self, corpus_ids, corpus_images, corpus_texts, dataset_name: str | None = None):
        return self._run(self.aindex(corpus_ids, corpus_images, corpus_texts, dataset_name))

    def search(self, query_ids, queries):
        return self._run(self.asearch(query_ids, queries))

    async def aclose(self) -> None:
        if self._rag is not None:
            await self._rag.finalize_storages()
            # RAG-Anything registers an atexit hook that finalizes again and logs to a
            # stream that may already be closed.
            atexit.unregister(self._rag.close)
            self._rag = None


def _status_name(doc_status: Any) -> str:
    status = (
        doc_status.get("status")
        if isinstance(doc_status, dict)
        else getattr(doc_status, "status", "")
    )
    return str(getattr(status, "value", status)).lower()
