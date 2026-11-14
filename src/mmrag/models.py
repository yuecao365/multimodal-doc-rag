"""Model functions handed to LightRAG / RAG-Anything."""

from __future__ import annotations

import os

import numpy as np

# What LightRAG's entity-extraction parser reads as "nothing extracted".
EMPTY_EXTRACTION = "<|COMPLETE|>"


class NoKGLLM:
    """Stand-in LLM that extracts no entities, so indexing builds no knowledge graph.

    LightRAG always runs entity extraction on insert; there is no switch to skip
    it. Returning an empty extraction keeps the insert path intact at zero LLM
    cost. ``calls`` records how many LLM calls a real model would have received.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
        self.calls += 1
        return EMPTY_EXTRACTION


class OpenAICompatEmbedder:
    """Embeddings from any OpenAI-compatible endpoint, with usage accounting."""

    def __init__(self, base_url: str, api_key: str, model: str, dim: int, max_tokens: int = 8192):
        from openai import AsyncOpenAI

        self.model = model
        self.dim = dim
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.calls = 0
        self.texts = 0
        self.total_tokens = 0
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=5)

    @classmethod
    def from_env(cls) -> OpenAICompatEmbedder:
        missing = [
            k
            for k in ("EMBEDDING_BASE_URL", "EMBEDDING_API_KEY", "EMBEDDING_MODEL", "EMBEDDING_DIM")
            if not os.getenv(k)
        ]
        if missing:
            raise RuntimeError(f"Missing in .env: {', '.join(missing)} (see .env.example)")
        return cls(
            base_url=os.environ["EMBEDDING_BASE_URL"],
            api_key=os.environ["EMBEDDING_API_KEY"],
            model=os.environ["EMBEDDING_MODEL"],
            dim=int(os.environ["EMBEDDING_DIM"]),
            max_tokens=int(os.getenv("EMBEDDING_MAX_TOKENS", "8192")),
        )

    async def __call__(self, texts: list[str], **kwargs) -> np.ndarray:
        resp = await self._client.embeddings.create(
            model=self.model, input=texts, encoding_format="float"
        )
        self.calls += 1
        self.texts += len(texts)
        if resp.usage is not None:
            self.total_tokens += resp.usage.total_tokens or 0
        data = sorted(resp.data, key=lambda d: d.index)
        return np.array([d.embedding for d in data], dtype=np.float32)

    def usage(self) -> dict:
        return {"calls": self.calls, "texts": self.texts, "total_tokens": self.total_tokens}
