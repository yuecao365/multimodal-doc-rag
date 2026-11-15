"""Model clients: embeddings, chat/VLM, and the no-knowledge-graph LLM stub.

All remote models go through OpenAI-compatible endpoints configured in .env:

  EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL / EMBEDDING_DIM
  LLM_BASE_URL / LLM_API_KEY / LLM_MODEL          answer generation (a VLM)
  JUDGE_MODEL, RERANK_MODEL                        optional, default to LLM_MODEL
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

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


def _require_env(names: list[str]) -> None:
    missing = [k for k in names if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)} (see DECISIONS.md)")


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
        _require_env(
            ["EMBEDDING_BASE_URL", "EMBEDDING_API_KEY", "EMBEDDING_MODEL", "EMBEDDING_DIM"]
        )
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
        return {
            "model": self.model,
            "calls": self.calls,
            "texts": self.texts,
            "total_tokens": self.total_tokens,
            "cost_usd": round(cost_usd(self.model, self.total_tokens, 0, 0), 5),
        }


class CachedEmbedder:
    """Disk cache in front of an embedder: each text is sent to the API once.

    Re-indexing the same pages (another route combination, a truncated dimension)
    then costs nothing and gives identical vectors. ``usage`` reports the API calls
    actually made plus the cache hits.
    """

    def __init__(self, inner: Any, path: str | Path) -> None:
        self.inner = inner
        self.model = inner.model
        self.dim = inner.dim
        self.max_tokens = inner.max_tokens
        self.hits = 0
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}|{text}".encode()).hexdigest()

    async def __call__(self, texts: list[str], **kwargs) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        with closing(sqlite3.connect(self.path)) as db:
            for key in set(keys):
                row = db.execute("SELECT vector FROM vectors WHERE key = ?", (key,)).fetchone()
                if row:
                    found[key] = np.frombuffer(row[0], dtype=np.float32)
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in found))
        if missing:
            fresh = np.asarray(await self.inner(missing), dtype=np.float32)
            for text, vector in zip(missing, fresh):
                found[self._key(text)] = vector
            with closing(sqlite3.connect(self.path)) as db, db:
                db.executemany(
                    "INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                    [(self._key(t), v.tobytes()) for t, v in zip(missing, fresh)],
                )
        self.hits += len(texts) - len(missing)
        return np.stack([found[k] for k in keys])

    def usage(self) -> dict:
        return {**self.inner.usage(), "cache_hits": self.hits}


class TruncatedEmbedder:
    """Matryoshka truncation: keep the first ``dim`` dimensions and re-normalize.

    ``text-embedding-3-*`` is trained so that a prefix of the vector is itself a
    usable embedding, so a smaller index needs no new model.
    """

    def __init__(self, inner: Any, dim: int) -> None:
        if not 0 < dim <= inner.dim:
            raise ValueError(f"embedding_dim must be between 1 and {inner.dim}")
        self.inner = inner
        self.dim = dim
        self.max_tokens = inner.max_tokens

    async def __call__(self, texts: list[str], **kwargs) -> np.ndarray:
        vectors = (await self.inner(texts))[:, : self.dim]
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def usage(self) -> dict:
        return self.inner.usage()


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(image: Any, max_side: int = 1568, quality: int = 85) -> dict:
    """A PIL image as an OpenAI-style image content part (downscaled JPEG data URL)."""
    image = image.convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


# USD per 1M tokens: (input, cached input, output). Reasoning tokens are billed as output.
# Source: developers.openai.com/api/docs/pricing, read 2026-11-14.
PRICES = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "text-embedding-3-small": (0.02, 0.02, 0.0),
}


def cost_usd(model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    """Cost of a token count at the listed price; a model without a listed price is an error."""
    if model not in PRICES:
        raise KeyError(f"No price for '{model}' in mmrag.models.PRICES")
    fresh, cached, output = PRICES[model]
    return (
        (prompt_tokens - cached_tokens) * fresh
        + cached_tokens * cached
        + completion_tokens * output
    ) / 1e6


class ChatClient:
    """Chat / vision model behind an OpenAI-compatible endpoint, with usage accounting.

    The models used are reasoning models: they take ``max_completion_tokens`` (which
    covers reasoning and the visible reply) and ``reasoning_effort``, and no temperature.

    ``complete(system, parts)`` is the call most of the code relies on, so tests
    substitute any object with that method; the agent uses ``chat`` for tool calls.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        reasoning_effort: str = "low",
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.calls = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.truncated = 0
        self._lock = threading.Lock()  # answer runs call the client from several threads
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=8, timeout=180)

    @classmethod
    def from_env(cls, model_var: str = "LLM_MODEL", **kwargs) -> ChatClient:
        _require_env(["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"])
        return cls(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=os.getenv(model_var) or os.environ["LLM_MODEL"],
            **kwargs,
        )

    def chat(self, messages: list[dict], **kwargs: Any) -> tuple[Any, int]:
        """One chat completion; returns (reply message, tokens this call used)."""
        # The chat completions endpoint rejects function tools together with reasoning
        # for these models, so tool-calling turns (the agent) run without reasoning.
        effort = "none" if "tools" in kwargs else self.reasoning_effort
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_tokens,
            reasoning_effort=effort,
            seed=0,  # the only handle on reproducibility these models offer
            **kwargs,
        )
        usage = resp.usage
        with self._lock:
            self.calls += 1
            self.truncated += resp.choices[0].finish_reason == "length"
            if usage is not None:
                self.prompt_tokens += usage.prompt_tokens or 0
                self.completion_tokens += usage.completion_tokens or 0
                prompt_details = getattr(usage, "prompt_tokens_details", None)
                self.cached_tokens += getattr(prompt_details, "cached_tokens", 0) or 0
                details = getattr(usage, "completion_tokens_details", None)
                self.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0
        return resp.choices[0].message, (usage.total_tokens or 0) if usage is not None else 0

    def complete(self, system: str, parts: list[dict] | str, json_mode: bool = False) -> str:
        content = parts if isinstance(parts, list) else [text_part(parts)]
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        return self.chat(messages, **kwargs)[0].content or ""

    def usage(self) -> dict:
        """Token counts and cost. ``completion_tokens`` includes ``reasoning_tokens``."""
        return {
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "truncated_replies": self.truncated,
            "cost_usd": round(
                cost_usd(
                    self.model, self.prompt_tokens, self.cached_tokens, self.completion_tokens
                ),
                5,
            ),
        }
