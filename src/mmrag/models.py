"""Model clients: embeddings, chat/VLM, and the no-knowledge-graph LLM stub.

All remote models go through OpenAI-compatible endpoints configured in .env:

  EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL / EMBEDDING_DIM
  LLM_BASE_URL / LLM_API_KEY / LLM_MODEL          answer generation (a VLM)
  JUDGE_MODEL, RERANK_MODEL                        optional, default to LLM_MODEL
"""

from __future__ import annotations

import base64
import io
import os
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
        return {"calls": self.calls, "texts": self.texts, "total_tokens": self.total_tokens}


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


class ChatClient:
    """Chat / vision model behind an OpenAI-compatible endpoint, with usage accounting.

    ``complete(system, parts)`` is the only call the rest of the code relies on,
    so tests substitute any object with that method.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=5)

    @classmethod
    def from_env(cls, model_var: str = "LLM_MODEL", **kwargs) -> ChatClient:
        _require_env(["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"])
        return cls(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=os.getenv(model_var) or os.environ["LLM_MODEL"],
            **kwargs,
        )

    def complete(self, system: str, parts: list[dict] | str, json_mode: bool = False) -> str:
        content = parts if isinstance(parts, list) else [text_part(parts)]
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        self.calls += 1
        if resp.usage is not None:
            self.prompt_tokens += resp.usage.prompt_tokens or 0
            self.completion_tokens += resp.usage.completion_tokens or 0
        return resp.choices[0].message.content or ""

    def usage(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
