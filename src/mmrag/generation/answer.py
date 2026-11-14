"""Answer generation: a VLM reads the assembled pages and answers with citations.

RAG-Anything's ``aquery_vlm_enhanced`` builds its prompt from LightRAG's own
retrieved context and the image paths found inside it. Here the context is a
list of fused page-level results, so the prompt is built directly; the idea is
the same (the model sees the page images next to the text).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from mmrag.generation.context import PageContext
from mmrag.models import image_part, text_part

SYSTEM_PROMPT = """You answer questions about documents using only the pages provided.

Each page is introduced by a line "[PAGE <id>]" followed by the page image or its text.
Rules:
- Use only information visible on the provided pages. If they do not contain the answer, say so.
- Cite every page you relied on, by its id exactly as given.
- For each citation give a short quote copied verbatim from that page (at most 25 words).
  If the evidence is a chart or picture with no quotable text, use an empty quote.
- Answer in the language of the question.

Reply with one JSON object and nothing else:
{"answer": "<answer>", "citations": [{"page": "<id>", "quote": "<verbatim quote>"}]}"""

REGION_RULE = """
- For each citation also give "region": [x1, y1, x2, y2], the box around the evidence on the
  page image, in coordinates from 0 to 1000 (0,0 is the top-left corner). Use null if unsure."""


@dataclass
class Citation:
    page: str
    quote: str = ""
    region: list[float] | None = None


@dataclass
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    raw: str = ""
    parsed: bool = True

    @property
    def cited_pages(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.citations:
            seen.setdefault(c.page, None)
        return list(seen)


def build_prompt(query: str, contexts: list[PageContext]) -> list[dict]:
    parts: list[dict] = []
    for ctx in contexts:
        header = f"[PAGE {ctx.page_id}]" + (f" ({ctx.label})" if ctx.label else "")
        parts.append(text_part(header))
        if ctx.form == "image":
            parts.append(image_part(ctx.image))
        else:
            parts.append(text_part(ctx.text))
    parts.append(text_part(f"Question: {query}"))
    return parts


def _extract_json(raw: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_answer(raw: str) -> Answer:
    """Parse the model's JSON reply; fall back to the raw text with no citations."""
    obj = _extract_json(raw)
    if obj is None or "answer" not in obj:
        return Answer(answer=raw.strip(), raw=raw, parsed=False)
    citations = []
    for item in obj.get("citations") or []:
        if not isinstance(item, dict) or item.get("page") is None:
            continue
        page = re.sub(
            r"^\[?PAGE\s*|\]$", "", str(item["page"]).strip(), flags=re.IGNORECASE
        ).strip()
        region = item.get("region")
        if not (
            isinstance(region, list)
            and len(region) == 4
            and all(isinstance(v, (int, float)) for v in region)
        ):
            region = None
        citations.append(Citation(page=page, quote=str(item.get("quote") or ""), region=region))
    return Answer(answer=str(obj["answer"]), citations=citations, raw=raw)


def generate_answer(
    client: Any, query: str, contexts: list[PageContext], with_regions: bool = False
) -> Answer:
    """Ask the model. ``client`` needs ``complete(system, parts, json_mode=...) -> str``."""
    if not contexts:
        return Answer(answer="No relevant pages were retrieved.", raw="", parsed=True)
    system = SYSTEM_PROMPT + (REGION_RULE if with_regions else "")
    raw = client.complete(system, build_prompt(query, contexts), json_mode=True)
    return parse_answer(raw)
