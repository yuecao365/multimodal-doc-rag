"""Context assembly: choose which retrieved pages go to the VLM, and in what form.

A page enters the context either as its image or as its parsed text. Pages are
taken in rank order until the page limit or the token budget is reached.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

Form = Literal["image", "text"]

_CJK = re.compile(r"[㐀-鿿぀-ヿ가-힯]")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: one per CJK character, one per four other characters."""
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk + 3) // 4


@dataclass
class PageContext:
    page_id: str
    form: Form
    text: str = ""
    image: Any = None
    tokens: int = 0
    truncated: bool = False
    label: str = ""  # human-readable source, e.g. "report.pdf p.12"


def assemble_context(
    ranked_page_ids: list[str],
    texts: dict[str, str],
    images: dict[str, Any] | None = None,
    forms: dict[str, Form] | None = None,
    default_form: Form = "text",
    max_pages: int = 5,
    token_budget: int = 12_000,
    image_tokens: int = 1_500,
    min_text_tokens: int = 150,
    labels: dict[str, str] | None = None,
) -> list[PageContext]:
    """Pick pages for the prompt.

    ``forms`` gives the preferred form per page (for example "image" for pages
    the page analyzer marked visual). A page falls back to the other form when
    the preferred one is unavailable. A text page that does not fit is truncated
    to the remaining budget; an image that does not fit is sent as text if that
    fits, otherwise skipped.
    """
    images = images or {}
    forms = forms or {}
    labels = labels or {}
    remaining = token_budget
    out: list[PageContext] = []

    for pid in ranked_page_ids:
        if len(out) >= max_pages or remaining <= 0:
            break
        text = (texts.get(pid) or "").strip()
        image = images.get(pid)
        form = forms.get(pid, default_form)
        if form == "image" and image is None:
            form = "text"
        if form == "text" and not text:
            form = "image" if image is not None else form
        if form == "text" and not text:
            continue  # nothing to show for this page

        if form == "image" and image_tokens > remaining:
            if not text:
                continue
            form = "text"

        if form == "image":
            out.append(
                PageContext(
                    pid,
                    "image",
                    text=text,
                    image=image,
                    tokens=image_tokens,
                    label=labels.get(pid, ""),
                )
            )
            remaining -= image_tokens
            continue

        tokens = estimate_tokens(text)
        truncated = False
        if tokens > remaining:
            if remaining < min_text_tokens:
                continue
            text = text[: max(1, int(len(text) * remaining / tokens))]
            tokens = estimate_tokens(text)
            truncated = True
        out.append(
            PageContext(
                pid,
                "text",
                text=text,
                tokens=tokens,
                truncated=truncated,
                label=labels.get(pid, ""),
            )
        )
        remaining -= tokens
    return out
