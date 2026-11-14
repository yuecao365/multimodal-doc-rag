"""Page analyzer: decide per page whether it is a text page or a visual page.

Text pages go to parsing + text retrieval; visual pages go to visual retrieval.
The decision uses what AGENTS.md names: share of figures and tables, text
density, and OCR confidence.

  from the parsed text   characters of running text, share of them inside
                         tables, figure captions
  from the page image    share of coloured pixels (charts, photos, infographics
                         are coloured; body text is not) and ink coverage
  from the parser        OCR confidence, when the parser reports one

A page is visual if any rule fires; the rules that fired are kept as reasons so
routing decisions can be inspected in the failure analysis.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}")
_HTML_TABLE = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)
_IMAGE_REF = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\b|<figure\b", re.IGNORECASE)
_CAPTION = re.compile(
    r"^\s*(?:\*\*|#+\s*)?(?:figure|fig\.|chart|graph|exhibit|diagram|图|圖)\s*[\dA-Z]",
    re.IGNORECASE | re.MULTILINE,
)
_MARKUP = re.compile(r"[#*_`>|\-:=~\[\]()]")


@dataclass
class AnalyzerConfig:
    min_text_chars: int = 300  # fewer characters of text than this -> visual
    max_table_ratio: float = 0.4  # more of the text inside tables than this -> visual
    max_color_ratio: float = 0.06  # more coloured pixels than this -> visual
    min_ocr_confidence: float = 0.8  # lower parser confidence than this -> visual
    # A captioned figure makes the page visual unless the page is mostly running text.
    figure_text_chars: int = 1500
    # Ink that the text does not account for: a page well covered with ink but with
    # few characters per unit of ink is a scan, diagram or picture.
    min_ink_ratio: float = 0.12
    min_chars_per_ink: float = 4000.0
    thumbnail: int = 256


@dataclass
class PageAnalysis:
    page_id: str
    route: str  # "text" or "visual"
    reasons: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _visible_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", _MARKUP.sub("", text)))


def text_features(text: str) -> dict[str, float]:
    text = text or ""
    table_chars = sum(_visible_chars(m) for m in _HTML_TABLE.findall(text))
    rest = _HTML_TABLE.sub("", text)
    body_chars = 0
    for line in rest.splitlines():
        if _TABLE_RULE.match(line) and set(line.strip()) <= set("|-: "):
            continue
        chars = _visible_chars(line)
        if _TABLE_LINE.match(line):
            table_chars += chars
        else:
            body_chars += chars
    total = body_chars + table_chars
    return {
        "text_chars": float(total),
        "table_ratio": table_chars / total if total else 0.0,
        "n_figures": float(len(_IMAGE_REF.findall(text)) + len(_CAPTION.findall(text))),
    }


def image_features(image: Any, thumbnail: int = 256) -> dict[str, float]:
    """Colour and ink coverage of a page image, measured on a small thumbnail."""
    small = image.convert("RGB")
    small.thumbnail((thumbnail, thumbnail))
    rgb = np.asarray(small, dtype=np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)  # 0 for grey pixels, large for saturated ones
    luminance = rgb.mean(axis=2)
    return {
        "color_ratio": float((spread > 40).mean()),
        "ink_ratio": float((luminance < 200).mean()),
    }


class PageAnalyzer:
    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self.config = config or AnalyzerConfig()

    def analyze_page(
        self,
        page_id: str,
        text: str,
        image: Any | None = None,
        ocr_confidence: float | None = None,
    ) -> PageAnalysis:
        cfg = self.config
        features = text_features(text)
        if image is not None:
            features.update(image_features(image, cfg.thumbnail))
        if ocr_confidence is not None:
            features["ocr_confidence"] = float(ocr_confidence)

        chars = features["text_chars"]
        reasons = []
        if chars < cfg.min_text_chars:
            reasons.append("little_text")
        if features["table_ratio"] > cfg.max_table_ratio:
            reasons.append("table_heavy")
        if features["n_figures"] >= 1 and chars < cfg.figure_text_chars:
            reasons.append("figure")
        if features.get("color_ratio", 0.0) > cfg.max_color_ratio:
            reasons.append("coloured_graphics")
        ink = features.get("ink_ratio")
        if ink is not None and ink > cfg.min_ink_ratio and chars / ink < cfg.min_chars_per_ink:
            reasons.append("ink_without_text")
        if ocr_confidence is not None and ocr_confidence < cfg.min_ocr_confidence:
            reasons.append("low_ocr_confidence")
        return PageAnalysis(str(page_id), "visual" if reasons else "text", reasons, features)

    def analyze(
        self,
        page_ids: list[str],
        texts: list[str],
        images: Iterable[tuple[str, Any]] | None = None,
        ocr_confidences: dict[str, float] | None = None,
    ) -> dict[str, PageAnalysis]:
        """Analyze a corpus. ``images`` yields (page_id, image) in any order; pages
        without an image are judged on their text alone."""
        text_by_id = dict(zip(map(str, page_ids), texts))
        ocr = ocr_confidences or {}
        out: dict[str, PageAnalysis] = {}
        for pid, image in images or ():
            pid = str(pid)
            if pid in text_by_id:
                out[pid] = self.analyze_page(pid, text_by_id[pid], image, ocr.get(pid))
        for pid, text in text_by_id.items():
            if pid not in out:
                out[pid] = self.analyze_page(pid, text, None, ocr.get(pid))
        return {pid: out[pid] for pid in text_by_id}


def route_counts(analyses: dict[str, PageAnalysis]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for a in analyses.values():
        for r in a.reasons:
            reasons[r] = reasons.get(r, 0) + 1
    n_visual = sum(1 for a in analyses.values() if a.route == "visual")
    return {
        "n_pages": len(analyses),
        "n_text": len(analyses) - n_visual,
        "n_visual": n_visual,
        "visual_reasons": dict(sorted(reasons.items())),
    }
