"""Citation verification, done in code rather than by a model.

For every citation in an answer two things are checked:
  1. the cited page is one of the pages the model was actually shown;
  2. the quoted text really occurs on that page.

A quote is compared against the page's parsed text after normalisation (case,
whitespace, Markdown markup, typographic quotes and dashes). Small differences
are tolerated through a fuzzy match, because models tidy up punctuation and
parsed text contains line breaks and table pipes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from mmrag.generation.answer import Answer

# Citation statuses
VERIFIED = "verified"  # page was shown and the quote is on it
PAGE_NOT_IN_CONTEXT = "page_not_in_context"  # cites a page the model never saw
QUOTE_NOT_FOUND = "quote_not_found"  # page was shown, quote is not on it
NO_QUOTE = "no_quote"  # page was shown, no quote given (e.g. a chart)
NO_PAGE_TEXT = "no_page_text"  # page was shown, but has no parsed text to check against

_TRANSLATE = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
        "…": "...",
    }
)
_MARKUP = re.compile(r"[*_`#>|~]+|!\[|\]\([^)]*\)|[\[\]]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE).lower()
    text = _MARKUP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_match(quote: str, page_text: str) -> float:
    """How much of the quote is found on the page, from 0 to 1 (1 = verbatim)."""
    q, t = normalize(quote), normalize(page_text)
    if not q:
        return 0.0
    if q in t:
        return 1.0
    if not t:
        return 0.0
    # Longest run of the quote that appears on the page, then extend around it.
    matcher = SequenceMatcher(None, q, t, autojunk=False)
    block = matcher.find_longest_match(0, len(q), 0, len(t))
    if block.size == 0:
        return 0.0
    start = max(0, block.b - block.a - 10)
    window = t[start : start + len(q) + 20]
    matched = sum(b.size for b in SequenceMatcher(None, q, window, autojunk=False)
                  .get_matching_blocks())  # fmt: skip
    return matched / len(q)


@dataclass
class CitationCheck:
    page: str
    quote: str
    status: str
    match: float = 0.0


@dataclass
class CitationReport:
    checks: list[CitationCheck] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.checks)

    def count(self, status: str) -> int:
        return sum(1 for c in self.checks if c.status == status)

    @property
    def valid_pages(self) -> list[str]:
        """Cited pages that were really in the context, in citation order, without repeats."""
        seen: dict[str, None] = {}
        for c in self.checks:
            if c.status != PAGE_NOT_IN_CONTEXT:
                seen.setdefault(c.page, None)
        return list(seen)

    @property
    def all_ok(self) -> bool:
        """True when the answer cites something and no citation failed a check."""
        bad = self.count(PAGE_NOT_IN_CONTEXT) + self.count(QUOTE_NOT_FOUND)
        return self.n > 0 and bad == 0

    def to_dict(self) -> dict:
        return {
            "n_citations": self.n,
            "all_ok": self.all_ok,
            **{
                s: self.count(s)
                for s in (VERIFIED, PAGE_NOT_IN_CONTEXT, QUOTE_NOT_FOUND, NO_QUOTE, NO_PAGE_TEXT)
            },
            "checks": [asdict(c) for c in self.checks],
        }


def verify_citations(
    answer: Answer,
    context_page_ids: list[str],
    page_texts: dict[str, str],
    min_match: float = 0.85,
) -> CitationReport:
    """Check every citation of ``answer``.

    ``context_page_ids`` are the pages placed in the prompt; ``page_texts`` maps
    page ids to their full parsed text (not the possibly truncated prompt text).
    """
    shown = set(context_page_ids)
    report = CitationReport()
    for citation in answer.citations:
        quote = citation.quote.strip()
        if citation.page not in shown:
            report.checks.append(CitationCheck(citation.page, quote, PAGE_NOT_IN_CONTEXT))
            continue
        if not quote:
            report.checks.append(CitationCheck(citation.page, quote, NO_QUOTE))
            continue
        text = page_texts.get(citation.page) or ""
        if not text.strip():
            report.checks.append(CitationCheck(citation.page, quote, NO_PAGE_TEXT))
            continue
        match = quote_match(quote, text)
        status = VERIFIED if match >= min_match else QUOTE_NOT_FOUND
        report.checks.append(CitationCheck(citation.page, quote, status, round(match, 3)))
    return report


def summarize_reports(reports: list[dict]) -> dict:
    """Citation statistics over answers; ``reports`` are ``CitationReport.to_dict()`` outputs."""
    total = sum(r["n_citations"] for r in reports)

    def rate(status: str) -> float | None:
        return sum(r[status] for r in reports) / total if total else None

    n = len(reports)
    return {
        "n_answers": n,
        "n_citations": total,
        "answers_with_citations": sum(1 for r in reports if r["n_citations"]) / n if n else None,
        "answers_all_citations_ok": sum(1 for r in reports if r["all_ok"]) / n if n else None,
        "citation_verified_rate": rate(VERIFIED),
        "citation_page_not_in_context_rate": rate(PAGE_NOT_IN_CONTEXT),
        "citation_quote_not_found_rate": rate(QUOTE_NOT_FOUND),
        "citation_no_quote_rate": rate(NO_QUOTE),
        "citation_unverifiable_rate": rate(NO_PAGE_TEXT),
    }
