"""Query rewriting with a hypothetical answer passage (HyDE).

Users' wording can differ from the documents' wording. The model writes a short
passage that could answer the question; the search text is the question followed
by that passage. The text and keyword routes search with it; the visual route keeps
the question (see ``pipelines.hybrid.REWRITTEN_ROUTES``). Passages are kept on disk,
so a question is rewritten once and later runs reuse it.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HYDE_SYSTEM = """You help a search engine find pages of reports.
Write a short passage (two to four sentences) that could appear in such a report and would
answer the question. State plausible specifics in the style of the report; do not mention
the question, and do not say that you are unsure. Reply with the passage only."""


class HydeRewriter:
    """rewriter(queries) -> search texts; ``client`` needs ``complete(system, text)``."""

    def __init__(self, client: Any, cache_path: str | Path | None = None, workers: int = 8):
        self.client = client
        self.cache_path = Path(cache_path) if cache_path else None
        self.workers = workers
        self.passages: dict[str, str] = {}
        if self.cache_path is not None and self.cache_path.exists():
            self.passages = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.written = 0

    def __call__(self, queries: list[str]) -> list[str]:
        missing = [q for q in dict.fromkeys(queries) if q not in self.passages]
        if missing:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                replies = pool.map(lambda q: self.client.complete(HYDE_SYSTEM, q), missing)
                self.passages.update(zip(missing, (r.strip() for r in replies)))
            self.written += len(missing)
            if self.cache_path is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(
                    json.dumps(self.passages, ensure_ascii=False), encoding="utf-8"
                )
        return [f"{q}\n{self.passages[q]}".strip() for q in queries]

    def info(self) -> dict:
        info: dict[str, Any] = {"rewriter": "hyde", "passages_written": self.written}
        if hasattr(self.client, "usage"):
            info["usage"] = self.client.usage()
        return info
