"""Agentic retrieval: a model decides what to search for and which pages to answer from.

The agent is a search strategy on top of the hybrid pipeline, not a second
pipeline. It starts from the one-shot fused results, may search each route again
with its own wording (``text_search``, ``keyword_search``, ``visual_search``),
may open a page (``read_page``), and ends by naming the pages to answer from
(``finish``). Those pages head the ranking, followed by the one-shot candidates;
answer generation then reads only the selected pages (at most ``max_pages``).

Limits per question: ``max_tool_calls`` tool calls and ``token_budget`` tokens
over all model calls. When either is reached the model must finish.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mmrag.models import image_part, text_part
from mmrag.retrieval.base import Run, top_k_of
from mmrag.retrieval.fusion import ranked_ids
from mmrag.tracing import span

SEARCH_TOOLS = {
    "text": ("text_search", "Semantic search over the parsed text of pages (dense embeddings)."),
    "bm25": ("keyword_search", "Keyword search (BM25): exact names, terms, codes and numbers."),
    "visual": ("visual_search", "Search over page images: charts, tables, figures, layouts."),
}

AGENT_SYSTEM = """You find the pages of a collection of reports that are needed to answer a question.
You do not answer the question yourself; another model will answer from the pages you select.

You start with the results of one search. Then:
- If these results clearly contain everything the question needs, finish at once.
- Otherwise search again. Use different wording from the question, and search separately for
  each part of a question that compares things, lists things or needs several facts.
- Open a page with read_page when a snippet is not enough to tell whether the page has the evidence.
- Finish with the ids of the pages that contain the evidence, most useful first, at most
  {max_pages}. Leave out pages that do not help.
You can make at most {max_tool_calls} tool calls before finishing."""


def function_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    parameters = {"type": "object", "properties": properties, "required": required}
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def snippet(text: str, query: str, width: int = 240) -> str:
    """The stretch of the page that contains the most query words."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= width:
        return text
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 3}
    lower = text.lower()
    starts = range(0, len(text) - width + 1, width // 3)
    best = max(starts, key=lambda s: sum(w in lower[s : s + width] for w in words))
    return text[best : best + width]


class AgentSearch:
    name = "agent"

    def __init__(
        self,
        pipeline: Any,
        client: Any,
        max_tool_calls: int = 8,
        token_budget: int = 60_000,
        max_pages: int = 5,
        results_per_search: int = 8,
        workers: int = 6,
    ) -> None:
        self.pipeline = pipeline
        self.client = client
        self.max_tool_calls = max_tool_calls
        self.token_budget = token_budget
        self.max_pages = max_pages
        self.results_per_search = results_per_search
        self.workers = workers
        self.texts: dict[str, str] = {}
        self.labels: dict[str, str] = {}
        self.selected: dict[str, list[str]] = {}
        self.steps: dict[str, list[dict]] = {}
        self.stats: dict[str, dict] = {}
        self.embedder = getattr(pipeline, "embedder", None)
        self.llm = getattr(pipeline, "llm", None)
        # The text route's event loop and the query encoder process serve one caller at a time.
        self._routes_lock = threading.Lock()

    # -- indexing ------------------------------------------------------------

    def index(self, corpus_ids, corpus_images, corpus_texts, dataset_name=None) -> None:
        self.pipeline.index(corpus_ids, corpus_images, corpus_texts, dataset_name)
        self.texts = dict(zip(map(str, corpus_ids), (t or "" for t in corpus_texts)))

    # -- tools ---------------------------------------------------------------

    def tools(self) -> list[dict]:
        query = {"query": {"type": "string", "description": "what to search for"}}
        tools = [
            function_tool(name, description, query, ["query"])
            for route, (name, description) in SEARCH_TOOLS.items()
            if route in self.pipeline.routes
        ]
        page = {"page_id": {"type": "string"}}
        tools.append(
            function_tool("read_page", "Read one page in full (text or image).", page, ["page_id"])
        )
        pages = {"page_ids": {"type": "array", "items": {"type": "string"}}}
        tools.append(
            function_tool(
                "finish", "Name the pages to answer from, best first.", pages, ["page_ids"]
            )
        )
        return tools

    def _listing(self, scores: dict[str, float], query: str) -> str:
        rows = [
            {"page_id": p, "score": round(scores[p], 4), "snippet": snippet(self.texts[p], query)}
            for p in ranked_ids(scores)[: self.results_per_search]
        ]
        return json.dumps(rows, ensure_ascii=False)

    def _search(self, route: str, query: str) -> str:
        text = self.pipeline.rewriter([query])[0] if self.pipeline.rewriter else query
        with self._routes_lock:
            run = self.pipeline.routes[route].search(["q"], [text], top_k=self.results_per_search)
        return self._listing(run["q"], query)

    def _read_page(self, page_id: str) -> tuple[str, dict | None]:
        """(tool reply, optional image message that has to follow the tool replies)."""
        if page_id not in self.texts:
            return f"There is no page '{page_id}'.", None
        text = self.texts[page_id].strip()
        as_image = self.pipeline.page_routes.get(page_id) == "visual" or not text
        if as_image and self.pipeline.image_provider is not None:
            with self._routes_lock:
                image = dict(self.pipeline.image_provider([page_id]))[page_id]
            parts = [text_part(f"[PAGE {page_id}] as an image:"), image_part(image)]
            return "The page image follows.", {"role": "user", "content": parts}
        return text[:8000], None

    # -- one question ----------------------------------------------------------

    def _run(self, query: str, fused: dict[str, float]) -> tuple[list[str], list[dict], dict]:
        system = AGENT_SYSTEM.format(max_pages=self.max_pages, max_tool_calls=self.max_tool_calls)
        first = f"Question: {query}\n\nResults of the first search:\n{self._listing(fused, query)}"
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": first},
        ]
        routes = {name: route for route, (name, _) in SEARCH_TOOLS.items()}
        steps: list[dict] = []
        tokens = calls = 0
        selected: list[str] | None = None
        forced = False
        while selected is None:
            forced = calls >= self.max_tool_calls or tokens >= self.token_budget
            choice: Any = "required"
            if forced:
                choice = {"type": "function", "function": {"name": "finish"}}
            message, used = self.client.chat(messages, tools=self.tools(), tool_choice=choice)
            tokens += used
            tool_calls = message.tool_calls or []
            if not tool_calls:  # the model replied in prose: keep the one-shot order
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in tool_calls
                    ],
                }
            )
            images = []
            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                steps.append({"tool": name, "args": args})
                if name == "finish":
                    pages = [str(p) for p in args.get("page_ids") or []]
                    selected = [p for p in dict.fromkeys(pages) if p in self.texts]
                    reply = "ok"
                elif name in routes and args.get("query"):
                    calls += 1
                    reply = self._search(routes[name], str(args["query"]))
                elif name == "read_page":
                    calls += 1
                    reply, image = self._read_page(str(args.get("page_id")))
                    if image is not None:
                        images.append(image)
                else:
                    reply = f"Unknown tool or missing argument: {name}"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": reply})
            messages.extend(images)
        stats = {"tool_calls": calls, "tokens": tokens, "forced_finish": forced}
        return (selected or [])[: self.max_pages], steps, stats

    # -- search --------------------------------------------------------------

    def search(self, query_ids: list[str], queries: list[str], top_k: int = 10) -> Run:
        query_ids = [str(q) for q in query_ids]
        depth = max(self.pipeline.candidates, top_k)
        fused = self.pipeline.search(query_ids, list(queries), top_k=depth)

        def one(item: tuple[str, str]) -> tuple[str, list[str], list[dict], dict]:
            qid, query = item
            started = time.perf_counter()
            with span("agent question", question=query):
                selected, steps, stats = self._run(query, fused[qid])
            stats["seconds"] = round(time.perf_counter() - started, 2)
            return qid, selected, steps, stats

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            done = list(pool.map(one, zip(query_ids, queries)))
        run: Run = {}
        for qid, selected, steps, stats in done:
            self.selected[qid], self.steps[qid], self.stats[qid] = selected, steps, stats
            # Selected pages first, also those only the agent's own searches found.
            order = selected + [p for p in ranked_ids(fused[qid]) if p not in selected]
            run[qid] = top_k_of({p: 1.0 / (rank + 1) for rank, p in enumerate(order)}, top_k)
        return run

    # -- reporting -----------------------------------------------------------

    def storage_bytes(self) -> int:
        return self.pipeline.storage_bytes()

    def info(self) -> dict:
        info = self.pipeline.info()
        if self.stats:
            n = len(self.stats)
            info["agent"] = {
                "questions": n,
                "tool_calls_per_question": sum(s["tool_calls"] for s in self.stats.values()) / n,
                "tokens_per_question": sum(s["tokens"] for s in self.stats.values()) / n,
                "seconds_per_question": sum(s["seconds"] for s in self.stats.values()) / n,
                "pages_selected_per_question": sum(len(p) for p in self.selected.values()) / n,
                "forced_finishes": sum(s["forced_finish"] for s in self.stats.values()),
                "no_selection": sum(1 for p in self.selected.values() if not p),
                "usage": self.client.usage() if hasattr(self.client, "usage") else None,
            }
            info["selected_pages"] = dict(self.selected)
            info["agent_steps"] = {q: {**self.stats[q], "steps": self.steps[q]} for q in self.steps}
        return info

    def close(self) -> None:
        self.pipeline.close()
