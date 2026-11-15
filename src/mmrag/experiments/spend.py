"""Running total of money spent on model APIs, kept in results/spend.md.

Every script that calls a paid model records what it spent. ``results/spend.jsonl``
is the log; ``results/spend.md`` is rendered from it. A script refuses to start once
the total has reached the cap (``BUDGET_USD`` in .env, default 25).
"""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any

from mmrag.experiments import common


def total_cost(usage: Any) -> float:
    """Sum of every ``cost_usd`` found anywhere inside a nested usage report."""
    if isinstance(usage, dict):
        own = float(usage.get("cost_usd") or 0.0)
        return own + sum(total_cost(v) for k, v in usage.items() if k != "cost_usd")
    if isinstance(usage, list):
        return sum(total_cost(v) for v in usage)
    return 0.0


def _rows() -> list[dict]:
    path = common.ROOT / "results" / "spend.jsonl"
    return common.read_jsonl(path) if path.exists() else []


def spent() -> float:
    return sum(row["usd"] for row in _rows())


def check_budget() -> None:
    cap = float(os.getenv("BUDGET_USD", "25"))
    if spent() >= cap:
        raise RuntimeError(f"Spending cap reached: {spent():.2f} of {cap:.2f} USD")


def record(run: str, what: str, usage: Any) -> float:
    """Add one paid step to the log and rewrite spend.md. Free steps are not logged."""
    usd = total_cost(usage)
    if usd <= 0:
        return 0.0
    rows = [*_rows(), {"time": dt.datetime.now().isoformat(timespec="seconds"), "run": run,
                       "what": what, "usd": round(usd, 5)}]  # fmt: skip
    results = common.ROOT / "results"
    results.mkdir(exist_ok=True)
    with (results / "spend.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    lines = [
        "# API spend",
        "",
        "| time | run | what | USD | running total |",
        "|---|---|---|---|---|",
    ]
    running = 0.0
    for row in rows:
        running += row["usd"]
        lines.append(
            f"| {row['time']} | {row['run']} | {row['what']} | {row['usd']:.4f} | {running:.4f} |"
        )
    cap = float(os.getenv("BUDGET_USD", "25"))
    lines += ["", f"Total: {running:.2f} USD of a {cap:.0f} USD cap.", ""]
    (results / "spend.md").write_text("\n".join(lines), encoding="utf-8")
    return usd
