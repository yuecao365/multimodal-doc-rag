"""Helpers shared by the experiment scripts: result directories, environment capture."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

TRACKED_PACKAGES = (
    "raganything", "lightrag-hku", "pyarrow", "pytrec-eval-terrier", "openai", "numpy",
    "scipy", "snowballstemmer", "torch", "colpali-engine", "transformers",
)  # fmt: skip


def results_dir(name: str, date: str | None = None) -> Path:
    """results/<date>_<name>/ under the project root, created if needed."""
    path = ROOT / "results" / f"{date or dt.date.today().isoformat()}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def environment() -> dict:
    """What is needed to reproduce a run: interpreter, commit, lock file, key packages."""

    def version(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return out.stdout.strip() or None

    lock = ROOT / "uv.lock"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "packages": {p: version(p) for p in TRACKED_PACKAGES},
    }
