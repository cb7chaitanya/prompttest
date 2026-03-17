"""Performance history: append and query eval results over time."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prompttest.core.eval_runner import EvalResult
from prompttest.core.models import PromptConfig

HISTORY_DIR = "history"
HISTORY_FILE = "runs.jsonl"


@dataclass
class HistoryEntry:
    """A single recorded eval run."""

    timestamp: str
    prompt_name: str
    prompt_version: str
    model: str
    provider: str
    scorer: str
    total: int
    passed: int
    failed: int
    errors: int
    accuracy: float
    average_score: float
    pass_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "provider": self.provider,
            "scorer": self.scorer,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "accuracy": self.accuracy,
            "average_score": self.average_score,
            "pass_threshold": self.pass_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEntry:
        return cls(
            timestamp=data["timestamp"],
            prompt_name=data["prompt_name"],
            prompt_version=data["prompt_version"],
            model=data["model"],
            provider=data["provider"],
            scorer=data["scorer"],
            total=int(data["total"]),
            passed=int(data["passed"]),
            failed=int(data["failed"]),
            errors=int(data["errors"]),
            accuracy=float(data["accuracy"]),
            average_score=float(data["average_score"]),
            pass_threshold=float(data["pass_threshold"]),
        )


def _history_path(root: Path) -> Path:
    return root / HISTORY_DIR / HISTORY_FILE


def record(
    root: Path,
    result: EvalResult,
    prompt_config: PromptConfig,
    *,
    timestamp: datetime | None = None,
) -> HistoryEntry:
    """Append an eval result to the history file and return the entry."""
    ts = timestamp or datetime.now(timezone.utc)
    entry = HistoryEntry(
        timestamp=ts.isoformat(),
        prompt_name=result.prompt_name,
        prompt_version=result.prompt_version,
        model=prompt_config.model,
        provider=prompt_config.provider,
        scorer=result.scoring,
        total=result.total,
        passed=result.passed,
        failed=result.failed,
        errors=result.errors,
        accuracy=result.accuracy,
        average_score=result.average_score,
        pass_threshold=result.pass_threshold,
    )

    path = _history_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    return entry


def load_history(
    root: Path,
    *,
    prompt_name: str | None = None,
) -> list[HistoryEntry]:
    """Load all history entries, optionally filtered by prompt name."""
    path = _history_path(root)
    if not path.exists():
        return []

    entries: list[HistoryEntry] = []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        entry = HistoryEntry.from_dict(data)
        if prompt_name is not None and entry.prompt_name != prompt_name:
            continue
        entries.append(entry)
    return entries


def detect_trend(entries: list[HistoryEntry], window: int = 3) -> str:
    """Detect score trend from the last *window* entries.

    Returns ``"improving"``, ``"degrading"``, or ``"stable"``.
    """
    if len(entries) < 2:
        return "stable"

    recent = entries[-window:] if len(entries) >= window else entries
    scores = [e.average_score for e in recent]

    deltas = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
    avg_delta = sum(deltas) / len(deltas)

    if avg_delta > 0.01:
        return "improving"
    if avg_delta < -0.01:
        return "degrading"
    return "stable"
