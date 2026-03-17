"""Baseline comparison: compare current eval results against a saved baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompttest.core.eval_runner import EvalResult


@dataclass
class CaseDiff:
    """Score difference for a single test case."""

    index: int
    input_summary: str
    expected: str
    baseline_score: float
    current_score: float

    @property
    def delta(self) -> float:
        return self.current_score - self.baseline_score

    @property
    def regressed(self) -> bool:
        return self.delta < 0


@dataclass
class BaselineComparison:
    """Full comparison between a baseline and a current eval run."""

    baseline_avg: float
    current_avg: float
    baseline_pass_rate: float
    current_pass_rate: float
    case_diffs: list[CaseDiff] = field(default_factory=list)

    @property
    def avg_delta(self) -> float:
        return self.current_avg - self.baseline_avg

    @property
    def pass_rate_delta(self) -> float:
        return self.current_pass_rate - self.baseline_pass_rate

    @property
    def regressions(self) -> list[CaseDiff]:
        return [d for d in self.case_diffs if d.regressed]

    @property
    def improvements(self) -> list[CaseDiff]:
        return [d for d in self.case_diffs if d.delta > 0]

    @property
    def has_regression(self) -> bool:
        return self.avg_delta < 0 or len(self.regressions) > 0


def load_baseline(path: Path) -> dict[str, Any]:
    """Load a baseline JSON file (produced by ``--output results.json``)."""
    data = json.loads(path.read_text())
    if "summary" not in data or "results" not in data:
        raise ValueError(
            f"Invalid baseline file: {path} — expected 'summary' and 'results' keys"
        )
    return data


def compare(baseline_data: dict[str, Any], current: EvalResult) -> BaselineComparison:
    """Compare *current* eval result against loaded *baseline_data*."""
    b_summary = baseline_data["summary"]
    b_results = baseline_data["results"]

    baseline_avg = float(b_summary.get("average_score", b_summary.get("accuracy", 0)))
    baseline_total = int(b_summary.get("total", 0))
    baseline_passed = int(b_summary.get("passed", 0))
    baseline_pass_rate = baseline_passed / baseline_total if baseline_total else 0.0

    current_avg = current.average_score
    current_pass_rate = current.accuracy

    # Per-case comparison (matched by index)
    case_diffs: list[CaseDiff] = []
    for i, cr in enumerate(current.case_results):
        baseline_score = 0.0
        if i < len(b_results):
            baseline_score = float(b_results[i].get("score", 0))

        case_diffs.append(CaseDiff(
            index=i + 1,
            input_summary=cr.case.input_summary[:60],
            expected=cr.case.expected,
            baseline_score=baseline_score,
            current_score=cr.score,
        ))

    return BaselineComparison(
        baseline_avg=baseline_avg,
        current_avg=current_avg,
        baseline_pass_rate=baseline_pass_rate,
        current_pass_rate=current_pass_rate,
        case_diffs=case_diffs,
    )
