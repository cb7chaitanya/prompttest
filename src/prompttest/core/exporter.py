"""Export evaluation results to JSON and CSV."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prompttest.core.eval_runner import EvalResult
from prompttest.core.models import PromptConfig, Verdict


def _build_payload(
    result: EvalResult,
    prompt_config: PromptConfig,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build the canonical dict representation of an eval result."""
    ts = timestamp or datetime.now(timezone.utc)

    cases: list[dict[str, Any]] = []
    for cr in result.case_results:
        cases.append({
            "input": cr.case.input,
            "expected": cr.case.expected,
            "actual": cr.output,
            "score": cr.score,
            "passed": cr.verdict == Verdict.PASS,
            "verdict": cr.verdict.value,
            "reason": cr.reason,
        })

    return {
        "metadata": {
            "timestamp": ts.isoformat(),
            "prompt_name": result.prompt_name,
            "prompt_version": result.prompt_version,
            "model": prompt_config.model,
            "provider": prompt_config.provider,
            "scorer": result.scoring,
        },
        "summary": {
            "total": result.total,
            "passed": result.passed,
            "failed": result.failed,
            "errors": result.errors,
            "accuracy": result.accuracy,
            "average_score": result.average_score,
            "pass_threshold": result.pass_threshold,
        },
        "results": cases,
    }


def export_json(
    result: EvalResult,
    prompt_config: PromptConfig,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Serialize an eval result to a JSON string."""
    payload = _build_payload(result, prompt_config, timestamp)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def export_csv(
    result: EvalResult,
    prompt_config: PromptConfig,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Serialize an eval result to a CSV string."""
    payload = _build_payload(result, prompt_config, timestamp)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "index", "input", "expected", "actual",
        "score", "passed", "verdict", "reason",
    ])
    for i, case in enumerate(payload["results"], 1):
        input_str = json.dumps(case["input"], ensure_ascii=False)
        writer.writerow([
            i,
            input_str,
            case["expected"],
            case["actual"],
            case["score"],
            case["passed"],
            case["verdict"],
            case["reason"],
        ])
    return buf.getvalue()


def export_diff_json(
    diff_text: str,
    name: str,
    version_a: str,
    version_b: str,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Serialize a prompt diff to a JSON string."""
    ts = timestamp or datetime.now(timezone.utc)
    payload = {
        "metadata": {
            "timestamp": ts.isoformat(),
            "prompt_name": name,
            "version_a": version_a,
            "version_b": version_b,
        },
        "diff": diff_text,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


_EXPORTERS = {
    "json": export_json,
    "csv": export_csv,
}


def save_result(
    result: EvalResult,
    prompt_config: PromptConfig,
    path: Path,
    fmt: str,
    *,
    timestamp: datetime | None = None,
) -> Path:
    """Export *result* to *path* in the given format and return the resolved path."""
    if fmt not in _EXPORTERS:
        raise ValueError(f"Unknown format '{fmt}'. Choose from: {', '.join(_EXPORTERS)}")

    exporter = _EXPORTERS[fmt]
    content = exporter(result, prompt_config, timestamp=timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path.resolve()


def auto_filename(
    prompt_config: PromptConfig,
    fmt: str,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Generate a timestamped filename like ``summarize_v1_20260317T120000Z.json``."""
    ts = timestamp or datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y%m%dT%H%M%SZ")
    return f"{prompt_config.name}_v{prompt_config.version}_{ts_str}.{fmt}"
