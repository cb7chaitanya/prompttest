"""HTML report generation for evaluation results."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prompttest.core.eval_runner import EvalResult
from prompttest.core.models import PromptConfig, Verdict

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
  :root {
    --green: #22c55e; --red: #ef4444; --yellow: #eab308;
    --gray: #6b7280; --light: #f9fafb; --border: #e5e7eb;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         color: #111827; background: #fff; line-height: 1.5; padding: 2rem; max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  .subtitle { color: var(--gray); font-size: 0.875rem; margin-bottom: 1.5rem; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
  .card { background: var(--light); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center; }
  .card .value { font-size: 1.75rem; font-weight: 700; }
  .card .label { font-size: 0.75rem; color: var(--gray); text-transform: uppercase; letter-spacing: 0.05em; }
  .pass { color: var(--green); } .fail { color: var(--red); } .error { color: var(--red); font-weight: 700; }
  .chart-section { margin-bottom: 2rem; }
  .chart-section h2 { font-size: 1.125rem; margin-bottom: 0.75rem; }
  .bar-chart { display: flex; flex-direction: column; gap: 4px; }
  .bar-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; }
  .bar-row .bar-label { width: 30px; text-align: right; color: var(--gray); flex-shrink: 0; }
  .bar-row .bar-track { flex: 1; height: 20px; background: var(--light); border-radius: 4px; overflow: hidden; border: 1px solid var(--border); }
  .bar-row .bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  .bar-row .bar-val { width: 40px; font-size: 0.75rem; color: var(--gray); }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-bottom: 2rem; }
  th { background: var(--light); text-align: left; padding: 0.5rem 0.75rem; border-bottom: 2px solid var(--border);
       font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gray); }
  td { padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); vertical-align: top; }
  tr.row-fail { background: #fef2f2; }
  tr.row-error { background: #fef2f2; }
  tr.critical-fail { background: #fecaca; border-left: 3px solid var(--red); }
  .critical-badge { background: #dc2626; color: #fff; font-size: 0.6rem; padding: 0.1rem 0.35rem; border-radius: 4px; margin-left: 4px; vertical-align: middle; }
  .badge { display: inline-block; padding: 0.125rem 0.5rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; }
  .badge-pass { background: #dcfce7; color: #166534; }
  .badge-fail { background: #fee2e2; color: #991b1b; }
  .badge-error { background: #fee2e2; color: #991b1b; }
  .score-bar { display: inline-block; width: 60px; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; vertical-align: middle; margin-right: 4px; }
  .score-fill { height: 100%; border-radius: 4px; }
  .meta { font-size: 0.75rem; color: var(--gray); margin-top: 2rem; border-top: 1px solid var(--border); padding-top: 1rem; }
  .truncate { max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
</head>
<body>

<h1>{{ title }}</h1>
<p class="subtitle">{{ subtitle }}</p>

<div class="cards">
  <div class="card">
    <div class="value">{{ summary.total }}</div>
    <div class="label">Total</div>
  </div>
  <div class="card">
    <div class="value pass">{{ summary.passed }}</div>
    <div class="label">Passed</div>
  </div>
  <div class="card">
    <div class="value fail">{{ summary.failed }}</div>
    <div class="label">Failed</div>
  </div>
  {% if summary.errors > 0 %}
  <div class="card">
    <div class="value error">{{ summary.errors }}</div>
    <div class="label">Errors</div>
  </div>
  {% endif %}
  <div class="card">
    <div class="value">{{ "%.0f" | format(summary.accuracy * 100) }}%</div>
    <div class="label">Pass Rate</div>
  </div>
  <div class="card">
    <div class="value" style="color: {{ 'var(--green)' if summary.average_score >= summary.pass_threshold else 'var(--red)' }}">
      {{ "%.2f" | format(summary.average_score) }}
    </div>
    <div class="label">Avg Score</div>
  </div>
  <div class="card">
    <div class="value">{{ "%.2f" | format(summary.pass_threshold) }}</div>
    <div class="label">Threshold</div>
  </div>
</div>

<div class="chart-section">
  <h2>Score Distribution</h2>
  <div class="bar-chart">
    {% for bucket in score_buckets %}
    <div class="bar-row">
      <span class="bar-label">{{ bucket.label }}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width: {{ bucket.pct }}%; background: {{ bucket.color }};"></div>
      </div>
      <span class="bar-val">{{ bucket.count }}</span>
    </div>
    {% endfor %}
  </div>
</div>

<h2 style="font-size: 1.125rem; margin-bottom: 0.75rem;">Test Results</h2>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Input</th>
      <th>Expected</th>
      <th>Output</th>
      <th>Score</th>
      <th>Verdict</th>
      <th>Reason</th>
    </tr>
  </thead>
  <tbody>
    {% for r in results %}
    <tr class="{{ r.row_class }}">
      <td>{{ r.index }}{% if r.critical %}<span class="critical-badge">CRITICAL</span>{% endif %}</td>
      <td class="truncate" title="{{ r.input_escaped }}">{{ r.input_display }}</td>
      <td class="truncate" title="{{ r.expected_escaped }}">{{ r.expected_display }}</td>
      <td class="truncate" title="{{ r.actual_escaped }}">{{ r.actual_display }}</td>
      <td>
        <span class="score-bar"><span class="score-fill" style="width: {{ "%.0f" | format(r.score * 100) }}%; background: {{ r.score_color }};"></span></span>
        {{ "%.2f" | format(r.score) }}
      </td>
      <td><span class="badge badge-{{ r.verdict }}">{{ r.verdict | upper }}</span></td>
      <td class="truncate" title="{{ r.reason_escaped }}">{{ r.reason_display }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

<div class="meta">
  <strong>{{ metadata.prompt_name }}</strong> v{{ metadata.prompt_version }}
  &middot; model: {{ metadata.model }}
  &middot; provider: {{ metadata.provider }}
  &middot; scorer: {{ metadata.scorer }}
  &middot; generated: {{ metadata.timestamp }}
</div>

</body>
</html>
"""


def _truncate(text: str, maxlen: int = 80) -> str:
    return text[:maxlen] + ("..." if len(text) > maxlen else "")


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "var(--green)"
    if score >= 0.5:
        return "var(--yellow)"
    return "var(--red)"


def _build_score_buckets(scores: list[float]) -> list[dict[str, Any]]:
    """Build histogram buckets for score distribution."""
    buckets = [
        {"label": "0.0", "min": 0.0, "max": 0.1},
        {"label": "0.1", "min": 0.1, "max": 0.2},
        {"label": "0.2", "min": 0.2, "max": 0.3},
        {"label": "0.3", "min": 0.3, "max": 0.4},
        {"label": "0.4", "min": 0.4, "max": 0.5},
        {"label": "0.5", "min": 0.5, "max": 0.6},
        {"label": "0.6", "min": 0.6, "max": 0.7},
        {"label": "0.7", "min": 0.7, "max": 0.8},
        {"label": "0.8", "min": 0.8, "max": 0.9},
        {"label": "0.9", "min": 0.9, "max": 1.01},
    ]
    max_count = 0
    for b in buckets:
        b["count"] = sum(1 for s in scores if b["min"] <= s < b["max"])
        max_count = max(max_count, b["count"])

    for b in buckets:
        b["pct"] = (b["count"] / max_count * 100) if max_count > 0 else 0
        b["color"] = _score_color(b["min"] + 0.05)

    return buckets


def export_html(
    result: EvalResult,
    prompt_config: PromptConfig,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Render an eval result to a self-contained HTML report string."""
    try:
        from jinja2 import Template
    except ImportError:
        raise ImportError(
            "Jinja2 is required for HTML reports: pip install prompttest[report]"
        )

    ts = timestamp or datetime.now(timezone.utc)

    metadata = {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "prompt_name": result.prompt_name,
        "prompt_version": result.prompt_version,
        "model": prompt_config.model,
        "provider": prompt_config.provider,
        "scorer": result.scoring,
    }

    summary = {
        "total": result.total,
        "passed": result.passed,
        "failed": result.failed,
        "errors": result.errors,
        "accuracy": result.accuracy,
        "average_score": result.average_score,
        "pass_threshold": result.pass_threshold,
    }

    rows = []
    scores = []
    for i, cr in enumerate(result.case_results, 1):
        input_str = ", ".join(f"{k}={v!r}" for k, v in cr.case.input.items())
        scores.append(cr.score)
        is_critical_fail = cr.case.critical and cr.verdict != Verdict.PASS
        row_class = f"row-{cr.verdict.value}"
        if is_critical_fail:
            row_class = "critical-fail"

        rows.append({
            "index": i,
            "input_display": html.escape(_truncate(input_str)),
            "input_escaped": html.escape(input_str),
            "expected_display": html.escape(_truncate(cr.case.expected)),
            "expected_escaped": html.escape(cr.case.expected),
            "actual_display": html.escape(_truncate(cr.output)),
            "actual_escaped": html.escape(cr.output),
            "score": cr.score,
            "score_color": _score_color(cr.score),
            "verdict": cr.verdict.value,
            "reason_display": html.escape(_truncate(cr.reason)),
            "reason_escaped": html.escape(cr.reason),
            "critical": cr.case.critical,
            "row_class": row_class,
        })

    score_buckets = _build_score_buckets(scores)

    template = Template(_TEMPLATE)
    return template.render(
        title=f"Eval Report: {result.prompt_name} v{result.prompt_version}",
        subtitle=f"{prompt_config.model} via {prompt_config.provider} | {result.scoring} scorer | {ts.strftime('%Y-%m-%d %H:%M UTC')}",
        metadata=metadata,
        summary=summary,
        results=rows,
        score_buckets=score_buckets,
    )


def save_html_report(
    result: EvalResult,
    prompt_config: PromptConfig,
    path: Path,
    *,
    timestamp: datetime | None = None,
) -> Path:
    """Render and save an HTML report to *path*."""
    content = export_html(result, prompt_config, timestamp=timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path.resolve()
