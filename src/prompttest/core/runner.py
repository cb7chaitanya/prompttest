"""Orchestrate test runs: load data, call provider, evaluate."""

from __future__ import annotations

from pathlib import Path

from prompttest.core.loader import discover_datasets, find_prompt_by_name
from prompttest.core.models import CaseResult, RunResult, TestCase, Verdict
from prompttest.core.scoring import PASS_THRESHOLD, contains
from prompttest.providers.registry import get_provider


def render_template(template: str, input_text: str | dict) -> str:
    """Replace placeholders in the prompt template.

    Supports both a plain string (replaces ``{{input}}``) and a dict
    (replaces each ``{{key}}`` with its value).
    """
    if isinstance(input_text, dict):
        result = template
        for key, value in input_text.items():
            result = result.replace("{{" + key + "}}", str(value))
        return result
    return template.replace("{{input}}", input_text)


def run_all(root: Path | None = None) -> list[RunResult]:
    """Run every dataset against its linked prompt."""
    datasets = discover_datasets(root)
    results: list[RunResult] = []
    for ds in datasets:
        prompt = find_prompt_by_name(ds.prompt, root)
        if prompt is None:
            raise ValueError(f"Prompt '{ds.prompt}' not found for dataset '{ds.name}'")
        provider = get_provider(prompt.provider)
        case_results: list[CaseResult] = []
        for case in ds.cases:
            user_message = render_template(prompt.template, case.input)
            try:
                output = provider.complete(
                    model=prompt.model,
                    system=prompt.system,
                    user_message=user_message,
                    parameters=prompt.parameters,
                )
                score, reason = contains(output, case.expected)
                verdict = Verdict.PASS if score >= PASS_THRESHOLD else Verdict.FAIL
                result = CaseResult(
                    case=case, output=output, verdict=verdict,
                    score=score, reason=reason,
                )
            except Exception as exc:
                result = CaseResult(
                    case=case,
                    output=str(exc),
                    verdict=Verdict.ERROR,
                    score=0.0,
                    reason=f"provider error: {exc}",
                )
            case_results.append(result)
        results.append(
            RunResult(
                prompt_name=prompt.name,
                prompt_version=prompt.version,
                dataset_name=ds.name,
                results=case_results,
            )
        )
    return results
