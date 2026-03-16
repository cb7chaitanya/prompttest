"""Orchestrate test runs: load data, call provider, evaluate."""

from __future__ import annotations

from pathlib import Path

from prompttest.core.evaluator import evaluate_case
from prompttest.core.loader import discover_datasets, find_prompt_by_name
from prompttest.core.models import CaseResult, RunResult, Verdict
from prompttest.providers.registry import get_provider


def render_template(template: str, input_text: str) -> str:
    """Replace {{input}} placeholder in the prompt template."""
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
                result = evaluate_case(case, output)
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
