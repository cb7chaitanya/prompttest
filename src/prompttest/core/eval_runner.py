"""Evaluation dataset engine: load, template, run, and score."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from prompttest.core.models import PromptConfig, Verdict
from prompttest.core.scoring import DEFAULT_SCORER, PASS_THRESHOLD, get_scorer
from prompttest.providers.base import LLMProvider
from prompttest.providers.registry import get_provider


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """A single evaluation test case with dict-based input."""

    input: dict[str, str]
    expected: str
    tags: list[str] = field(default_factory=list)

    @property
    def input_summary(self) -> str:
        """Short string representation of the input dict."""
        parts = [f"{k}={v!r}" for k, v in self.input.items()]
        return ", ".join(parts)


@dataclass
class EvalCaseResult:
    """Result of evaluating a single eval case."""

    case: EvalCase
    output: str
    verdict: Verdict
    score: float
    reason: str


@dataclass
class EvalDataset:
    """An evaluation dataset loaded from YAML."""

    prompt: str
    scoring: str
    tests: list[EvalCase]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalDataset:
        tests = []
        for t in data["tests"]:
            inp = t["input"]
            if isinstance(inp, str):
                inp = {"input": inp}
            tests.append(
                EvalCase(
                    input=inp,
                    expected=t["expected"],
                    tags=t.get("tags", []),
                )
            )
        return cls(
            prompt=data["prompt"],
            scoring=data.get("scoring", DEFAULT_SCORER),
            tests=tests,
        )


@dataclass
class EvalResult:
    """Aggregated evaluation result."""

    prompt_name: str
    prompt_version: str
    scoring: str
    case_results: list[EvalCaseResult]

    @property
    def total(self) -> int:
        return len(self.case_results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.case_results if r.verdict == Verdict.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.case_results if r.verdict == Verdict.FAIL)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.case_results if r.verdict == Verdict.ERROR)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Templating
# ---------------------------------------------------------------------------

def render_template_dict(template: str, variables: dict[str, str]) -> str:
    """Replace all ``{{key}}`` placeholders in *template* with values from *variables*."""
    result = template
    for key, value in variables.items():
        result = result.replace("{{" + key + "}}", value)
    return result


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_eval_dataset(path: Path) -> EvalDataset:
    """Load an evaluation dataset from a YAML file."""
    data = yaml.safe_load(path.read_text())
    return EvalDataset.from_dict(data)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _score_case(
    case: EvalCase,
    output: str,
    scorer: Any,
) -> EvalCaseResult:
    score, reason = scorer(output, case.expected)
    verdict = Verdict.PASS if score >= PASS_THRESHOLD else Verdict.FAIL
    return EvalCaseResult(case=case, output=output, verdict=verdict, score=score, reason=reason)


def _error_case(case: EvalCase, exc: Exception) -> EvalCaseResult:
    return EvalCaseResult(
        case=case, output=str(exc), verdict=Verdict.ERROR, score=0.0,
        reason=f"provider error: {exc}",
    )


def run_eval(
    dataset_path: Path,
    prompt_config: PromptConfig,
    provider_override: LLMProvider | None = None,
    *,
    strict: bool = True,
) -> EvalResult:
    """Run an evaluation dataset against a prompt and return scored results.

    When *strict* is ``True`` (default), a :class:`~prompttest.validation.prompt_validator.ValidationError`
    is raised if any test case is missing required placeholders.  When ``False``,
    missing placeholders are silently skipped and only warnings are emitted.
    """
    from prompttest.validation.prompt_validator import validate_dataset

    dataset = load_eval_dataset(dataset_path)

    # --- Validate before running ---
    validation = validate_dataset(prompt_config, dataset)
    if validation.errors and strict:
        raise validation.errors[0]

    scorer = get_scorer(dataset.scoring)
    provider = provider_override or get_provider(prompt_config.provider)

    case_results: list[EvalCaseResult] = []
    for case in dataset.tests:
        user_message = render_template_dict(prompt_config.template, case.input)
        try:
            output = provider.complete(
                model=prompt_config.model,
                system=prompt_config.system,
                user_message=user_message,
                parameters=prompt_config.parameters,
            )
            case_results.append(_score_case(case, output, scorer))
        except Exception as exc:
            case_results.append(_error_case(case, exc))

    return EvalResult(
        prompt_name=prompt_config.name,
        prompt_version=prompt_config.version,
        scoring=dataset.scoring,
        case_results=case_results,
    )


async def run_eval_async(
    dataset_path: Path,
    prompt_config: PromptConfig,
    provider_override: LLMProvider | None = None,
    *,
    strict: bool = True,
) -> EvalResult:
    """Async version of :func:`run_eval` — runs all cases concurrently."""
    from prompttest.validation.prompt_validator import validate_dataset

    dataset = load_eval_dataset(dataset_path)

    # --- Validate before running ---
    validation = validate_dataset(prompt_config, dataset)
    if validation.errors and strict:
        raise validation.errors[0]

    scorer = get_scorer(dataset.scoring)
    provider = provider_override or get_provider(prompt_config.provider)

    async def _run_one(case: EvalCase) -> EvalCaseResult:
        user_message = render_template_dict(prompt_config.template, case.input)
        try:
            output = await provider.acomplete(
                model=prompt_config.model,
                system=prompt_config.system,
                user_message=user_message,
                parameters=prompt_config.parameters,
            )
            return _score_case(case, output, scorer)
        except Exception as exc:
            return _error_case(case, exc)

    case_results = await asyncio.gather(*[_run_one(c) for c in dataset.tests])

    return EvalResult(
        prompt_name=prompt_config.name,
        prompt_version=prompt_config.version,
        scoring=dataset.scoring,
        case_results=list(case_results),
    )
