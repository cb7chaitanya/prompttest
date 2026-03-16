"""Evaluation dataset engine: load, template, run, and score."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from prompttest.core.models import PromptConfig, Verdict
from prompttest.core.scoring import DEFAULT_SCORER, PASS_THRESHOLD, get_scorer
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

def run_eval(
    dataset_path: Path,
    prompt_config: PromptConfig,
) -> EvalResult:
    """Run an evaluation dataset against a prompt and return scored results."""
    dataset = load_eval_dataset(dataset_path)
    scorer = get_scorer(dataset.scoring)
    provider = get_provider(prompt_config.provider)

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
            score, reason = scorer(output, case.expected)
            verdict = Verdict.PASS if score >= PASS_THRESHOLD else Verdict.FAIL
        except Exception as exc:
            output = str(exc)
            score = 0.0
            reason = f"provider error: {exc}"
            verdict = Verdict.ERROR

        case_results.append(
            EvalCaseResult(
                case=case,
                output=output,
                verdict=verdict,
                score=score,
                reason=reason,
            )
        )

    return EvalResult(
        prompt_name=prompt_config.name,
        prompt_version=prompt_config.version,
        scoring=dataset.scoring,
        case_results=case_results,
    )
