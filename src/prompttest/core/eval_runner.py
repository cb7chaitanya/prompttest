"""Evaluation dataset engine: load, template, run, and score."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from prompttest.core.concurrency import ConcurrencyConfig, RateLimiter, run_with_retry
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
    critical: bool = False

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
        for t in data.get("tests", data.get("cases", [])):
            inp = t["input"]
            if isinstance(inp, str):
                inp = {"input": inp}
            tests.append(
                EvalCase(
                    input=inp,
                    expected=t["expected"],
                    tags=t.get("tags", []),
                    critical=bool(t.get("critical", False)),
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
    pass_threshold: float = PASS_THRESHOLD

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

    @property
    def average_score(self) -> float:
        """Mean score across all non-error cases (0.0 if none)."""
        scored = [r.score for r in self.case_results if r.verdict != Verdict.ERROR]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def critical_total(self) -> int:
        """Number of test cases marked as critical."""
        return sum(1 for r in self.case_results if r.case.critical)

    @property
    def critical_failed(self) -> int:
        """Number of critical test cases that failed or errored."""
        return sum(
            1 for r in self.case_results
            if r.case.critical and r.verdict != Verdict.PASS
        )


# ---------------------------------------------------------------------------
# Tag filtering
# ---------------------------------------------------------------------------

def filter_by_tags(
    dataset: EvalDataset,
    tags: list[str],
    match: str = "any",
) -> tuple[int, int]:
    """Filter *dataset*.tests in-place, keeping only cases that match *tags*.

    *match* is ``"any"`` (case has at least one of the tags) or ``"all"``
    (case has every tag).

    Returns ``(original_count, filtered_count)``.
    """
    if not tags:
        n = len(dataset.tests)
        return n, n

    tag_set = set(tags)

    if match == "all":
        filtered = [c for c in dataset.tests if tag_set.issubset(c.tags)]
    else:
        filtered = [c for c in dataset.tests if tag_set.intersection(c.tags)]

    original = len(dataset.tests)
    dataset.tests = filtered
    return original, len(filtered)


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
    pass_threshold: float = PASS_THRESHOLD,
) -> EvalCaseResult:
    score, reason = scorer(output, case.expected)
    verdict = Verdict.PASS if score >= pass_threshold else Verdict.FAIL
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
    pass_threshold: float = PASS_THRESHOLD,
) -> EvalResult:
    """Run an evaluation dataset against a prompt and return scored results.

    When *strict* is ``True`` (default), a :class:`~prompttest.validation.prompt_validator.ValidationError`
    is raised if any test case is missing required placeholders.  When ``False``,
    missing placeholders are silently skipped and only warnings are emitted.

    *pass_threshold* sets the minimum score for a case to be considered passing.
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
            case_results.append(_score_case(case, output, scorer, pass_threshold))
        except Exception as exc:
            case_results.append(_error_case(case, exc))

    return EvalResult(
        prompt_name=prompt_config.name,
        prompt_version=prompt_config.version,
        scoring=dataset.scoring,
        case_results=case_results,
        pass_threshold=pass_threshold,
    )


async def run_eval_async(
    dataset_path: Path,
    prompt_config: PromptConfig,
    provider_override: LLMProvider | None = None,
    *,
    strict: bool = True,
    pass_threshold: float = PASS_THRESHOLD,
    concurrency_config: ConcurrencyConfig | None = None,
    on_case_complete: Callable[[], None] | None = None,
) -> EvalResult:
    """Async version of :func:`run_eval` — runs cases with concurrency control.

    *pass_threshold* sets the minimum score for a case to be considered passing.

    *concurrency_config* controls parallelism, rate limiting, and retries.
    When ``None`` a default config is used (10 concurrent, no rate limit,
    3 retries).

    *on_case_complete* is called after each case finishes (for progress bars).
    """
    from prompttest.validation.prompt_validator import validate_dataset

    dataset = load_eval_dataset(dataset_path)

    # --- Validate before running ---
    validation = validate_dataset(prompt_config, dataset)
    if validation.errors and strict:
        raise validation.errors[0]

    scorer = get_scorer(dataset.scoring)
    provider = provider_override or get_provider(prompt_config.provider)
    cfg = concurrency_config or ConcurrencyConfig()

    def _make_task(case: EvalCase) -> Callable[[], Awaitable[EvalCaseResult]]:
        async def _run() -> EvalCaseResult:
            user_message = render_template_dict(prompt_config.template, case.input)
            output = await provider.acomplete(
                model=prompt_config.model,
                system=prompt_config.system,
                user_message=user_message,
                parameters=prompt_config.parameters,
            )
            return _score_case(case, output, scorer, pass_threshold)
        return _run

    # Run with concurrency control; catch per-task errors as ERROR verdicts
    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    limiter = RateLimiter(cfg.rate_limit)

    async def _guarded(case: EvalCase, fn: Callable[[], Awaitable[EvalCaseResult]]) -> EvalCaseResult:
        async with semaphore:
            await limiter.acquire()
            try:
                result = await run_with_retry(
                    fn,
                    max_retries=cfg.max_retries,
                    base_delay=cfg.base_delay,
                    max_delay=cfg.max_delay,
                )
                return result
            except Exception as exc:
                return _error_case(case, exc)
            finally:
                if on_case_complete is not None:
                    on_case_complete()

    case_results = await asyncio.gather(
        *[_guarded(case, _make_task(case)) for case in dataset.tests]
    )

    return EvalResult(
        prompt_name=prompt_config.name,
        prompt_version=prompt_config.version,
        scoring=dataset.scoring,
        case_results=list(case_results),
        pass_threshold=pass_threshold,
    )
