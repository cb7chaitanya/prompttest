"""Pipeline evaluation runner: evaluate any EvalTarget against a dataset."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from prompttest.core.concurrency import ConcurrencyConfig, RateLimiter, run_with_retry
from prompttest.core.eval_runner import (
    EvalCase,
    EvalCaseResult,
    EvalDataset,
    EvalResult,
    _error_case,
    _score_case,
    filter_by_tags,
    load_eval_dataset,
)
from prompttest.core.models import Verdict
from prompttest.core.scoring import PASS_THRESHOLD, get_scorer
from prompttest.pipeline.targets import EvalTarget


def evaluate(
    target: EvalTarget,
    dataset: EvalDataset | Path | str,
    *,
    scorer_name: str = "",
    pass_threshold: float = PASS_THRESHOLD,
    tags: list[str] | None = None,
    tag_match: str = "any",
) -> EvalResult:
    """Evaluate *target* against *dataset* synchronously.

    *dataset* can be an ``EvalDataset`` instance, a ``Path``, or a path string.
    """
    ds = _resolve_dataset(dataset)

    if scorer_name:
        ds.scoring = scorer_name

    if tags:
        filter_by_tags(ds, tags, tag_match)

    scorer = get_scorer(ds.scoring)
    case_results: list[EvalCaseResult] = []

    for case in ds.tests:
        try:
            output = target.call(case.input)
            case_results.append(_score_case(case, output, scorer, pass_threshold))
        except Exception as exc:
            case_results.append(_error_case(case, exc))

    return EvalResult(
        prompt_name=target.name,
        prompt_version=target.version,
        scoring=ds.scoring,
        case_results=case_results,
        pass_threshold=pass_threshold,
    )


async def evaluate_async(
    target: EvalTarget,
    dataset: EvalDataset | Path | str,
    *,
    scorer_name: str = "",
    pass_threshold: float = PASS_THRESHOLD,
    tags: list[str] | None = None,
    tag_match: str = "any",
    concurrency_config: ConcurrencyConfig | None = None,
    on_case_complete: Callable[[], None] | None = None,
) -> EvalResult:
    """Evaluate *target* against *dataset* asynchronously with concurrency control."""
    ds = _resolve_dataset(dataset)

    if scorer_name:
        ds.scoring = scorer_name

    if tags:
        filter_by_tags(ds, tags, tag_match)

    scorer = get_scorer(ds.scoring)
    cfg = concurrency_config or ConcurrencyConfig()

    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    limiter = RateLimiter(cfg.rate_limit)

    async def _guarded(case: EvalCase) -> EvalCaseResult:
        async with semaphore:
            await limiter.acquire()
            try:
                async def _run() -> EvalCaseResult:
                    output = await target.acall(case.input)
                    return _score_case(case, output, scorer, pass_threshold)

                result = await run_with_retry(
                    _run,
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

    case_results = await asyncio.gather(*[_guarded(c) for c in ds.tests])

    return EvalResult(
        prompt_name=target.name,
        prompt_version=target.version,
        scoring=ds.scoring,
        case_results=list(case_results),
        pass_threshold=pass_threshold,
    )


def _resolve_dataset(dataset: EvalDataset | Path | str) -> EvalDataset:
    if isinstance(dataset, EvalDataset):
        return dataset
    return load_eval_dataset(Path(dataset))
