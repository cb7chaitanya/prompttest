"""LLM-powered failure explanation for eval results."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from prompttest.core.eval_runner import EvalCaseResult, EvalResult
from prompttest.core.models import Verdict


@dataclass
class FailureExplanation:
    """Explanation for a single failed test case."""

    index: int
    expected: str
    actual: str
    score: float
    explanation: str


def _build_prompt(expected: str, actual: str, score: float) -> str:
    return (
        f"A test case failed during LLM prompt evaluation.\n\n"
        f"Expected output:\n{expected}\n\n"
        f"Actual output:\n{actual}\n\n"
        f"Score: {score:.2f}\n\n"
        f"Explain concisely why the actual output does not match the expected output. "
        f"Focus on what is missing, incorrect, or different. Keep the explanation to 1-3 sentences."
    )


def explain_failure(
    expected: str,
    actual: str,
    score: float,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Ask an LLM to explain why actual output doesn't match expected.

    Uses OpenAI by default. Set ``PROMPTTEST_EXPLAIN_MODEL`` to override
    the model (default: ``gpt-4o-mini``).
    """
    try:
        import openai
    except ImportError:
        return "openai package not installed — pip install prompttest[openai]"

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return "OPENAI_API_KEY not set — cannot generate explanation"

    mdl = model or os.environ.get("PROMPTTEST_EXPLAIN_MODEL", "gpt-4o-mini")
    prompt = _build_prompt(expected, actual, score)

    try:
        client = openai.OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=mdl,
            messages=[
                {"role": "system", "content": "You are a concise test failure analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"Error generating explanation: {exc}"


def explain_failures(
    result: EvalResult,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> list[FailureExplanation]:
    """Generate explanations for all failed/errored cases in *result*."""
    explanations: list[FailureExplanation] = []

    for i, cr in enumerate(result.case_results, 1):
        if cr.verdict == Verdict.PASS:
            continue

        explanation = explain_failure(
            cr.case.expected,
            cr.output,
            cr.score,
            model=model,
            api_key=api_key,
        )
        explanations.append(FailureExplanation(
            index=i,
            expected=cr.case.expected,
            actual=cr.output,
            score=cr.score,
            explanation=explanation,
        ))

    return explanations
