"""Evaluate LLM outputs against expected results."""

from __future__ import annotations

from prompttest.core.models import CaseResult, TestCase, Verdict


def exact_match(output: str, expected: str) -> tuple[float, str]:
    """Score 1.0 if normalized strings match exactly."""
    norm_out = output.strip().lower()
    norm_exp = expected.strip().lower()
    if norm_out == norm_exp:
        return 1.0, "exact match"
    return 0.0, f"expected '{expected}', got '{output}'"


def contains_match(output: str, expected: str) -> tuple[float, str]:
    """Score 1.0 if expected text appears anywhere in output."""
    if expected.strip().lower() in output.strip().lower():
        return 1.0, "contains expected text"
    return 0.0, f"output does not contain '{expected}'"


DEFAULT_EVALUATOR = contains_match
PASS_THRESHOLD = 1.0


def evaluate_case(case: TestCase, output: str) -> CaseResult:
    score, reason = DEFAULT_EVALUATOR(output, case.expected)
    verdict = Verdict.PASS if score >= PASS_THRESHOLD else Verdict.FAIL
    return CaseResult(case=case, output=output, verdict=verdict, score=score, reason=reason)
