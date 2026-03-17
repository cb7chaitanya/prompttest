"""Tests for prompttest.core.explainer."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

from prompttest.core.eval_runner import EvalCase, EvalCaseResult, EvalResult
from prompttest.core.explainer import (
    FailureExplanation,
    _build_prompt,
    explain_failure,
    explain_failures,
)
from prompttest.core.models import Verdict


def _mock_openai():
    """Create a mock openai module with a mock client that returns a canned response."""
    mock_openai_mod = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked explanation."
    mock_openai_mod.OpenAI.return_value.chat.completions.create.return_value = mock_response
    return mock_openai_mod


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_expected_and_actual(self):
        prompt = _build_prompt("30 days", "Check our website", 0.0)
        assert "30 days" in prompt
        assert "Check our website" in prompt
        assert "0.00" in prompt

    def test_contains_instructions(self):
        prompt = _build_prompt("x", "y", 0.5)
        assert "missing" in prompt.lower() or "incorrect" in prompt.lower()


# ---------------------------------------------------------------------------
# explain_failure — no API key (with openai mocked as importable)
# ---------------------------------------------------------------------------

class TestExplainFailureNoKey:
    def test_missing_key_returns_message(self):
        mock_mod = MagicMock()
        with patch.dict(os.environ, {}, clear=True), \
             patch.dict(sys.modules, {"openai": mock_mod}):
            result = explain_failure("expected", "actual", 0.0)
            assert "OPENAI_API_KEY" in result

    def test_explicit_empty_key(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_mod}):
            result = explain_failure("expected", "actual", 0.0, api_key="")
            assert "OPENAI_API_KEY" in result


# ---------------------------------------------------------------------------
# explain_failure — mocked API
# ---------------------------------------------------------------------------

class TestExplainFailureMocked:
    def test_returns_explanation(self):
        mock_mod = _mock_openai()
        with patch.dict(sys.modules, {"openai": mock_mod}):
            result = explain_failure("30 days", "Check our website", 0.0, api_key="sk-test")
        assert result == "Mocked explanation."

    def test_api_error_returns_error_message(self):
        mock_mod = MagicMock()
        mock_mod.OpenAI.return_value.chat.completions.create.side_effect = Exception("API down")
        with patch.dict(sys.modules, {"openai": mock_mod}):
            result = explain_failure("x", "y", 0.0, api_key="sk-test")
        assert "Error generating explanation" in result
        assert "API down" in result

    def test_custom_model(self):
        mock_mod = _mock_openai()
        with patch.dict(sys.modules, {"openai": mock_mod}):
            explain_failure("x", "y", 0.0, api_key="sk-test", model="gpt-4o")
        call_kwargs = mock_mod.OpenAI.return_value.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# explain_failures — filters only failed cases
# ---------------------------------------------------------------------------

class TestExplainFailures:
    def _make_result(self) -> EvalResult:
        return EvalResult(
            prompt_name="test", prompt_version="1", scoring="contains",
            case_results=[
                EvalCaseResult(
                    case=EvalCase(input={"q": "a"}, expected="pass this"),
                    output="pass this", verdict=Verdict.PASS, score=1.0, reason="ok",
                ),
                EvalCaseResult(
                    case=EvalCase(input={"q": "b"}, expected="30 days"),
                    output="Check our website", verdict=Verdict.FAIL, score=0.0, reason="no match",
                ),
                EvalCaseResult(
                    case=EvalCase(input={"q": "c"}, expected="yes"),
                    output="error", verdict=Verdict.ERROR, score=0.0, reason="provider error",
                ),
            ],
            pass_threshold=0.7,
        )

    def test_only_explains_failures(self):
        mock_mod = _mock_openai()
        with patch.dict(sys.modules, {"openai": mock_mod}):
            explanations = explain_failures(self._make_result(), api_key="sk-test")

        # Should explain case #2 (fail) and #3 (error), skip #1 (pass)
        assert len(explanations) == 2
        assert explanations[0].index == 2
        assert explanations[0].expected == "30 days"
        assert explanations[0].actual == "Check our website"
        assert explanations[1].index == 3

    def test_no_failures_returns_empty(self):
        result = EvalResult(
            prompt_name="t", prompt_version="1", scoring="exact",
            case_results=[
                EvalCaseResult(
                    case=EvalCase(input={"q": "a"}, expected="a"),
                    output="a", verdict=Verdict.PASS, score=1.0, reason="ok",
                ),
            ],
            pass_threshold=0.7,
        )
        explanations = explain_failures(result, api_key="sk-test")
        assert explanations == []


# ---------------------------------------------------------------------------
# FailureExplanation
# ---------------------------------------------------------------------------

class TestFailureExplanation:
    def test_fields(self):
        exp = FailureExplanation(
            index=3, expected="30 days", actual="Check website",
            score=0.0, explanation="Missing refund duration.",
        )
        assert exp.index == 3
        assert exp.explanation == "Missing refund duration."
