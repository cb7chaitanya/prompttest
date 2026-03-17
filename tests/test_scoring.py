"""Tests for extended scoring system."""

from __future__ import annotations

from pathlib import Path

import pytest

from prompttest.core.eval_runner import EvalResult, run_eval
from prompttest.core.models import PromptConfig, Verdict
from prompttest.core.scoring import (
    PASS_THRESHOLD,
    fuzzy,
    get_scorer,
    list_scorers,
    regex,
    register_scorer,
)


# ---------------------------------------------------------------------------
# fuzzy scorer
# ---------------------------------------------------------------------------

class TestFuzzyScorer:
    def test_exact_match(self):
        score, reason = fuzzy("hello world", "hello world")
        assert score == 1.0
        assert "exact" in reason

    def test_close_match(self):
        score, _ = fuzzy("hello world", "hello worl")
        assert 0.8 < score < 1.0

    def test_completely_different(self):
        score, _ = fuzzy("abc", "xyz")
        assert score < 0.3

    def test_case_insensitive(self):
        score, _ = fuzzy("Hello World", "hello world")
        assert score == 1.0

    def test_whitespace_normalized(self):
        score, _ = fuzzy("  hello  ", "hello")
        assert score == 1.0

    def test_returns_float_between_0_and_1(self):
        score, _ = fuzzy("The quick brown fox", "The slow red fox")
        assert 0.0 <= score <= 1.0

    def test_partial_overlap(self):
        score, _ = fuzzy("Our refund policy is 30 days", "30 days refund policy")
        assert 0.3 < score < 1.0


# ---------------------------------------------------------------------------
# regex scorer
# ---------------------------------------------------------------------------

class TestRegexScorer:
    def test_match(self):
        score, _ = regex("The answer is 42.", r"\d+")
        assert score == 1.0

    def test_no_match(self):
        score, _ = regex("No numbers here", r"\d+")
        assert score == 0.0

    def test_case_insensitive(self):
        score, _ = regex("Hello World", r"hello")
        assert score == 1.0

    def test_complex_pattern(self):
        score, _ = regex("2026-03-17", r"\d{4}-\d{2}-\d{2}")
        assert score == 1.0

    def test_invalid_regex_returns_zero(self):
        score, reason = regex("test", r"[invalid")
        assert score == 0.0
        assert "invalid regex" in reason


# ---------------------------------------------------------------------------
# Registry includes new scorers
# ---------------------------------------------------------------------------

class TestScorerRegistry:
    def test_fuzzy_registered(self):
        assert "fuzzy" in list_scorers()
        assert get_scorer("fuzzy") is fuzzy

    def test_regex_registered(self):
        assert "regex" in list_scorers()
        assert get_scorer("regex") is regex

    def test_semantic_registered(self):
        assert "semantic" in list_scorers()

    def test_llm_judge_registered(self):
        assert "llm_judge" in list_scorers()

    def test_all_scorers(self):
        names = list_scorers()
        for expected in ["contains", "exact", "starts_with", "ends_with",
                         "regex", "fuzzy", "semantic", "llm_judge"]:
            assert expected in names


# ---------------------------------------------------------------------------
# Pass threshold
# ---------------------------------------------------------------------------

class TestPassThreshold:
    def test_default_threshold(self):
        assert PASS_THRESHOLD == 0.7

    def test_custom_threshold_in_eval(self, tmp_path: Path):
        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{input}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\n"
            "scoring: fuzzy\n"
            "tests:\n"
            "  - input: hello world\n"
            "    expected: hello worl\n"  # close but not exact
        )

        # With high threshold, fuzzy near-match should fail
        result_strict = run_eval(dataset_yaml, prompt, pass_threshold=1.0)
        assert result_strict.case_results[0].verdict == Verdict.FAIL

        # With lower threshold, same score should pass
        result_lenient = run_eval(dataset_yaml, prompt, pass_threshold=0.5)
        assert result_lenient.case_results[0].verdict == Verdict.PASS

    def test_threshold_stored_in_result(self, tmp_path: Path):
        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{input}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\n"
            "tests:\n"
            "  - input: hello\n"
            "    expected: hello\n"
        )
        result = run_eval(dataset_yaml, prompt, pass_threshold=0.85)
        assert result.pass_threshold == 0.85


# ---------------------------------------------------------------------------
# average_score
# ---------------------------------------------------------------------------

class TestAverageScore:
    def test_average_score(self, tmp_path: Path):
        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{input}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\n"
            "scoring: fuzzy\n"
            "tests:\n"
            "  - input: hello world\n"
            "    expected: hello world\n"
            "  - input: abc\n"
            "    expected: xyz\n"
        )
        result = run_eval(dataset_yaml, prompt)
        # First case: exact match = 1.0. Second: very different ~ 0.0
        assert 0.3 < result.average_score < 0.8

    def test_average_score_all_pass(self, tmp_path: Path):
        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{input}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\n"
            "scoring: exact\n"
            "tests:\n"
            "  - input: hello\n"
            "    expected: hello\n"
            "  - input: world\n"
            "    expected: world\n"
        )
        result = run_eval(dataset_yaml, prompt)
        assert result.average_score == 1.0

    def test_average_score_empty(self):
        result = EvalResult(
            prompt_name="x", prompt_version="1",
            scoring="exact", case_results=[],
        )
        assert result.average_score == 0.0


# ---------------------------------------------------------------------------
# Exporter includes new fields
# ---------------------------------------------------------------------------

class TestExporterNewFields:
    def test_json_includes_average_and_threshold(self, tmp_path: Path):
        import json
        from prompttest.core.exporter import export_json

        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{input}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\n"
            "scoring: exact\n"
            "tests:\n"
            "  - input: hello\n"
            "    expected: hello\n"
        )
        result = run_eval(dataset_yaml, prompt, pass_threshold=0.8)
        data = json.loads(export_json(result, prompt))
        assert "average_score" in data["summary"]
        assert "pass_threshold" in data["summary"]
        assert data["summary"]["average_score"] == 1.0
        assert data["summary"]["pass_threshold"] == 0.8


# ---------------------------------------------------------------------------
# Custom scorer registration still works
# ---------------------------------------------------------------------------

class TestCustomScorer:
    def test_register_float_scorer(self):
        def half_scorer(output: str, expected: str) -> tuple[float, str]:
            return 0.5, "always 50%"

        register_scorer("half", half_scorer)
        fn = get_scorer("half")
        score, _ = fn("anything", "anything")
        assert score == 0.5

    def test_custom_scorer_with_eval(self, tmp_path: Path):
        def len_ratio(output: str, expected: str) -> tuple[float, str]:
            if not expected:
                return 1.0 if not output else 0.0, "empty expected"
            ratio = min(len(output), len(expected)) / max(len(output), len(expected))
            return ratio, f"length ratio {ratio:.2f}"

        register_scorer("len_ratio", len_ratio)

        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{input}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\n"
            "scoring: len_ratio\n"
            "tests:\n"
            "  - input: hello\n"
            "    expected: hello\n"
            "  - input: hi\n"
            "    expected: hello world\n"
        )
        result = run_eval(dataset_yaml, prompt, pass_threshold=0.5)
        assert result.case_results[0].score == 1.0
        assert result.case_results[0].verdict == Verdict.PASS
        # "hi" (2 chars) vs "hello world" (11 chars) → ratio ~0.18
        assert result.case_results[1].score < 0.5
        assert result.case_results[1].verdict == Verdict.FAIL
