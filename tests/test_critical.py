"""Tests for critical (golden) test case support."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompttest.core.eval_runner import (
    EvalCase,
    EvalCaseResult,
    EvalDataset,
    EvalResult,
    load_eval_dataset,
    run_eval,
)
from prompttest.core.models import PromptConfig, Verdict


# ---------------------------------------------------------------------------
# EvalCase critical field
# ---------------------------------------------------------------------------

class TestEvalCaseCritical:
    def test_default_false(self):
        case = EvalCase(input={"q": "x"}, expected="y")
        assert case.critical is False

    def test_set_true(self):
        case = EvalCase(input={"q": "x"}, expected="y", critical=True)
        assert case.critical is True


# ---------------------------------------------------------------------------
# EvalDataset parsing
# ---------------------------------------------------------------------------

class TestDatasetParseCritical:
    def test_parses_critical_field(self, tmp_path: Path):
        yaml = (
            "prompt: test\n"
            "tests:\n"
            "  - input:\n"
            "      q: hello\n"
            "    expected: hello\n"
            "    critical: true\n"
            "  - input:\n"
            "      q: world\n"
            "    expected: world\n"
        )
        p = tmp_path / "ds.yaml"
        p.write_text(yaml)
        ds = load_eval_dataset(p)
        assert ds.tests[0].critical is True
        assert ds.tests[1].critical is False

    def test_critical_missing_defaults_false(self, tmp_path: Path):
        yaml = (
            "prompt: test\n"
            "tests:\n"
            "  - input:\n"
            "      q: hello\n"
            "    expected: hello\n"
        )
        p = tmp_path / "ds.yaml"
        p.write_text(yaml)
        ds = load_eval_dataset(p)
        assert ds.tests[0].critical is False


# ---------------------------------------------------------------------------
# EvalResult critical properties
# ---------------------------------------------------------------------------

class TestEvalResultCritical:
    def _make_result(self, cases):
        return EvalResult(
            prompt_name="test", prompt_version="1", scoring="contains",
            case_results=cases, pass_threshold=0.7,
        )

    def test_critical_total(self):
        cases = [
            EvalCaseResult(
                case=EvalCase(input={"q": "a"}, expected="a", critical=True),
                output="a", verdict=Verdict.PASS, score=1.0, reason="ok",
            ),
            EvalCaseResult(
                case=EvalCase(input={"q": "b"}, expected="b", critical=True),
                output="x", verdict=Verdict.FAIL, score=0.0, reason="no",
            ),
            EvalCaseResult(
                case=EvalCase(input={"q": "c"}, expected="c"),
                output="x", verdict=Verdict.FAIL, score=0.0, reason="no",
            ),
        ]
        result = self._make_result(cases)
        assert result.critical_total == 2
        assert result.critical_failed == 1

    def test_no_critical(self):
        cases = [
            EvalCaseResult(
                case=EvalCase(input={"q": "a"}, expected="a"),
                output="x", verdict=Verdict.FAIL, score=0.0, reason="no",
            ),
        ]
        result = self._make_result(cases)
        assert result.critical_total == 0
        assert result.critical_failed == 0

    def test_all_critical_pass(self):
        cases = [
            EvalCaseResult(
                case=EvalCase(input={"q": "a"}, expected="a", critical=True),
                output="a", verdict=Verdict.PASS, score=1.0, reason="ok",
            ),
        ]
        result = self._make_result(cases)
        assert result.critical_total == 1
        assert result.critical_failed == 0

    def test_critical_error_counts_as_failed(self):
        cases = [
            EvalCaseResult(
                case=EvalCase(input={"q": "a"}, expected="a", critical=True),
                output="err", verdict=Verdict.ERROR, score=0.0, reason="boom",
            ),
        ]
        result = self._make_result(cases)
        assert result.critical_failed == 1


# ---------------------------------------------------------------------------
# Integration: eval with critical cases
# ---------------------------------------------------------------------------

class TestEvalWithCritical:
    def test_critical_tracked_through_eval(self, tmp_path: Path):
        yaml = (
            "prompt: test\n"
            "scoring: contains\n"
            "tests:\n"
            "  - input:\n"
            "      q: hello\n"
            "    expected: hello\n"
            "    critical: true\n"
            "  - input:\n"
            "      q: world\n"
            "    expected: NOPE\n"
            "    critical: true\n"
            "  - input:\n"
            "      q: foo\n"
            "    expected: NOPE\n"
        )
        p = tmp_path / "ds.yaml"
        p.write_text(yaml)

        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{q}}",
        )
        result = run_eval(p, prompt)
        assert result.critical_total == 2
        assert result.critical_failed == 1  # "world" doesn't contain "NOPE"
        assert result.failed == 2  # both non-matching cases fail


# ---------------------------------------------------------------------------
# Exporter includes critical
# ---------------------------------------------------------------------------

class TestExporterCritical:
    def test_json_includes_critical(self, tmp_path: Path):
        from prompttest.core.exporter import export_json

        cases = [
            EvalCaseResult(
                case=EvalCase(input={"q": "a"}, expected="a", critical=True),
                output="a", verdict=Verdict.PASS, score=1.0, reason="ok",
            ),
            EvalCaseResult(
                case=EvalCase(input={"q": "b"}, expected="b"),
                output="x", verdict=Verdict.FAIL, score=0.0, reason="no",
            ),
        ]
        result = EvalResult(
            prompt_name="test", prompt_version="1", scoring="exact",
            case_results=cases, pass_threshold=0.7,
        )
        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{q}}",
        )
        data = json.loads(export_json(result, prompt))
        assert data["results"][0]["critical"] is True
        assert data["results"][1]["critical"] is False
        assert data["summary"]["critical_total"] == 1
        assert data["summary"]["critical_failed"] == 0


# ---------------------------------------------------------------------------
# HTML report highlights critical
# ---------------------------------------------------------------------------

class TestReportCritical:
    def test_critical_badge_in_report(self):
        from prompttest.core.report import export_html

        cases = [
            EvalCaseResult(
                case=EvalCase(input={"q": "a"}, expected="a", critical=True),
                output="x", verdict=Verdict.FAIL, score=0.0, reason="no",
            ),
        ]
        result = EvalResult(
            prompt_name="test", prompt_version="1", scoring="exact",
            case_results=cases,
        )
        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{q}}",
        )
        html = export_html(result, prompt)
        assert "CRITICAL" in html
        assert "critical-fail" in html

    def test_no_critical_fail_row_when_passing(self):
        from prompttest.core.report import export_html

        cases = [
            EvalCaseResult(
                case=EvalCase(input={"q": "a"}, expected="a", critical=True),
                output="a", verdict=Verdict.PASS, score=1.0, reason="ok",
            ),
        ]
        result = EvalResult(
            prompt_name="test", prompt_version="1", scoring="exact",
            case_results=cases,
        )
        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{q}}",
        )
        html = export_html(result, prompt)
        # CSS class definition exists, but no row should use it
        assert 'class="critical-fail"' not in html
