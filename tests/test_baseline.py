"""Tests for prompttest.core.baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompttest.core.baseline import (
    BaselineComparison,
    CaseDiff,
    compare,
    load_baseline,
)
from prompttest.core.eval_runner import EvalCase, EvalCaseResult, EvalResult
from prompttest.core.models import Verdict


def _make_baseline_json(
    avg_score: float = 0.86,
    accuracy: float = 0.8,
    cases: list[dict] | None = None,
) -> dict:
    if cases is None:
        cases = [
            {"input": {"q": "a"}, "expected": "a", "actual": "a", "score": 1.0,
             "passed": True, "verdict": "pass", "reason": "ok"},
            {"input": {"q": "b"}, "expected": "b", "actual": "x", "score": 0.5,
             "passed": False, "verdict": "fail", "reason": "partial"},
        ]
    return {
        "metadata": {"timestamp": "2026-01-01T00:00:00Z", "prompt_name": "test",
                      "prompt_version": "1", "model": "echo", "provider": "echo",
                      "scorer": "fuzzy"},
        "summary": {"total": len(cases), "passed": sum(1 for c in cases if c["passed"]),
                     "failed": sum(1 for c in cases if not c["passed"]),
                     "errors": 0, "accuracy": accuracy,
                     "average_score": avg_score, "pass_threshold": 0.7},
        "results": cases,
    }


def _make_eval_result(scores: list[float]) -> EvalResult:
    case_results = []
    for s in scores:
        case_results.append(EvalCaseResult(
            case=EvalCase(input={"q": "x"}, expected="x"),
            output="x",
            verdict=Verdict.PASS if s >= 0.7 else Verdict.FAIL,
            score=s,
            reason="test",
        ))
    return EvalResult(
        prompt_name="test",
        prompt_version="1",
        scoring="fuzzy",
        case_results=case_results,
        pass_threshold=0.7,
    )


# ---------------------------------------------------------------------------
# load_baseline
# ---------------------------------------------------------------------------

class TestLoadBaseline:
    def test_valid_file(self, tmp_path: Path):
        p = tmp_path / "baseline.json"
        p.write_text(json.dumps(_make_baseline_json()))
        data = load_baseline(p)
        assert "summary" in data
        assert "results" in data

    def test_invalid_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            load_baseline(p)

    def test_missing_keys(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"foo": "bar"}))
        with pytest.raises(ValueError, match="Invalid baseline file"):
            load_baseline(p)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_no_regression(self):
        baseline = _make_baseline_json(avg_score=0.75)
        current = _make_eval_result([1.0, 0.8])  # avg = 0.9
        cmp = compare(baseline, current)
        assert cmp.baseline_avg == 0.75
        assert cmp.current_avg == 0.9
        assert cmp.avg_delta == pytest.approx(0.15)
        assert not cmp.has_regression

    def test_regression_detected(self):
        baseline = _make_baseline_json(avg_score=0.9)
        current = _make_eval_result([0.5, 0.3])  # avg = 0.4
        cmp = compare(baseline, current)
        assert cmp.avg_delta < 0
        assert cmp.has_regression

    def test_per_case_diffs(self):
        baseline = _make_baseline_json(
            avg_score=0.75,
            cases=[
                {"input": {"q": "a"}, "expected": "a", "actual": "a", "score": 1.0,
                 "passed": True, "verdict": "pass", "reason": "ok"},
                {"input": {"q": "b"}, "expected": "b", "actual": "x", "score": 0.5,
                 "passed": False, "verdict": "fail", "reason": "partial"},
            ],
        )
        current = _make_eval_result([0.8, 0.9])  # case 1 regressed, case 2 improved
        cmp = compare(baseline, current)
        assert len(cmp.case_diffs) == 2
        assert cmp.case_diffs[0].delta == pytest.approx(-0.2)  # 0.8 - 1.0
        assert cmp.case_diffs[0].regressed
        assert cmp.case_diffs[1].delta == pytest.approx(0.4)   # 0.9 - 0.5
        assert not cmp.case_diffs[1].regressed

    def test_regressions_list(self):
        baseline = _make_baseline_json(
            avg_score=0.9,
            cases=[
                {"input": {"q": "a"}, "expected": "a", "actual": "a", "score": 1.0,
                 "passed": True, "verdict": "pass", "reason": "ok"},
            ],
        )
        current = _make_eval_result([0.5])
        cmp = compare(baseline, current)
        assert len(cmp.regressions) == 1
        assert cmp.regressions[0].index == 1

    def test_improvements_list(self):
        baseline = _make_baseline_json(
            avg_score=0.3,
            cases=[
                {"input": {"q": "a"}, "expected": "a", "actual": "x", "score": 0.3,
                 "passed": False, "verdict": "fail", "reason": "low"},
            ],
        )
        current = _make_eval_result([0.9])
        cmp = compare(baseline, current)
        assert len(cmp.improvements) == 1

    def test_pass_rate_comparison(self):
        baseline = _make_baseline_json(accuracy=0.5, avg_score=0.5)
        current = _make_eval_result([1.0, 1.0])  # 100% pass rate
        cmp = compare(baseline, current)
        assert cmp.baseline_pass_rate == 0.5
        assert cmp.current_pass_rate == 1.0
        assert cmp.pass_rate_delta == pytest.approx(0.5)

    def test_more_current_cases_than_baseline(self):
        baseline = _make_baseline_json(
            avg_score=1.0,
            cases=[
                {"input": {"q": "a"}, "expected": "a", "actual": "a", "score": 1.0,
                 "passed": True, "verdict": "pass", "reason": "ok"},
            ],
        )
        current = _make_eval_result([1.0, 0.5])  # extra case
        cmp = compare(baseline, current)
        assert len(cmp.case_diffs) == 2
        # Second case has no baseline → baseline_score=0.0
        assert cmp.case_diffs[1].baseline_score == 0.0

    def test_equal_scores_no_regression(self):
        baseline = _make_baseline_json(
            avg_score=0.8,
            cases=[
                {"input": {"q": "a"}, "expected": "a", "actual": "a", "score": 0.8,
                 "passed": True, "verdict": "pass", "reason": "ok"},
                {"input": {"q": "b"}, "expected": "b", "actual": "b", "score": 0.8,
                 "passed": True, "verdict": "pass", "reason": "ok"},
            ],
        )
        current = _make_eval_result([0.8, 0.8])
        cmp = compare(baseline, current)
        assert cmp.avg_delta == pytest.approx(0.0)
        assert not cmp.has_regression


# ---------------------------------------------------------------------------
# CaseDiff
# ---------------------------------------------------------------------------

class TestCaseDiff:
    def test_delta(self):
        d = CaseDiff(index=1, input_summary="q", expected="a",
                      baseline_score=0.8, current_score=0.6)
        assert d.delta == pytest.approx(-0.2)
        assert d.regressed

    def test_no_regression(self):
        d = CaseDiff(index=1, input_summary="q", expected="a",
                      baseline_score=0.5, current_score=0.9)
        assert not d.regressed


# ---------------------------------------------------------------------------
# Integration: export → load → compare round-trip
# ---------------------------------------------------------------------------

class TestExportBaselineRoundTrip:
    def test_round_trip(self, tmp_path: Path):
        from prompttest.core.eval_runner import run_eval
        from prompttest.core.exporter import save_result
        from prompttest.core.models import PromptConfig

        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{input}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\nscoring: contains\ntests:\n"
            "  - input: hello\n    expected: hello\n"
            "  - input: world\n    expected: world\n"
        )

        # Run eval and save as baseline
        result1 = run_eval(dataset_yaml, prompt)
        baseline_path = tmp_path / "baseline.json"
        save_result(result1, prompt, baseline_path, "json")

        # Load baseline and compare against same result
        baseline_data = load_baseline(baseline_path)
        cmp = compare(baseline_data, result1)
        assert cmp.avg_delta == pytest.approx(0.0)
        assert not cmp.has_regression
        assert len(cmp.regressions) == 0
