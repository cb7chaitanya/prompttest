"""Tests for prompttest.core.report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from prompttest.core.eval_runner import EvalCase, EvalCaseResult, EvalResult
from prompttest.core.models import PromptConfig, Verdict
from prompttest.core.report import export_html, save_html_report

FIXED_TS = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)


def _make_prompt() -> PromptConfig:
    return PromptConfig(
        name="support", version="2", model="gpt-4o", provider="openai",
        system="You help users.", template="Q: {{question}}",
    )


def _make_result(scores: list[float] | None = None) -> EvalResult:
    if scores is None:
        scores = [1.0, 0.7, 0.3, 0.0]
    cases = []
    for i, s in enumerate(scores):
        cases.append(EvalCaseResult(
            case=EvalCase(input={"question": f"q{i}"}, expected=f"a{i}"),
            output=f"out{i}",
            verdict=Verdict.PASS if s >= 0.7 else Verdict.FAIL,
            score=s,
            reason=f"reason {i}",
        ))
    return EvalResult(
        prompt_name="support", prompt_version="2", scoring="fuzzy",
        case_results=cases, pass_threshold=0.7,
    )


# ---------------------------------------------------------------------------
# export_html
# ---------------------------------------------------------------------------

class TestExportHtml:
    def test_returns_valid_html(self):
        html = export_html(_make_result(), _make_prompt(), timestamp=FIXED_TS)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_title(self):
        html = export_html(_make_result(), _make_prompt(), timestamp=FIXED_TS)
        assert "support" in html
        assert "v2" in html

    def test_contains_summary_metrics(self):
        result = _make_result([1.0, 0.8, 0.5, 0.0])
        html = export_html(result, _make_prompt(), timestamp=FIXED_TS)
        assert ">4<" in html  # total
        assert "Passed" in html
        assert "Failed" in html
        assert "Avg Score" in html
        assert "Threshold" in html

    def test_contains_per_case_rows(self):
        html = export_html(_make_result(), _make_prompt(), timestamp=FIXED_TS)
        assert "q0" in html
        assert "q1" in html
        assert "q2" in html
        assert "q3" in html

    def test_pass_fail_badges(self):
        html = export_html(_make_result(), _make_prompt(), timestamp=FIXED_TS)
        assert "badge-pass" in html
        assert "badge-fail" in html

    def test_metadata_footer(self):
        html = export_html(_make_result(), _make_prompt(), timestamp=FIXED_TS)
        assert "gpt-4o" in html
        assert "openai" in html
        assert "fuzzy" in html

    def test_score_distribution_chart(self):
        html = export_html(_make_result(), _make_prompt(), timestamp=FIXED_TS)
        assert "Score Distribution" in html
        assert "bar-fill" in html

    def test_no_errors_card_when_zero(self):
        result = _make_result([1.0, 0.5])
        html = export_html(result, _make_prompt(), timestamp=FIXED_TS)
        assert "Errors" not in html

    def test_errors_card_shown(self):
        cases = [
            EvalCaseResult(
                case=EvalCase(input={"q": "x"}, expected="y"),
                output="err", verdict=Verdict.ERROR, score=0.0, reason="boom",
            ),
        ]
        result = EvalResult(
            prompt_name="t", prompt_version="1", scoring="exact",
            case_results=cases,
        )
        html = export_html(result, _make_prompt(), timestamp=FIXED_TS)
        assert "Errors" in html

    def test_html_escaping(self):
        cases = [
            EvalCaseResult(
                case=EvalCase(input={"q": "<script>alert(1)</script>"}, expected="safe"),
                output="<b>bad</b>", verdict=Verdict.FAIL, score=0.0, reason="xss",
            ),
        ]
        result = EvalResult(
            prompt_name="t", prompt_version="1", scoring="exact",
            case_results=cases,
        )
        html = export_html(result, _make_prompt(), timestamp=FIXED_TS)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# save_html_report
# ---------------------------------------------------------------------------

class TestSaveHtmlReport:
    def test_saves_file(self, tmp_path: Path):
        dest = tmp_path / "report.html"
        saved = save_html_report(_make_result(), _make_prompt(), dest, timestamp=FIXED_TS)
        assert saved.exists()
        content = saved.read_text()
        assert "<!DOCTYPE html>" in content

    def test_creates_parent_dirs(self, tmp_path: Path):
        dest = tmp_path / "sub" / "dir" / "report.html"
        saved = save_html_report(_make_result(), _make_prompt(), dest, timestamp=FIXED_TS)
        assert saved.exists()


# ---------------------------------------------------------------------------
# Integration: eval → HTML report round-trip
# ---------------------------------------------------------------------------

class TestEvalReportRoundTrip:
    def test_eval_then_report(self, tmp_path: Path):
        from prompttest.core.eval_runner import run_eval

        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="Answer: {{question}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\nscoring: contains\ntests:\n"
            "  - input:\n      question: hello\n    expected: hello\n"
            "  - input:\n      question: world\n    expected: NOPE\n"
        )

        result = run_eval(dataset_yaml, prompt)
        dest = tmp_path / "report.html"
        save_html_report(result, prompt, dest, timestamp=FIXED_TS)

        content = dest.read_text()
        assert "hello" in content
        assert "PASS" in content
        assert "FAIL" in content
        assert "Avg Score" in content
