"""Tests for prompttest.core.exporter."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prompttest.core.eval_runner import EvalCase, EvalCaseResult, EvalResult
from prompttest.core.exporter import (
    auto_filename,
    export_csv,
    export_diff_json,
    export_json,
    save_result,
)
from prompttest.core.models import PromptConfig, Verdict

FIXED_TS = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)


def _make_prompt() -> PromptConfig:
    return PromptConfig(
        name="support",
        version="2",
        model="gpt-4o",
        provider="openai",
        system="You help users.",
        template="Q: {{question}}",
    )


def _make_result() -> EvalResult:
    cases = [
        EvalCaseResult(
            case=EvalCase(input={"question": "Refund?"}, expected="30 days"),
            output="Our refund policy is 30 days.",
            verdict=Verdict.PASS,
            score=1.0,
            reason="contains match",
        ),
        EvalCaseResult(
            case=EvalCase(input={"question": "Hours?"}, expected="9-5"),
            output="We are open 24/7.",
            verdict=Verdict.FAIL,
            score=0.0,
            reason="expected not found",
        ),
    ]
    return EvalResult(
        prompt_name="support",
        prompt_version="2",
        scoring="contains",
        case_results=cases,
    )


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_structure(self):
        data = json.loads(export_json(_make_result(), _make_prompt(), timestamp=FIXED_TS))

        assert data["metadata"]["timestamp"] == "2026-03-17T12:00:00+00:00"
        assert data["metadata"]["prompt_name"] == "support"
        assert data["metadata"]["prompt_version"] == "2"
        assert data["metadata"]["model"] == "gpt-4o"
        assert data["metadata"]["provider"] == "openai"
        assert data["metadata"]["scorer"] == "contains"

        assert data["summary"]["total"] == 2
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1
        assert data["summary"]["errors"] == 0
        assert data["summary"]["accuracy"] == 0.5

        assert len(data["results"]) == 2
        r0 = data["results"][0]
        assert r0["input"] == {"question": "Refund?"}
        assert r0["expected"] == "30 days"
        assert r0["actual"] == "Our refund policy is 30 days."
        assert r0["score"] == 1.0
        assert r0["passed"] is True
        assert r0["verdict"] == "pass"

    def test_is_valid_json(self):
        text = export_json(_make_result(), _make_prompt(), timestamp=FIXED_TS)
        json.loads(text)  # should not raise


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------

class TestExportCsv:
    def test_structure(self):
        text = export_csv(_make_result(), _make_prompt(), timestamp=FIXED_TS)
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 2

        assert rows[0]["expected"] == "30 days"
        assert rows[0]["passed"] == "True"
        assert rows[0]["score"] == "1.0"

        assert rows[1]["passed"] == "False"

    def test_header_columns(self):
        text = export_csv(_make_result(), _make_prompt(), timestamp=FIXED_TS)
        header = text.splitlines()[0]
        for col in ["index", "input", "expected", "actual", "score", "passed", "verdict", "reason"]:
            assert col in header


# ---------------------------------------------------------------------------
# export_diff_json
# ---------------------------------------------------------------------------

class TestExportDiffJson:
    def test_structure(self):
        diff = "--- a\n+++ b\n@@ ...\n-old\n+new"
        text = export_diff_json(diff, "bot", "v1", "v2", timestamp=FIXED_TS)
        data = json.loads(text)
        assert data["metadata"]["prompt_name"] == "bot"
        assert data["metadata"]["version_a"] == "v1"
        assert data["metadata"]["version_b"] == "v2"
        assert data["diff"] == diff


# ---------------------------------------------------------------------------
# save_result
# ---------------------------------------------------------------------------

class TestSaveResult:
    def test_save_json(self, tmp_path: Path):
        dest = tmp_path / "out.json"
        saved = save_result(_make_result(), _make_prompt(), dest, "json", timestamp=FIXED_TS)
        assert saved.exists()
        data = json.loads(saved.read_text())
        assert data["summary"]["total"] == 2

    def test_save_csv(self, tmp_path: Path):
        dest = tmp_path / "out.csv"
        saved = save_result(_make_result(), _make_prompt(), dest, "csv", timestamp=FIXED_TS)
        assert saved.exists()
        lines = saved.read_text().strip().splitlines()
        assert len(lines) == 3  # header + 2 rows

    def test_creates_parent_dirs(self, tmp_path: Path):
        dest = tmp_path / "sub" / "dir" / "out.json"
        saved = save_result(_make_result(), _make_prompt(), dest, "json", timestamp=FIXED_TS)
        assert saved.exists()

    def test_unknown_format_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown format"):
            save_result(_make_result(), _make_prompt(), tmp_path / "x.txt", "txt", timestamp=FIXED_TS)


# ---------------------------------------------------------------------------
# auto_filename
# ---------------------------------------------------------------------------

class TestAutoFilename:
    def test_json(self):
        name = auto_filename(_make_prompt(), "json", timestamp=FIXED_TS)
        assert name == "support_v2_20260317T120000Z.json"

    def test_csv(self):
        name = auto_filename(_make_prompt(), "csv", timestamp=FIXED_TS)
        assert name == "support_v2_20260317T120000Z.csv"


# ---------------------------------------------------------------------------
# Integration: end-to-end eval → export round-trip
# ---------------------------------------------------------------------------

class TestEvalExportRoundTrip:
    def test_eval_then_export_json(self, tmp_path: Path):
        from prompttest.core.eval_runner import run_eval

        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="Answer: {{question}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\n"
            "scoring: contains\n"
            "tests:\n"
            "  - input:\n"
            "      question: hello\n"
            "    expected: hello\n"
        )

        result = run_eval(dataset_yaml, prompt)
        dest = tmp_path / "results.json"
        save_result(result, prompt, dest, "json", timestamp=FIXED_TS)

        data = json.loads(dest.read_text())
        assert data["summary"]["passed"] == 1
        assert data["metadata"]["model"] == "echo"
        assert data["results"][0]["passed"] is True
