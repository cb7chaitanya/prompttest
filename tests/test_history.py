"""Tests for prompttest.core.history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prompttest.core.eval_runner import EvalCase, EvalCaseResult, EvalResult
from prompttest.core.history import (
    HistoryEntry,
    detect_trend,
    load_history,
    record,
)
from prompttest.core.models import PromptConfig, Verdict


def _make_prompt() -> PromptConfig:
    return PromptConfig(
        name="support", version="2", model="gpt-4o", provider="openai",
        system="", template="Q: {{q}}",
    )


def _make_result(avg: float = 0.8) -> EvalResult:
    # Build a result with a single case at the given score
    cases = [EvalCaseResult(
        case=EvalCase(input={"q": "x"}, expected="x"),
        output="x", verdict=Verdict.PASS if avg >= 0.7 else Verdict.FAIL,
        score=avg, reason="test",
    )]
    return EvalResult(
        prompt_name="support", prompt_version="2", scoring="contains",
        case_results=cases, pass_threshold=0.7,
    )


TS1 = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
TS2 = datetime(2026, 3, 16, 10, 0, 0, tzinfo=timezone.utc)
TS3 = datetime(2026, 3, 17, 10, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

class TestRecord:
    def test_creates_history_file(self, tmp_path: Path):
        root = tmp_path / ".prompttest"
        root.mkdir()
        entry = record(root, _make_result(), _make_prompt(), timestamp=TS1)
        assert entry.prompt_name == "support"
        assert entry.average_score == 0.8
        hist_file = root / "history" / "runs.jsonl"
        assert hist_file.exists()

    def test_appends_entries(self, tmp_path: Path):
        root = tmp_path / ".prompttest"
        root.mkdir()
        record(root, _make_result(0.8), _make_prompt(), timestamp=TS1)
        record(root, _make_result(0.9), _make_prompt(), timestamp=TS2)
        lines = (root / "history" / "runs.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_entry_fields(self, tmp_path: Path):
        root = tmp_path / ".prompttest"
        root.mkdir()
        entry = record(root, _make_result(0.75), _make_prompt(), timestamp=TS1)
        assert entry.timestamp == TS1.isoformat()
        assert entry.model == "gpt-4o"
        assert entry.provider == "openai"
        assert entry.scorer == "contains"
        assert entry.total == 1
        assert entry.pass_threshold == 0.7


# ---------------------------------------------------------------------------
# load_history
# ---------------------------------------------------------------------------

class TestLoadHistory:
    def test_load_all(self, tmp_path: Path):
        root = tmp_path / ".prompttest"
        root.mkdir()
        record(root, _make_result(0.8), _make_prompt(), timestamp=TS1)
        record(root, _make_result(0.9), _make_prompt(), timestamp=TS2)
        entries = load_history(root)
        assert len(entries) == 2
        assert entries[0].average_score == 0.8
        assert entries[1].average_score == 0.9

    def test_filter_by_prompt_name(self, tmp_path: Path):
        root = tmp_path / ".prompttest"
        root.mkdir()
        record(root, _make_result(0.8), _make_prompt(), timestamp=TS1)
        # Record a different prompt
        other_prompt = PromptConfig(
            name="other", version="1", model="echo", provider="echo",
            system="", template="{{q}}",
        )
        other_result = EvalResult(
            prompt_name="other", prompt_version="1", scoring="exact",
            case_results=[], pass_threshold=0.7,
        )
        record(root, other_result, other_prompt, timestamp=TS2)

        support_entries = load_history(root, prompt_name="support")
        assert len(support_entries) == 1
        assert support_entries[0].prompt_name == "support"

        other_entries = load_history(root, prompt_name="other")
        assert len(other_entries) == 1

    def test_empty_history(self, tmp_path: Path):
        root = tmp_path / ".prompttest"
        root.mkdir()
        assert load_history(root) == []

    def test_no_history_dir(self, tmp_path: Path):
        root = tmp_path / ".prompttest"
        root.mkdir()
        assert load_history(root) == []


# ---------------------------------------------------------------------------
# detect_trend
# ---------------------------------------------------------------------------

class TestDetectTrend:
    def _entry(self, score: float) -> HistoryEntry:
        return HistoryEntry(
            timestamp="2026-01-01T00:00:00Z",
            prompt_name="x", prompt_version="1", model="m",
            provider="p", scorer="s", total=1, passed=1,
            failed=0, errors=0, accuracy=1.0,
            average_score=score, pass_threshold=0.7,
        )

    def test_improving(self):
        entries = [self._entry(0.5), self._entry(0.6), self._entry(0.7), self._entry(0.8)]
        assert detect_trend(entries) == "improving"

    def test_degrading(self):
        entries = [self._entry(0.9), self._entry(0.8), self._entry(0.7), self._entry(0.6)]
        assert detect_trend(entries) == "degrading"

    def test_stable(self):
        entries = [self._entry(0.8), self._entry(0.8), self._entry(0.8)]
        assert detect_trend(entries) == "stable"

    def test_single_entry(self):
        assert detect_trend([self._entry(0.8)]) == "stable"

    def test_empty(self):
        assert detect_trend([]) == "stable"

    def test_uses_last_window(self):
        # Long history but only last 3 matter
        entries = [self._entry(0.9), self._entry(0.8), self._entry(0.5),
                   self._entry(0.6), self._entry(0.7), self._entry(0.8)]
        assert detect_trend(entries, window=3) == "improving"


# ---------------------------------------------------------------------------
# HistoryEntry serialization
# ---------------------------------------------------------------------------

class TestHistoryEntry:
    def test_round_trip(self):
        entry = HistoryEntry(
            timestamp="2026-03-17T12:00:00Z",
            prompt_name="test", prompt_version="1", model="echo",
            provider="echo", scorer="contains", total=5, passed=4,
            failed=1, errors=0, accuracy=0.8,
            average_score=0.85, pass_threshold=0.7,
        )
        data = entry.to_dict()
        restored = HistoryEntry.from_dict(data)
        assert restored.prompt_name == entry.prompt_name
        assert restored.average_score == entry.average_score
        assert restored.total == entry.total


# ---------------------------------------------------------------------------
# Integration: eval → record → load round-trip
# ---------------------------------------------------------------------------

class TestEvalHistoryRoundTrip:
    def test_round_trip(self, tmp_path: Path):
        from prompttest.core.eval_runner import run_eval

        root = tmp_path / ".prompttest"
        root.mkdir()
        prompts_dir = root / "prompts"
        prompts_dir.mkdir()

        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{input}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\nscoring: contains\ntests:\n"
            "  - input: hello\n    expected: hello\n"
        )

        result = run_eval(dataset_yaml, prompt)
        record(root, result, prompt, timestamp=TS1)
        record(root, result, prompt, timestamp=TS2)

        entries = load_history(root)
        assert len(entries) == 2
        assert entries[0].average_score == 1.0
        assert detect_trend(entries) == "stable"
