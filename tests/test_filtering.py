"""Tests for dataset tag filtering."""

from __future__ import annotations

from pathlib import Path

from prompttest.core.eval_runner import (
    EvalCase,
    EvalDataset,
    filter_by_tags,
    run_eval,
)
from prompttest.core.models import PromptConfig


def _make_dataset() -> EvalDataset:
    return EvalDataset(
        prompt="test",
        scoring="contains",
        tests=[
            EvalCase(input={"q": "1"}, expected="a", tags=["billing", "critical"]),
            EvalCase(input={"q": "2"}, expected="b", tags=["billing"]),
            EvalCase(input={"q": "3"}, expected="c", tags=["support"]),
            EvalCase(input={"q": "4"}, expected="d", tags=["critical"]),
            EvalCase(input={"q": "5"}, expected="e", tags=[]),
        ],
    )


# ---------------------------------------------------------------------------
# filter_by_tags
# ---------------------------------------------------------------------------

class TestFilterByTags:
    def test_no_tags_returns_all(self):
        ds = _make_dataset()
        orig, filt = filter_by_tags(ds, [])
        assert orig == 5
        assert filt == 5
        assert len(ds.tests) == 5

    def test_any_single_tag(self):
        ds = _make_dataset()
        orig, filt = filter_by_tags(ds, ["billing"], "any")
        assert orig == 5
        assert filt == 2
        assert all("billing" in c.tags for c in ds.tests)

    def test_any_multiple_tags(self):
        ds = _make_dataset()
        orig, filt = filter_by_tags(ds, ["billing", "support"], "any")
        assert orig == 5
        assert filt == 3  # cases 1, 2, 3

    def test_all_single_tag(self):
        ds = _make_dataset()
        orig, filt = filter_by_tags(ds, ["critical"], "all")
        assert filt == 2  # cases 1, 4

    def test_all_multiple_tags(self):
        ds = _make_dataset()
        orig, filt = filter_by_tags(ds, ["billing", "critical"], "all")
        assert filt == 1  # only case 1 has both

    def test_no_matches(self):
        ds = _make_dataset()
        _, filt = filter_by_tags(ds, ["nonexistent"], "any")
        assert filt == 0
        assert ds.tests == []

    def test_default_match_is_any(self):
        ds = _make_dataset()
        _, filt = filter_by_tags(ds, ["support"])
        assert filt == 1

    def test_preserves_order(self):
        ds = _make_dataset()
        filter_by_tags(ds, ["billing", "critical"], "any")
        expected_qs = ["1", "2", "4"]
        assert [c.input["q"] for c in ds.tests] == expected_qs

    def test_returns_correct_counts(self):
        ds = _make_dataset()
        orig, filt = filter_by_tags(ds, ["billing"], "any")
        assert orig == 5
        assert filt == 2


# ---------------------------------------------------------------------------
# Integration: filter + eval
# ---------------------------------------------------------------------------

class TestFilteredEval:
    def _write_dataset(self, tmp_path: Path) -> Path:
        yaml = (
            "prompt: test\n"
            "scoring: contains\n"
            "tests:\n"
            "  - input:\n"
            "      q: hello\n"
            "    expected: hello\n"
            "    tags: [smoke]\n"
            "  - input:\n"
            "      q: world\n"
            "    expected: world\n"
            "    tags: [regression]\n"
            "  - input:\n"
            "      q: foo\n"
            "    expected: NOPE\n"
            "    tags: [smoke, regression]\n"
        )
        p = tmp_path / "ds.yaml"
        p.write_text(yaml)
        return p

    def test_eval_with_tag_filter(self, tmp_path: Path):
        from prompttest.core.eval_runner import load_eval_dataset

        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{q}}",
        )
        ds_path = self._write_dataset(tmp_path)

        # Load, filter, then run (simulating CLI flow)
        ds = load_eval_dataset(ds_path)
        orig, filt = filter_by_tags(ds, ["smoke"], "any")
        assert orig == 3
        assert filt == 2  # cases 1 and 3

        # Run eval on the already-filtered dataset by writing it back
        # Instead, we test via run_eval which loads fresh — so let's
        # write a filtered dataset
        filtered_yaml = (
            "prompt: test\n"
            "scoring: contains\n"
            "tests:\n"
            "  - input:\n"
            "      q: hello\n"
            "    expected: hello\n"
            "    tags: [smoke]\n"
        )
        filtered_path = tmp_path / "filtered.yaml"
        filtered_path.write_text(filtered_yaml)
        result = run_eval(filtered_path, prompt)
        assert result.total == 1
        assert result.passed == 1

    def test_all_match_filter(self, tmp_path: Path):
        from prompttest.core.eval_runner import load_eval_dataset

        ds_path = self._write_dataset(tmp_path)
        ds = load_eval_dataset(ds_path)
        _, filt = filter_by_tags(ds, ["smoke", "regression"], "all")
        assert filt == 1  # only case 3 has both tags
        assert ds.tests[0].input["q"] == "foo"
