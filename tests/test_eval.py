"""Tests for the evaluation dataset engine."""

from __future__ import annotations

from pathlib import Path

from prompttest.core.eval_runner import (
    EvalResult,
    load_eval_dataset,
    render_template_dict,
    run_eval,
)
from prompttest.core.models import PromptConfig
from prompttest.core.scoring import (
    contains,
    ends_with,
    exact,
    get_scorer,
    list_scorers,
    register_scorer,
    starts_with,
)


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

def test_contains_scorer_pass() -> None:
    score, _ = contains("Our refund policy is 30 days.", "30 days")
    assert score == 1.0


def test_contains_scorer_fail() -> None:
    score, _ = contains("No refunds.", "30 days")
    assert score == 0.0


def test_exact_scorer_pass() -> None:
    score, _ = exact("  Yes  ", "yes")
    assert score == 1.0


def test_exact_scorer_fail() -> None:
    score, _ = exact("Yes, we do.", "yes")
    assert score == 0.0


def test_starts_with_scorer() -> None:
    score, _ = starts_with("Hello world", "hello")
    assert score == 1.0


def test_ends_with_scorer() -> None:
    score, _ = ends_with("Hello world", "world")
    assert score == 1.0


def test_register_custom_scorer() -> None:
    def always_pass(output: str, expected: str) -> tuple[float, str]:
        return 1.0, "always passes"

    register_scorer("always_pass", always_pass)
    fn = get_scorer("always_pass")
    score, _ = fn("anything", "anything")
    assert score == 1.0
    assert "always_pass" in list_scorers()


# ---------------------------------------------------------------------------
# Templating tests
# ---------------------------------------------------------------------------

def test_render_template_dict_single_var() -> None:
    result = render_template_dict("Question: {{question}}", {"question": "How?"})
    assert result == "Question: How?"


def test_render_template_dict_multiple_vars() -> None:
    template = "{{name}} asks: {{question}}"
    result = render_template_dict(template, {"name": "Alice", "question": "Why?"})
    assert result == "Alice asks: Why?"


# ---------------------------------------------------------------------------
# Dataset loading tests
# ---------------------------------------------------------------------------

def test_load_eval_dataset(tmp_path: Path) -> None:
    yaml_content = """\
prompt: support_v1
scoring: exact
tests:
  - input:
      question: "What is your refund policy?"
    expected: "30 days"
  - input:
      question: "Do you offer support?"
    expected: "Yes"
"""
    dataset_file = tmp_path / "dataset.yaml"
    dataset_file.write_text(yaml_content)

    ds = load_eval_dataset(dataset_file)
    assert ds.prompt == "support_v1"
    assert ds.scoring == "exact"
    assert len(ds.tests) == 2
    assert ds.tests[0].input == {"question": "What is your refund policy?"}
    assert ds.tests[1].expected == "Yes"


def test_load_eval_dataset_defaults(tmp_path: Path) -> None:
    yaml_content = """\
prompt: bot
tests:
  - input:
      q: "hi"
    expected: "hello"
"""
    dataset_file = tmp_path / "ds.yaml"
    dataset_file.write_text(yaml_content)

    ds = load_eval_dataset(dataset_file)
    assert ds.scoring == "contains"  # default


def test_load_eval_dataset_string_input(tmp_path: Path) -> None:
    yaml_content = """\
prompt: bot
tests:
  - input: "hello"
    expected: "hello"
"""
    dataset_file = tmp_path / "ds.yaml"
    dataset_file.write_text(yaml_content)

    ds = load_eval_dataset(dataset_file)
    assert ds.tests[0].input == {"input": "hello"}


# ---------------------------------------------------------------------------
# End-to-end eval run (using echo provider)
# ---------------------------------------------------------------------------

def test_run_eval_with_echo_provider(tmp_path: Path) -> None:
    dataset_yaml = """\
prompt: test_prompt
scoring: contains
tests:
  - input:
      question: "What is your refund policy?"
    expected: "refund policy"
  - input:
      question: "Do you offer support?"
    expected: "NOPE_NOT_HERE"
"""
    dataset_file = tmp_path / "eval_dataset.yaml"
    dataset_file.write_text(dataset_yaml)

    # Echo provider returns the user_message as-is
    prompt = PromptConfig(
        name="test_prompt",
        version="1",
        model="echo",
        provider="echo",
        system="",
        template="Answer: {{question}}",
    )

    result = run_eval(dataset_file, prompt)
    assert isinstance(result, EvalResult)
    assert result.total == 2
    assert result.passed == 1  # "refund policy" is in the echoed output
    assert result.failed == 1  # "NOPE_NOT_HERE" is not in the echoed output
    assert result.accuracy == 0.5


def test_eval_case_input_summary() -> None:
    from prompttest.core.eval_runner import EvalCase

    case = EvalCase(input={"question": "hi", "context": "test"}, expected="hello")
    summary = case.input_summary
    assert "question=" in summary
    assert "context=" in summary
