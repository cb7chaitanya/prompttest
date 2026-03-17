"""Tests for prompttest.validation.prompt_validator."""

from __future__ import annotations

import pytest

from prompttest.core.eval_runner import EvalCase, EvalDataset
from prompttest.core.models import PromptConfig
from prompttest.validation.prompt_validator import (
    ValidationError,
    ValidationWarning,
    extract_placeholders,
    validate_dataset,
    validate_test_case,
)


# ---------------------------------------------------------------------------
# extract_placeholders
# ---------------------------------------------------------------------------

class TestExtractPlaceholders:
    def test_single_placeholder(self):
        assert extract_placeholders("Hello {{name}}!") == {"name"}

    def test_multiple_placeholders(self):
        result = extract_placeholders("{{greeting}}, {{name}}! How is {{topic}}?")
        assert result == {"greeting", "name", "topic"}

    def test_no_placeholders(self):
        assert extract_placeholders("Hello world!") == set()

    def test_duplicate_placeholders(self):
        assert extract_placeholders("{{x}} and {{x}}") == {"x"}

    def test_underscores_and_digits(self):
        assert extract_placeholders("{{user_name}} {{item2}}") == {"user_name", "item2"}

    def test_empty_string(self):
        assert extract_placeholders("") == set()

    def test_nested_braces_ignored(self):
        # {{{foo}}} — inner {{foo}} should still match
        assert extract_placeholders("{{{foo}}}") == {"foo"}


# ---------------------------------------------------------------------------
# validate_test_case
# ---------------------------------------------------------------------------

class TestValidateTestCase:
    def test_valid_case(self):
        required = {"question", "context"}
        case = EvalCase(input={"question": "What?", "context": "Here"}, expected="answer")
        error, warning = validate_test_case(required, case, case_index=1)
        assert error is None
        assert warning is None

    def test_missing_placeholder(self):
        required = {"question", "context"}
        case = EvalCase(input={"question": "What?"}, expected="answer")
        error, warning = validate_test_case(required, case, case_index=3)
        assert error is not None
        assert error.case_index == 3
        assert error.missing == ["context"]
        assert warning is None

    def test_extra_field(self):
        required = {"question"}
        case = EvalCase(input={"question": "What?", "bonus": "extra"}, expected="answer")
        error, warning = validate_test_case(required, case, case_index=1)
        assert error is None
        assert warning is not None
        assert warning.extra_fields == ["bonus"]

    def test_both_missing_and_extra(self):
        required = {"question", "context"}
        case = EvalCase(input={"bonus": "extra"}, expected="answer")
        error, warning = validate_test_case(required, case, case_index=2)
        assert error is not None
        assert sorted(error.missing) == ["context", "question"]
        assert warning is not None
        assert warning.extra_fields == ["bonus"]


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------

def _make_prompt(template: str) -> PromptConfig:
    return PromptConfig(
        name="test",
        version="1",
        model="echo",
        provider="echo",
        system="",
        template=template,
    )


class TestValidateDataset:
    def test_all_valid(self):
        prompt = _make_prompt("Question: {{question}}")
        dataset = EvalDataset(
            prompt="test",
            scoring="contains",
            tests=[
                EvalCase(input={"question": "What?"}, expected="answer"),
                EvalCase(input={"question": "Why?"}, expected="because"),
            ],
        )
        result = validate_dataset(prompt, dataset)
        assert result.ok
        assert result.errors == []
        assert result.warnings == []

    def test_missing_in_one_case(self):
        prompt = _make_prompt("{{question}} about {{topic}}")
        dataset = EvalDataset(
            prompt="test",
            scoring="contains",
            tests=[
                EvalCase(input={"question": "What?", "topic": "cats"}, expected="ok"),
                EvalCase(input={"question": "Why?"}, expected="ok"),  # missing topic
            ],
        )
        result = validate_dataset(prompt, dataset)
        assert not result.ok
        assert len(result.errors) == 1
        assert result.errors[0].case_index == 2
        assert result.errors[0].missing == ["topic"]

    def test_extra_fields_warning(self):
        prompt = _make_prompt("{{question}}")
        dataset = EvalDataset(
            prompt="test",
            scoring="contains",
            tests=[
                EvalCase(input={"question": "What?", "extra": "val"}, expected="ok"),
            ],
        )
        result = validate_dataset(prompt, dataset)
        assert result.ok  # warnings don't block
        assert len(result.warnings) == 1
        assert result.warnings[0].extra_fields == ["extra"]

    def test_empty_dataset(self):
        prompt = _make_prompt("{{question}}")
        dataset = EvalDataset(prompt="test", scoring="contains", tests=[])
        result = validate_dataset(prompt, dataset)
        assert result.ok

    def test_no_placeholders_in_template(self):
        prompt = _make_prompt("Static prompt with no placeholders")
        dataset = EvalDataset(
            prompt="test",
            scoring="contains",
            tests=[
                EvalCase(input={"question": "ignored"}, expected="ok"),
            ],
        )
        result = validate_dataset(prompt, dataset)
        assert result.ok  # no required fields = nothing can be missing
        assert len(result.warnings) == 1  # extra field warning


# ---------------------------------------------------------------------------
# ValidationError formatting
# ---------------------------------------------------------------------------

class TestValidationError:
    def test_str(self):
        err = ValidationError(case_index=3, missing=["question"])
        text = str(err)
        assert "Missing placeholder" in text
        assert '"question"' in text
        assert "#3" in text

    def test_is_exception(self):
        err = ValidationError(case_index=1, missing=["x"])
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# ValidationWarning formatting
# ---------------------------------------------------------------------------

class TestValidationWarning:
    def test_message(self):
        w = ValidationWarning(case_index=2, extra_fields=["bonus", "unused"])
        assert "bonus" in w.message
        assert "#2" in w.message


# ---------------------------------------------------------------------------
# Integration: run_eval with strict flag
# ---------------------------------------------------------------------------

class TestRunEvalStrictFlag:
    def test_strict_raises_on_missing(self, tmp_path):
        from prompttest.core.eval_runner import run_eval

        prompt = _make_prompt("Answer: {{question}}")
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\n"
            "tests:\n"
            "  - input:\n"
            "      wrong_key: value\n"
            "    expected: answer\n"
        )
        with pytest.raises(ValidationError) as exc_info:
            run_eval(dataset_yaml, prompt, strict=True)
        assert exc_info.value.missing == ["question"]
        assert exc_info.value.case_index == 1

    def test_no_strict_continues(self, tmp_path):
        """With strict=False the runner should proceed (echo provider echoes input)."""
        from prompttest.core.eval_runner import run_eval

        prompt = _make_prompt("Answer: {{question}}")
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\n"
            "tests:\n"
            "  - input:\n"
            "      wrong_key: value\n"
            "    expected: 'Answer: {{question}}'\n"
        )
        # Should not raise — runs with un-substituted placeholder
        result = run_eval(dataset_yaml, prompt, strict=False)
        assert result.total == 1
