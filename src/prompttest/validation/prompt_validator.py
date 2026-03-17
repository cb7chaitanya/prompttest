"""Prompt template validation: ensure test case inputs match template placeholders."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from prompttest.core.eval_runner import EvalCase, EvalDataset
from prompttest.core.models import PromptConfig

# Matches {{placeholder_name}} allowing word characters (letters, digits, underscores)
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def extract_placeholders(template: str) -> set[str]:
    """Extract all ``{{name}}`` placeholder names from a template string."""
    return set(_PLACEHOLDER_RE.findall(template))


@dataclass
class ValidationWarning:
    """A non-fatal validation issue (extra fields in test case input)."""

    case_index: int
    extra_fields: list[str]

    @property
    def message(self) -> str:
        fields = ", ".join(f'"{f}"' for f in self.extra_fields)
        return f"Extra unused fields in test case #{self.case_index}: {fields}"


@dataclass
class ValidationError(Exception):
    """Raised when a test case is missing required placeholders."""

    case_index: int
    missing: list[str]

    def __str__(self) -> str:
        fields = ", ".join(f'"{f}"' for f in self.missing)
        return (
            f"Validation Error\n"
            f'Missing placeholder: {fields}\n'
            f"In test case #{self.case_index}"
        )


@dataclass
class ValidationResult:
    """Collects all errors and warnings from validating a dataset."""

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_test_case(
    required: set[str],
    case: EvalCase,
    case_index: int,
) -> tuple[ValidationError | None, ValidationWarning | None]:
    """Validate a single test case against the required placeholders.

    Returns a (error, warning) tuple — either or both may be ``None``.
    """
    provided = set(case.input.keys())

    missing = sorted(required - provided)
    extra = sorted(provided - required)

    error = ValidationError(case_index=case_index, missing=missing) if missing else None
    warning = ValidationWarning(case_index=case_index, extra_fields=extra) if extra else None

    return error, warning


def validate_dataset(
    prompt_config: PromptConfig,
    dataset: EvalDataset,
) -> ValidationResult:
    """Validate every test case in *dataset* against the prompt template placeholders."""
    required = extract_placeholders(prompt_config.template)
    result = ValidationResult()

    for i, case in enumerate(dataset.tests, start=1):
        error, warning = validate_test_case(required, case, case_index=i)
        if error:
            result.errors.append(error)
        if warning:
            result.warnings.append(warning)

    return result
