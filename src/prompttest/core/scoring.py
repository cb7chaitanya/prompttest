"""Extensible scoring function registry for evaluation."""

from __future__ import annotations

from typing import Callable, Protocol

# A scorer takes (output, expected) and returns (score 0-1, reason).
ScorerFn = Callable[[str, str], tuple[float, str]]


class Scorer(Protocol):
    """Protocol that any scoring callable must satisfy."""

    def __call__(self, output: str, expected: str) -> tuple[float, str]: ...


# ---------------------------------------------------------------------------
# Built-in scorers
# ---------------------------------------------------------------------------

def contains(output: str, expected: str) -> tuple[float, str]:
    """Score 1.0 if expected text appears anywhere in the output (case-insensitive)."""
    if expected.strip().lower() in output.strip().lower():
        return 1.0, "contains expected text"
    return 0.0, f"output does not contain '{expected}'"


def exact(output: str, expected: str) -> tuple[float, str]:
    """Score 1.0 if normalized strings match exactly."""
    if output.strip().lower() == expected.strip().lower():
        return 1.0, "exact match"
    return 0.0, f"expected '{expected}', got '{output}'"


def starts_with(output: str, expected: str) -> tuple[float, str]:
    """Score 1.0 if output starts with the expected text (case-insensitive)."""
    if output.strip().lower().startswith(expected.strip().lower()):
        return 1.0, "output starts with expected text"
    return 0.0, f"output does not start with '{expected}'"


def ends_with(output: str, expected: str) -> tuple[float, str]:
    """Score 1.0 if output ends with the expected text (case-insensitive)."""
    if output.strip().lower().endswith(expected.strip().lower()):
        return 1.0, "output ends with expected text"
    return 0.0, f"output does not end with '{expected}'"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SCORERS: dict[str, ScorerFn] = {
    "contains": contains,
    "exact": exact,
    "starts_with": starts_with,
    "ends_with": ends_with,
}

DEFAULT_SCORER = "contains"
PASS_THRESHOLD = 1.0


def register_scorer(name: str, fn: ScorerFn) -> None:
    """Register a custom scoring function."""
    _SCORERS[name] = fn


def get_scorer(name: str) -> ScorerFn:
    """Look up a scorer by name. Raises KeyError if not found."""
    if name not in _SCORERS:
        available = ", ".join(sorted(_SCORERS))
        raise KeyError(f"Unknown scorer '{name}'. Available: {available}")
    return _SCORERS[name]


def list_scorers() -> list[str]:
    """Return sorted list of registered scorer names."""
    return sorted(_SCORERS)
