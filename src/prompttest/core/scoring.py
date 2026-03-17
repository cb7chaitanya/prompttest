"""Extensible scoring function registry for evaluation."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Callable, Protocol

# A scorer takes (output, expected) and returns (score 0-1, reason).
ScorerFn = Callable[[str, str], tuple[float, str]]


class Scorer(Protocol):
    """Protocol that any scoring callable must satisfy."""

    def __call__(self, output: str, expected: str) -> tuple[float, str]: ...


# ---------------------------------------------------------------------------
# Built-in scorers — binary
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


def regex(output: str, expected: str) -> tuple[float, str]:
    """Score 1.0 if *expected* (a regex pattern) matches anywhere in *output*."""
    try:
        if re.search(expected, output, re.IGNORECASE):
            return 1.0, f"regex /{expected}/ matched"
        return 0.0, f"regex /{expected}/ did not match"
    except re.error as exc:
        return 0.0, f"invalid regex: {exc}"


# ---------------------------------------------------------------------------
# Built-in scorers — fuzzy / continuous
# ---------------------------------------------------------------------------

def fuzzy(output: str, expected: str) -> tuple[float, str]:
    """Fuzzy string similarity using ``rapidfuzz`` (if available) or stdlib ``SequenceMatcher``.

    Returns a score between 0.0 and 1.0 representing character-level similarity.
    """
    a = output.strip().lower()
    b = expected.strip().lower()
    if a == b:
        return 1.0, "exact match"

    try:
        from rapidfuzz import fuzz  # type: ignore[import-untyped]
        score = fuzz.ratio(a, b) / 100.0
    except ImportError:
        score = SequenceMatcher(None, a, b).ratio()

    return score, f"fuzzy similarity {score:.2f}"


def semantic(output: str, expected: str) -> tuple[float, str]:
    """Semantic similarity using OpenAI embeddings and cosine distance.

    Requires the ``openai`` extra (``pip install prompttest[openai]``) and
    the ``OPENAI_API_KEY`` environment variable.
    """
    try:
        import openai
    except ImportError:
        return 0.0, "openai package not installed — pip install prompttest[openai]"

    import math
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return 0.0, "OPENAI_API_KEY not set"

    client = openai.OpenAI(api_key=api_key)
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[output.strip(), expected.strip()],
    )
    vec_a = resp.data[0].embedding
    vec_b = resp.data[1].embedding

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0, "zero-length embedding"

    score = max(0.0, min(1.0, dot / (norm_a * norm_b)))
    return score, f"semantic similarity {score:.2f}"


def llm_judge(output: str, expected: str) -> tuple[float, str]:
    """Ask an LLM whether *output* satisfactorily matches *expected*.

    Uses OpenAI by default.  The judge returns a score between 0.0 and 1.0.
    Requires the ``openai`` extra and ``OPENAI_API_KEY``.
    """
    try:
        import openai
    except ImportError:
        return 0.0, "openai package not installed — pip install prompttest[openai]"

    import json
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return 0.0, "OPENAI_API_KEY not set"

    judge_model = os.environ.get("PROMPTTEST_JUDGE_MODEL", "gpt-4o-mini")

    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluation judge. Compare the actual output "
                    "against the expected output and return a JSON object with two "
                    'fields: "score" (a float from 0.0 to 1.0 indicating how well '
                    'the actual output matches the expected output) and "reason" '
                    "(a brief explanation). Only return JSON, no other text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Expected output:\n{expected}\n\n"
                    f"Actual output:\n{output}"
                ),
            },
        ],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or ""
    try:
        data = json.loads(raw)
        score = float(data["score"])
        score = max(0.0, min(1.0, score))
        reason = str(data.get("reason", f"llm judge score {score:.2f}"))
        return score, reason
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0, f"judge returned unparseable response: {raw[:200]}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SCORERS: dict[str, ScorerFn] = {
    "contains": contains,
    "exact": exact,
    "starts_with": starts_with,
    "ends_with": ends_with,
    "regex": regex,
    "fuzzy": fuzzy,
    "semantic": semantic,
    "llm_judge": llm_judge,
}

DEFAULT_SCORER = "contains"
PASS_THRESHOLD = 0.7


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
