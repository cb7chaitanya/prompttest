"""Unit tests for scoring functions (migrated from evaluator)."""

from prompttest.core.scoring import contains, exact


def test_exact_match_pass():
    score, _ = exact("Hello World", "hello world")
    assert score == 1.0


def test_exact_match_fail():
    score, _ = exact("Hello", "Goodbye")
    assert score == 0.0


def test_contains_match_pass():
    score, _ = contains("The quick brown fox", "brown fox")
    assert score == 1.0


def test_contains_match_fail():
    score, _ = contains("The quick brown fox", "lazy dog")
    assert score == 0.0
