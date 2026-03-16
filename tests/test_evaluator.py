"""Unit tests for the evaluator."""

from prompttest.core.evaluator import contains_match, exact_match


def test_exact_match_pass():
    score, _ = exact_match("Hello World", "hello world")
    assert score == 1.0


def test_exact_match_fail():
    score, _ = exact_match("Hello", "Goodbye")
    assert score == 0.0


def test_contains_match_pass():
    score, _ = contains_match("The quick brown fox", "brown fox")
    assert score == 1.0


def test_contains_match_fail():
    score, _ = contains_match("The quick brown fox", "lazy dog")
    assert score == 0.0
