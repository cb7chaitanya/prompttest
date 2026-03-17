"""Tests for prompttest.providers.key_validator."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from prompttest.providers.key_validator import (
    KeyCheckResult,
    check_key_present,
    validate_provider_key,
)


# ---------------------------------------------------------------------------
# check_key_present
# ---------------------------------------------------------------------------

class TestCheckKeyPresent:
    def test_openai_present(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            result = check_key_present("openai")
            assert result is not None
            assert result.provider == "openai"
            assert result.env_var == "OPENAI_API_KEY"
            assert result.present is True

    def test_openai_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            result = check_key_present("openai")
            assert result is not None
            assert result.present is False

    def test_openai_empty_string(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "  "}):
            result = check_key_present("openai")
            assert result.present is False

    def test_anthropic_present(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            result = check_key_present("anthropic")
            assert result is not None
            assert result.provider == "anthropic"
            assert result.env_var == "ANTHROPIC_API_KEY"
            assert result.present is True

    def test_anthropic_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            result = check_key_present("anthropic")
            assert result is not None
            assert result.present is False

    def test_echo_returns_none(self):
        assert check_key_present("echo") is None

    def test_local_returns_none(self):
        assert check_key_present("local") is None

    def test_unknown_provider_returns_none(self):
        assert check_key_present("nonexistent") is None


# ---------------------------------------------------------------------------
# validate_provider_key (presence mode)
# ---------------------------------------------------------------------------

class TestValidateProviderKey:
    def test_presence_check_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            result = validate_provider_key("openai", live=False)
            assert result is not None
            assert result.present is True
            assert result.valid is None  # not live-tested

    def test_presence_check_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            result = validate_provider_key("anthropic", live=False)
            assert result is not None
            assert result.present is False

    def test_echo_skipped(self):
        assert validate_provider_key("echo") is None

    def test_local_skipped(self):
        assert validate_provider_key("local") is None


# ---------------------------------------------------------------------------
# KeyCheckResult
# ---------------------------------------------------------------------------

class TestKeyCheckResult:
    def test_fields(self):
        r = KeyCheckResult(
            provider="openai",
            env_var="OPENAI_API_KEY",
            present=True,
            valid=True,
            error="",
        )
        assert r.provider == "openai"
        assert r.present is True
        assert r.valid is True

    def test_defaults(self):
        r = KeyCheckResult(provider="x", env_var="X", present=False)
        assert r.valid is None
        assert r.error == ""
