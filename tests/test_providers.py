"""Tests for the provider registry and base provider."""

from __future__ import annotations

import asyncio

import pytest

from prompttest.providers.base import LLMProvider
from prompttest.providers.echo_provider import EchoProvider
from prompttest.providers.registry import get_provider, list_providers, resolve_model


def test_resolve_model_openai() -> None:
    provider, model_id = resolve_model("gpt-4o")
    assert provider == "openai"
    assert model_id == "gpt-4o"


def test_resolve_model_anthropic() -> None:
    provider, model_id = resolve_model("claude-3-haiku-20240307")
    assert provider == "anthropic"
    assert model_id == "claude-3-haiku-20240307"


def test_resolve_model_local() -> None:
    provider, model_id = resolve_model("llama3.1")
    assert provider == "local"


def test_resolve_model_unknown() -> None:
    with pytest.raises(ValueError, match="Cannot determine provider"):
        resolve_model("some-unknown-model")


def test_get_provider_echo() -> None:
    p = get_provider("echo")
    assert isinstance(p, EchoProvider)


def test_get_provider_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonexistent")


def test_list_providers() -> None:
    names = list_providers()
    assert "echo" in names
    assert "openai" in names
    assert "anthropic" in names
    assert "local" in names


def test_echo_provider_sync() -> None:
    p = EchoProvider()
    out = p.complete(model="x", system="s", user_message="hello")
    assert out == "hello"


def test_echo_provider_async() -> None:
    p = EchoProvider()
    out = asyncio.run(p.acomplete(model="x", system="s", user_message="hello"))
    assert out == "hello"


def test_base_provider_acomplete_fallback() -> None:
    """The default acomplete should delegate to complete via executor."""

    class SyncOnly(LLMProvider):
        def complete(self, model, system, user_message, parameters=None):
            return f"sync:{user_message}"

    p = SyncOnly()
    result = asyncio.run(p.acomplete(model="m", system="s", user_message="test"))
    assert result == "sync:test"
