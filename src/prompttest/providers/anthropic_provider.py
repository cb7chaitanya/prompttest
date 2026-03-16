"""Anthropic provider."""

from __future__ import annotations

import os
from typing import Any

from prompttest.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError("Install the anthropic extra: pip install prompttest[anthropic]")
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    @staticmethod
    def _prepare_params(parameters: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
        params = dict(parameters) if parameters else {}
        max_tokens = params.pop("max_tokens", 1024)
        return max_tokens, params

    def complete(
        self,
        model: str,
        system: str,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        max_tokens, params = self._prepare_params(parameters)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": user_message}],
            **params,
        )
        return resp.content[0].text

    async def acomplete(
        self,
        model: str,
        system: str,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        max_tokens, params = self._prepare_params(parameters)
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": user_message}],
            **params,
        )
        return resp.content[0].text
