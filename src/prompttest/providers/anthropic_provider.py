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

    def complete(
        self,
        model: str,
        system: str,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        params = parameters or {}
        resp = client.messages.create(
            model=model,
            max_tokens=params.pop("max_tokens", 1024),
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": user_message}],
            **params,
        )
        return resp.content[0].text
