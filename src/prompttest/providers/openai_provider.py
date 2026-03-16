"""OpenAI-compatible provider."""

from __future__ import annotations

import os
from typing import Any

from prompttest.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError("Install the openai extra: pip install prompttest[openai]")
        self._api_key = os.environ.get("OPENAI_API_KEY", "")

    def _build_messages(self, system: str, user_message: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})
        return messages

    def complete(
        self,
        model: str,
        system: str,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        messages = self._build_messages(system, user_message)
        params = parameters or {}
        resp = client.chat.completions.create(model=model, messages=messages, **params)
        return resp.choices[0].message.content or ""

    async def acomplete(
        self,
        model: str,
        system: str,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        import openai

        client = openai.AsyncOpenAI(api_key=self._api_key)
        messages = self._build_messages(system, user_message)
        params = parameters or {}
        resp = await client.chat.completions.create(model=model, messages=messages, **params)
        return resp.choices[0].message.content or ""
