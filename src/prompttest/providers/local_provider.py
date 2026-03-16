"""Local model provider — placeholder for locally-hosted LLMs.

Expects a local OpenAI-compatible server (e.g. llama.cpp, Ollama, vLLM)
at the URL specified by the ``LOCAL_MODEL_URL`` environment variable
(default: ``http://localhost:11434/v1``).
"""

from __future__ import annotations

import os
from typing import Any

from prompttest.providers.base import LLMProvider

DEFAULT_BASE_URL = "http://localhost:11434/v1"


class LocalProvider(LLMProvider):
    """Connects to a local OpenAI-compatible API endpoint."""

    def __init__(self) -> None:
        self._base_url = os.environ.get("LOCAL_MODEL_URL", DEFAULT_BASE_URL)

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
        import httpx

        messages = self._build_messages(system, user_message)
        params = parameters or {}
        payload: dict[str, Any] = {"model": model, "messages": messages, **params}

        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def acomplete(
        self,
        model: str,
        system: str,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        import httpx

        messages = self._build_messages(system, user_message)
        params = parameters or {}
        payload: dict[str, Any] = {"model": model, "messages": messages, **params}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
