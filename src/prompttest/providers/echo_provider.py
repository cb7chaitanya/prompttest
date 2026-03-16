"""Echo provider for testing without API keys."""

from __future__ import annotations

from typing import Any

from prompttest.providers.base import LLMProvider


class EchoProvider(LLMProvider):
    """Returns the user message as-is. Useful for smoke-testing the pipeline."""

    def complete(
        self,
        model: str,
        system: str,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        return user_message
