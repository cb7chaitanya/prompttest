"""Abstract base for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        model: str,
        system: str,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Send a prompt and return the completion text."""
