"""Abstract base for LLM providers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Base class every LLM provider must implement.

    Subclasses **must** implement :meth:`complete`.
    Override :meth:`acomplete` for true async I/O; the default
    implementation runs :meth:`complete` in a thread executor.
    """

    @abstractmethod
    def complete(
        self,
        model: str,
        system: str,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Send a prompt and return the completion text (sync)."""

    async def acomplete(
        self,
        model: str,
        system: str,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Async version of :meth:`complete`.

        The default implementation offloads the sync call to a thread so
        providers work out of the box with ``asyncio``.  Providers with
        native async SDKs should override this.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.complete(model, system, user_message, parameters)
        )
