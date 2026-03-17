"""Evaluation targets: abstractions over prompt templates, HTTP endpoints, and callables."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from prompttest.core.eval_runner import render_template_dict
from prompttest.core.models import PromptConfig
from prompttest.providers.base import LLMProvider
from prompttest.providers.registry import get_provider


class EvalTarget(ABC):
    """Base class for anything that can be evaluated against a dataset.

    An eval target accepts a dict of input variables and returns a string output.
    """

    @property
    def name(self) -> str:
        """Human-readable name for display and history."""
        return self.__class__.__name__

    @property
    def version(self) -> str:
        return "1"

    @abstractmethod
    def call(self, inputs: dict[str, str]) -> str:
        """Synchronously produce an output from *inputs*."""

    async def acall(self, inputs: dict[str, str]) -> str:
        """Async version of :meth:`call`.

        Default runs :meth:`call` in a thread executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.call(inputs))


class PromptTarget(EvalTarget):
    """Wraps an existing prompt config + provider as an eval target."""

    def __init__(
        self,
        prompt_config: PromptConfig,
        provider: LLMProvider | None = None,
    ) -> None:
        self._config = prompt_config
        self._provider = provider or get_provider(prompt_config.provider)

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def version(self) -> str:
        return self._config.version

    @property
    def config(self) -> PromptConfig:
        return self._config

    def call(self, inputs: dict[str, str]) -> str:
        user_message = render_template_dict(self._config.template, inputs)
        return self._provider.complete(
            model=self._config.model,
            system=self._config.system,
            user_message=user_message,
            parameters=self._config.parameters,
        )

    async def acall(self, inputs: dict[str, str]) -> str:
        user_message = render_template_dict(self._config.template, inputs)
        return await self._provider.acomplete(
            model=self._config.model,
            system=self._config.system,
            user_message=user_message,
            parameters=self._config.parameters,
        )


class HttpTarget(EvalTarget):
    """Sends inputs as a POST request to an HTTP endpoint.

    The endpoint receives a JSON body with the input dict and must return
    a JSON response with an ``"output"`` field (configurable via *response_key*).
    """

    def __init__(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        response_key: str = "output",
        timeout: float = 120.0,
        name: str = "",
        version: str = "1",
    ) -> None:
        self._endpoint = endpoint
        self._headers = headers or {}
        self._response_key = response_key
        self._timeout = timeout
        self._name = name or endpoint
        self._version = version

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def call(self, inputs: dict[str, str]) -> str:
        import httpx

        resp = httpx.post(
            self._endpoint,
            json=inputs,
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, str):
            return data
        return str(data.get(self._response_key, data))

    async def acall(self, inputs: dict[str, str]) -> str:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._endpoint,
                json=inputs,
                headers=self._headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, str):
                return data
            return str(data.get(self._response_key, data))


class CallableTarget(EvalTarget):
    """Wraps a Python function as an eval target.

    The function receives the input dict and must return a string.
    Supports both sync functions and async coroutines.
    """

    def __init__(
        self,
        fn: Any,
        *,
        name: str = "",
        version: str = "1",
    ) -> None:
        self._fn = fn
        self._name = name or getattr(fn, "__name__", "callable")
        self._version = version
        self._is_async = asyncio.iscoroutinefunction(fn)

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def call(self, inputs: dict[str, str]) -> str:
        if self._is_async:
            raise RuntimeError(
                f"Cannot call async function {self._name!r} synchronously. "
                f"Use evaluate(..., use_async=True)."
            )
        result = self._fn(inputs)
        return str(result)

    async def acall(self, inputs: dict[str, str]) -> str:
        if self._is_async:
            result = await self._fn(inputs)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self._fn(inputs))
        return str(result)
