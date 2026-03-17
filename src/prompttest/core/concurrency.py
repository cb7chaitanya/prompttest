"""Concurrency control, rate limiting, and retry logic for async eval runs."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

# HTTP status codes and exception substrings that trigger a retry.
_RATE_LIMIT_STATUS = 429
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
_RETRYABLE_SUBSTRINGS = ("rate limit", "rate_limit", "overloaded", "connection", "timeout")


def is_retryable(exc: Exception) -> bool:
    """Return ``True`` if *exc* looks like a transient / rate-limit error."""
    # Check for status_code attribute (openai, anthropic, httpx all expose this)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None and int(status) in _RETRYABLE_STATUS_CODES:
        return True

    # Fallback: sniff the exception message
    msg = str(exc).lower()
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


@dataclass
class ConcurrencyConfig:
    """Runtime settings for concurrency, rate limiting, and retries."""

    max_concurrency: int = 10
    rate_limit: float = 0.0  # requests per second; 0 = unlimited
    max_retries: int = 3
    base_delay: float = 1.0  # seconds — initial retry delay
    max_delay: float = 60.0  # seconds — retry delay cap


class RateLimiter:
    """Simple token-bucket style rate limiter for asyncio.

    Ensures no more than *rps* calls proceed per second.  When *rps* is 0 or
    negative the limiter is a no-op.
    """

    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / rps if rps > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last: float = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._last + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = loop.time()


async def run_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> T:
    """Call *fn* with exponential-backoff retry on retryable errors.

    Non-retryable exceptions are re-raised immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries or not is_retryable(exc):
                raise
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5), max_delay)
            await asyncio.sleep(delay)

    # Should never reach here, but satisfies the type checker.
    raise last_exc  # type: ignore[misc]


async def run_concurrently(
    tasks: list[Callable[[], Awaitable[T]]],
    config: ConcurrencyConfig,
    on_complete: Callable[[], None] | None = None,
) -> list[T]:
    """Run *tasks* with bounded concurrency, rate limiting, and retry.

    *tasks* is a list of zero-argument async callables.  Returns results in
    the same order as *tasks*.

    *on_complete* is called (synchronously) after each task finishes,
    regardless of success or failure — useful for progress bars.
    """
    semaphore = asyncio.Semaphore(config.max_concurrency)
    limiter = RateLimiter(config.rate_limit)

    async def _guarded(fn: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            await limiter.acquire()
            try:
                return await run_with_retry(
                    fn,
                    max_retries=config.max_retries,
                    base_delay=config.base_delay,
                    max_delay=config.max_delay,
                )
            finally:
                if on_complete is not None:
                    on_complete()

    return list(await asyncio.gather(*[_guarded(t) for t in tasks]))
