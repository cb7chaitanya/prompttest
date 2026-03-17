"""Tests for prompttest.core.concurrency."""

from __future__ import annotations

import asyncio
import time

import pytest

from prompttest.core.concurrency import (
    ConcurrencyConfig,
    RateLimiter,
    is_retryable,
    run_concurrently,
    run_with_retry,
)


# ---------------------------------------------------------------------------
# is_retryable
# ---------------------------------------------------------------------------

class TestIsRetryable:
    def test_429_status_code_attr(self):
        exc = Exception("rate limited")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert is_retryable(exc)

    def test_500_status_attr(self):
        exc = Exception("internal error")
        exc.status = 500  # type: ignore[attr-defined]
        assert is_retryable(exc)

    def test_non_retryable_status(self):
        exc = Exception("bad request")
        exc.status_code = 400  # type: ignore[attr-defined]
        assert not is_retryable(exc)

    def test_rate_limit_in_message(self):
        assert is_retryable(Exception("Rate limit exceeded"))

    def test_connection_in_message(self):
        assert is_retryable(Exception("Connection reset by peer"))

    def test_timeout_in_message(self):
        assert is_retryable(Exception("Request timeout"))

    def test_non_retryable_message(self):
        assert not is_retryable(Exception("Invalid API key"))

    def test_overloaded_in_message(self):
        assert is_retryable(Exception("Server overloaded"))


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_unlimited_no_delay(self):
        limiter = RateLimiter(0)
        async def _run():
            t0 = time.monotonic()
            for _ in range(5):
                await limiter.acquire()
            return time.monotonic() - t0

        elapsed = asyncio.run(_run())
        assert elapsed < 0.1  # should be near-instant

    def test_rate_limited(self):
        limiter = RateLimiter(20)  # 20 rps → 50ms between calls
        async def _run():
            t0 = time.monotonic()
            for _ in range(3):
                await limiter.acquire()
            return time.monotonic() - t0

        elapsed = asyncio.run(_run())
        # 3 calls at 20rps → 2 intervals of ~50ms = ~100ms minimum
        assert elapsed >= 0.08  # allow a bit of timing slack


# ---------------------------------------------------------------------------
# run_with_retry
# ---------------------------------------------------------------------------

class TestRunWithRetry:
    def test_success_no_retry(self):
        call_count = 0
        async def _fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = asyncio.run(run_with_retry(_fn, max_retries=3, base_delay=0.01))
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_retryable_error(self):
        call_count = 0
        async def _fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = Exception("rate limit hit")
                exc.status_code = 429  # type: ignore[attr-defined]
                raise exc
            return "recovered"

        result = asyncio.run(run_with_retry(_fn, max_retries=3, base_delay=0.01))
        assert result == "recovered"
        assert call_count == 3

    def test_raises_non_retryable_immediately(self):
        call_count = 0
        async def _fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            asyncio.run(run_with_retry(_fn, max_retries=3, base_delay=0.01))
        assert call_count == 1

    def test_exhausts_retries(self):
        call_count = 0
        async def _fn():
            nonlocal call_count
            call_count += 1
            raise Exception("connection timeout")

        with pytest.raises(Exception, match="connection timeout"):
            asyncio.run(run_with_retry(_fn, max_retries=2, base_delay=0.01))
        assert call_count == 3  # 1 initial + 2 retries


# ---------------------------------------------------------------------------
# run_concurrently
# ---------------------------------------------------------------------------

class TestRunConcurrently:
    def test_preserves_order(self):
        async def _make(val: int):
            return val

        tasks = [lambda v=i: _make(v) for i in range(5)]
        cfg = ConcurrencyConfig(max_concurrency=5, max_retries=0)
        results = asyncio.run(run_concurrently(tasks, cfg))
        assert results == [0, 1, 2, 3, 4]

    def test_respects_max_concurrency(self):
        """At most N tasks run at the same time."""
        peak = 0
        current = 0
        lock = None

        async def _task():
            nonlocal peak, current, lock
            if lock is None:
                lock = asyncio.Lock()
            async with lock:
                current += 1
                if current > peak:
                    peak = current
            await asyncio.sleep(0.02)
            async with lock:
                current -= 1
            return True

        tasks = [lambda: _task() for _ in range(10)]
        cfg = ConcurrencyConfig(max_concurrency=3, max_retries=0)
        results = asyncio.run(run_concurrently(tasks, cfg))
        assert all(results)
        assert peak <= 3

    def test_on_complete_callback(self):
        completed = 0

        async def _task():
            return True

        def _on_complete():
            nonlocal completed
            completed += 1

        tasks = [lambda: _task() for _ in range(5)]
        cfg = ConcurrencyConfig(max_concurrency=5, max_retries=0)
        asyncio.run(run_concurrently(tasks, cfg, on_complete=_on_complete))
        assert completed == 5

    def test_retries_within_run_concurrently(self):
        call_counts: dict[int, int] = {}

        def _make_task(idx: int):
            async def _task():
                call_counts[idx] = call_counts.get(idx, 0) + 1
                if call_counts[idx] == 1 and idx == 0:
                    exc = Exception("rate limit")
                    exc.status_code = 429  # type: ignore[attr-defined]
                    raise exc
                return idx
            return _task

        tasks = [_make_task(i) for i in range(3)]
        cfg = ConcurrencyConfig(max_concurrency=3, max_retries=2, base_delay=0.01)
        results = asyncio.run(run_concurrently(tasks, cfg))
        assert results == [0, 1, 2]
        assert call_counts[0] == 2  # retried once


# ---------------------------------------------------------------------------
# Integration: run_eval_async with concurrency config
# ---------------------------------------------------------------------------

class TestEvalAsyncConcurrency:
    def test_async_eval_with_concurrency_config(self, tmp_path):
        from prompttest.core.eval_runner import run_eval_async
        from prompttest.core.models import PromptConfig

        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="Q: {{question}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\n"
            "scoring: contains\n"
            "tests:\n"
            "  - input:\n"
            "      question: hello\n"
            "    expected: hello\n"
            "  - input:\n"
            "      question: world\n"
            "    expected: world\n"
            "  - input:\n"
            "      question: foo\n"
            "    expected: foo\n"
        )

        completed = 0
        def _on_done():
            nonlocal completed
            completed += 1

        cfg = ConcurrencyConfig(max_concurrency=2, max_retries=0)
        result = asyncio.run(
            run_eval_async(
                dataset_yaml, prompt,
                concurrency_config=cfg,
                on_case_complete=_on_done,
            )
        )
        assert result.total == 3
        assert result.passed == 3
        assert completed == 3

    def test_async_eval_default_config(self, tmp_path):
        """Without explicit config, async eval still works (uses defaults)."""
        from prompttest.core.eval_runner import run_eval_async
        from prompttest.core.models import PromptConfig

        prompt = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{input}}",
        )
        dataset_yaml = tmp_path / "ds.yaml"
        dataset_yaml.write_text(
            "prompt: test\n"
            "tests:\n"
            "  - input: hello\n"
            "    expected: hello\n"
        )

        result = asyncio.run(run_eval_async(dataset_yaml, prompt))
        assert result.total == 1
        assert result.passed == 1
