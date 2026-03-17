"""Tests for prompttest.pipeline — targets and runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prompttest.core.eval_runner import EvalCase, EvalDataset, EvalResult
from prompttest.core.models import PromptConfig, Verdict
from prompttest.pipeline.runner import evaluate, evaluate_async
from prompttest.pipeline.targets import (
    CallableTarget,
    EvalTarget,
    HttpTarget,
    PromptTarget,
)


# ---------------------------------------------------------------------------
# CallableTarget
# ---------------------------------------------------------------------------

class TestCallableTarget:
    def test_sync_function(self):
        def echo(inputs: dict[str, str]) -> str:
            return inputs.get("q", "")

        target = CallableTarget(echo, name="echo_fn")
        assert target.name == "echo_fn"
        assert target.call({"q": "hello"}) == "hello"

    def test_name_from_function(self):
        def my_agent(inputs):
            return "ok"

        target = CallableTarget(my_agent)
        assert target.name == "my_agent"

    def test_async_function(self):
        async def async_echo(inputs: dict[str, str]) -> str:
            return inputs.get("q", "")

        target = CallableTarget(async_echo, name="async_fn")
        result = asyncio.run(target.acall({"q": "hello"}))
        assert result == "hello"

    def test_sync_function_via_acall(self):
        def echo(inputs):
            return inputs["q"]

        target = CallableTarget(echo)
        result = asyncio.run(target.acall({"q": "world"}))
        assert result == "world"

    def test_async_function_sync_call_raises(self):
        async def async_fn(inputs):
            return "ok"

        target = CallableTarget(async_fn)
        with pytest.raises(RuntimeError, match="Cannot call async function"):
            target.call({"q": "x"})

    def test_version(self):
        target = CallableTarget(lambda x: "ok", version="3")
        assert target.version == "3"

    def test_return_value_stringified(self):
        def returns_int(inputs):
            return 42

        target = CallableTarget(returns_int)
        assert target.call({}) == "42"


# ---------------------------------------------------------------------------
# PromptTarget
# ---------------------------------------------------------------------------

class TestPromptTarget:
    def test_wraps_prompt_config(self):
        config = PromptConfig(
            name="support", version="2", model="echo", provider="echo",
            system="", template="Q: {{q}}",
        )
        target = PromptTarget(config)
        assert target.name == "support"
        assert target.version == "2"

        # Echo provider returns the user message as-is
        output = target.call({"q": "hello"})
        assert output == "Q: hello"

    def test_async_call(self):
        config = PromptConfig(
            name="test", version="1", model="echo", provider="echo",
            system="", template="{{input}}",
        )
        target = PromptTarget(config)
        result = asyncio.run(target.acall({"input": "async test"}))
        assert result == "async test"


# ---------------------------------------------------------------------------
# HttpTarget
# ---------------------------------------------------------------------------

class TestHttpTarget:
    def test_properties(self):
        target = HttpTarget("http://localhost:8000/chat", name="my_api", version="2")
        assert target.name == "my_api"
        assert target.version == "2"

    def test_name_defaults_to_endpoint(self):
        target = HttpTarget("http://localhost:8000/chat")
        assert target.name == "http://localhost:8000/chat"

    def _mock_httpx(self, response_data):
        """Create a mock httpx module returning *response_data* from post()."""
        import sys
        mock_mod = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()
        mock_mod.post.return_value = mock_response
        return mock_mod

    def test_sync_call(self):
        mock_httpx = self._mock_httpx({"output": "hello world"})
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            target = HttpTarget("http://localhost:8000/chat")
            result = target.call({"q": "hi"})

        assert result == "hello world"
        mock_httpx.post.assert_called_once()
        call_kwargs = mock_httpx.post.call_args
        assert call_kwargs[1]["json"] == {"q": "hi"}

    def test_custom_response_key(self):
        mock_httpx = self._mock_httpx({"answer": "42"})
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            target = HttpTarget("http://x", response_key="answer")
            result = target.call({})

        assert result == "42"

    def test_string_response(self):
        mock_httpx = self._mock_httpx("plain string")
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            target = HttpTarget("http://x")
            result = target.call({})

        assert result == "plain string"

    def test_custom_headers(self):
        mock_httpx = self._mock_httpx({"output": "ok"})
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            target = HttpTarget("http://x", headers={"Authorization": "Bearer tok"})
            target.call({})

        call_kwargs = mock_httpx.post.call_args
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer tok"


# ---------------------------------------------------------------------------
# evaluate() with CallableTarget
# ---------------------------------------------------------------------------

class TestEvaluateCallable:
    def _write_dataset(self, tmp_path: Path) -> Path:
        yaml = (
            "prompt: test\n"
            "scoring: contains\n"
            "tests:\n"
            "  - input:\n"
            "      q: hello world\n"
            "    expected: hello\n"
            "  - input:\n"
            "      q: goodbye\n"
            "    expected: NOPE\n"
        )
        p = tmp_path / "ds.yaml"
        p.write_text(yaml)
        return p

    def test_sync_eval(self, tmp_path: Path):
        def echo(inputs):
            return inputs["q"]

        target = CallableTarget(echo, name="echo_test")
        result = evaluate(target, self._write_dataset(tmp_path))

        assert isinstance(result, EvalResult)
        assert result.prompt_name == "echo_test"
        assert result.total == 2
        assert result.passed == 1  # "hello world" contains "hello"
        assert result.failed == 1  # "goodbye" does not contain "NOPE"

    def test_async_eval(self, tmp_path: Path):
        async def async_echo(inputs):
            return inputs["q"]

        target = CallableTarget(async_echo, name="async_test")
        result = asyncio.run(evaluate_async(target, self._write_dataset(tmp_path)))

        assert result.total == 2
        assert result.passed == 1

    def test_scorer_override(self, tmp_path: Path):
        def echo(inputs):
            return inputs["q"]

        target = CallableTarget(echo)
        result = evaluate(target, self._write_dataset(tmp_path), scorer_name="exact")

        # "hello world" != "hello" with exact scorer
        assert result.case_results[0].verdict == Verdict.FAIL

    def test_tag_filtering(self, tmp_path: Path):
        yaml = (
            "prompt: test\n"
            "scoring: contains\n"
            "tests:\n"
            "  - input:\n"
            "      q: hello\n"
            "    expected: hello\n"
            "    tags: [smoke]\n"
            "  - input:\n"
            "      q: bye\n"
            "    expected: NOPE\n"
            "    tags: [regression]\n"
        )
        p = tmp_path / "ds.yaml"
        p.write_text(yaml)

        target = CallableTarget(lambda i: i["q"])
        result = evaluate(target, p, tags=["smoke"])
        assert result.total == 1
        assert result.passed == 1

    def test_pass_threshold(self, tmp_path: Path):
        yaml = (
            "prompt: test\n"
            "scoring: fuzzy\n"
            "tests:\n"
            "  - input:\n"
            "      q: hello world\n"
            "    expected: hello worl\n"
        )
        p = tmp_path / "ds.yaml"
        p.write_text(yaml)

        target = CallableTarget(lambda i: i["q"])

        result_strict = evaluate(target, p, pass_threshold=1.0)
        assert result_strict.case_results[0].verdict == Verdict.FAIL

        result_lenient = evaluate(target, p, pass_threshold=0.5)
        assert result_lenient.case_results[0].verdict == Verdict.PASS

    def test_exception_becomes_error(self, tmp_path: Path):
        def failing(inputs):
            raise RuntimeError("boom")

        yaml = (
            "prompt: test\n"
            "tests:\n"
            "  - input:\n"
            "      q: x\n"
            "    expected: y\n"
        )
        p = tmp_path / "ds.yaml"
        p.write_text(yaml)

        target = CallableTarget(failing)
        result = evaluate(target, p)
        assert result.errors == 1
        assert result.case_results[0].verdict == Verdict.ERROR

    def test_dataset_as_object(self):
        ds = EvalDataset(
            prompt="test",
            scoring="contains",
            tests=[EvalCase(input={"q": "hello"}, expected="hello")],
        )
        target = CallableTarget(lambda i: i["q"])
        result = evaluate(target, ds)
        assert result.total == 1
        assert result.passed == 1

    def test_dataset_as_string_path(self, tmp_path: Path):
        p = tmp_path / "ds.yaml"
        p.write_text(
            "prompt: test\ntests:\n"
            "  - input:\n      q: hi\n    expected: hi\n"
        )
        target = CallableTarget(lambda i: i["q"])
        result = evaluate(target, str(p))
        assert result.passed == 1
