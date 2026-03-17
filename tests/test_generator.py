"""Tests for prompttest.core.generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from prompttest.core.generator import (
    _build_generation_prompt,
    build_dataset_yaml,
    generate_cases,
    list_generation_types,
)


# ---------------------------------------------------------------------------
# _build_generation_prompt
# ---------------------------------------------------------------------------

class TestBuildGenerationPrompt:
    def test_contains_prompt_info(self):
        prompt = _build_generation_prompt(
            "edge", 5, "support", "You help users.", "Q: {{question}}",
            ["question"],
        )
        assert "support" in prompt
        assert "question" in prompt
        assert "edge" in prompt
        assert "5" in prompt

    def test_includes_existing_examples(self):
        examples = [{"input": {"q": "hi"}, "expected": "hello"}]
        prompt = _build_generation_prompt(
            "domain", 3, "bot", "", "{{q}}", ["q"],
            existing_examples=examples,
        )
        assert "hi" in prompt
        assert "hello" in prompt

    def test_unknown_type_falls_back_to_domain(self):
        prompt = _build_generation_prompt(
            "unknown_type", 2, "x", "", "{{q}}", ["q"],
        )
        # Should not crash, falls back to domain instructions
        assert "domain-specific" in prompt.lower() or "realistic" in prompt.lower()

    def test_multiple_input_keys(self):
        prompt = _build_generation_prompt(
            "paraphrase", 3, "bot", "", "{{name}} asks {{question}}",
            ["name", "question"],
        )
        assert '"name"' in prompt
        assert '"question"' in prompt


# ---------------------------------------------------------------------------
# generate_cases (mocked)
# ---------------------------------------------------------------------------

class TestGenerateCases:
    def _mock_openai(self, response_json: list):
        mock_mod = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(response_json)
        mock_mod.OpenAI.return_value.chat.completions.create.return_value = mock_response
        return mock_mod

    def test_generates_cases(self):
        fake_cases = [
            {"input": {"q": "What is X?"}, "expected": "X is ...", "tags": ["domain"]},
            {"input": {"q": "How does Y work?"}, "expected": "Y works by ...", "tags": ["domain"]},
        ]
        mock_mod = self._mock_openai(fake_cases)
        with patch.dict(sys.modules, {"openai": mock_mod}):
            cases = generate_cases(
                "support", "You help.", "Q: {{q}}", ["q"],
                gen_type="domain", size=2, api_key="sk-test",
            )
        assert len(cases) == 2
        assert cases[0]["input"]["q"] == "What is X?"
        assert "domain" in cases[0]["tags"]

    def test_adds_gen_type_tag(self):
        fake_cases = [
            {"input": {"q": "test"}, "expected": "ok", "tags": ["other"]},
        ]
        mock_mod = self._mock_openai(fake_cases)
        with patch.dict(sys.modules, {"openai": mock_mod}):
            cases = generate_cases(
                "x", "", "{{q}}", ["q"],
                gen_type="edge", size=1, api_key="sk-test",
            )
        assert "edge" in cases[0]["tags"]

    def test_handles_markdown_fences(self):
        fake_cases = [{"input": {"q": "a"}, "expected": "b", "tags": []}]
        mock_mod = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "```json\n" + json.dumps(fake_cases) + "\n```"
        )
        mock_mod.OpenAI.return_value.chat.completions.create.return_value = mock_response
        with patch.dict(sys.modules, {"openai": mock_mod}):
            cases = generate_cases(
                "x", "", "{{q}}", ["q"],
                gen_type="domain", size=1, api_key="sk-test",
            )
        assert len(cases) == 1

    def test_string_input_normalized(self):
        fake_cases = [
            {"input": "just a string", "expected": "ok", "tags": []},
        ]
        mock_mod = self._mock_openai(fake_cases)
        with patch.dict(sys.modules, {"openai": mock_mod}):
            cases = generate_cases(
                "x", "", "{{q}}", ["q"],
                gen_type="domain", size=1, api_key="sk-test",
            )
        assert isinstance(cases[0]["input"], dict)
        assert cases[0]["input"]["q"] == "just a string"

    def test_missing_api_key_raises(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_mod}), \
             patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                generate_cases("x", "", "{{q}}", ["q"], api_key="")


# ---------------------------------------------------------------------------
# build_dataset_yaml
# ---------------------------------------------------------------------------

class TestBuildDatasetYaml:
    def test_valid_yaml(self):
        cases = [
            {"input": {"q": "hello"}, "expected": "world", "tags": ["smoke"]},
        ]
        text = build_dataset_yaml("support_v1", cases, scoring="exact")
        data = yaml.safe_load(text)
        assert data["prompt"] == "support_v1"
        assert data["scoring"] == "exact"
        assert len(data["tests"]) == 1
        assert data["tests"][0]["input"]["q"] == "hello"

    def test_multiple_cases(self):
        cases = [
            {"input": {"q": "a"}, "expected": "b", "tags": []},
            {"input": {"q": "c"}, "expected": "d", "tags": ["edge"]},
        ]
        text = build_dataset_yaml("bot_v2", cases)
        data = yaml.safe_load(text)
        assert len(data["tests"]) == 2

    def test_default_scoring(self):
        text = build_dataset_yaml("x", [])
        data = yaml.safe_load(text)
        assert data["scoring"] == "contains"


# ---------------------------------------------------------------------------
# list_generation_types
# ---------------------------------------------------------------------------

class TestListGenerationTypes:
    def test_returns_all_types(self):
        types = list_generation_types()
        assert "edge" in types
        assert "adversarial" in types
        assert "paraphrase" in types
        assert "domain" in types

    def test_sorted(self):
        types = list_generation_types()
        assert types == sorted(types)


# ---------------------------------------------------------------------------
# Integration: generate → write → load round-trip
# ---------------------------------------------------------------------------

class TestGenerateRoundTrip:
    def test_generate_write_load(self, tmp_path: Path):
        from prompttest.core.eval_runner import load_eval_dataset

        cases = [
            {"input": {"question": "What is AI?"}, "expected": "artificial intelligence", "tags": ["domain"]},
            {"input": {"question": "Refund?"}, "expected": "30 days", "tags": ["domain", "billing"]},
        ]
        text = build_dataset_yaml("support_v1", cases, scoring="contains")
        path = tmp_path / "generated.yaml"
        path.write_text(text)

        ds = load_eval_dataset(path)
        assert ds.prompt == "support_v1"
        assert ds.scoring == "contains"
        assert len(ds.tests) == 2
        assert ds.tests[0].input["question"] == "What is AI?"
        assert ds.tests[1].tags == ["domain", "billing"]
