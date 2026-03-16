"""Tests for the prompt registry system."""

from __future__ import annotations

from pathlib import Path

from prompttest.core.models import PromptConfig
from prompttest.core.registry import PromptRegistry, parse_version


def test_parse_version_simple() -> None:
    v = parse_version("v1")
    assert v.major == 1
    assert v.minor == 0
    assert v.raw == "v1"


def test_parse_version_with_minor() -> None:
    v = parse_version("v2.3")
    assert v.major == 2
    assert v.minor == 3


def test_parse_version_bare_number() -> None:
    v = parse_version("1")
    assert v.major == 1
    assert v.minor == 0


def test_registry_register_and_get() -> None:
    registry = PromptRegistry()
    cfg = PromptConfig(
        name="support",
        version="v1",
        model="gpt-4o-mini",
        provider="openai",
        system="You are helpful.",
        template="{{input}}",
    )
    registry.register(cfg, Path("support_v1.yaml"))

    entry = registry.get("support", "v1")
    assert entry is not None
    assert entry.config.name == "support"


def test_registry_latest_version() -> None:
    registry = PromptRegistry()
    for ver in ["v1", "v2", "v3"]:
        cfg = PromptConfig(
            name="bot",
            version=ver,
            model="m",
            provider="echo",
            system="s",
            template="t",
        )
        registry.register(cfg, Path(f"bot_{ver}.yaml"))

    latest = registry.get("bot")
    assert latest is not None
    assert latest.config.version == "v3"


def test_registry_versions_sorted() -> None:
    registry = PromptRegistry()
    for ver in ["v3", "v1", "v2"]:
        cfg = PromptConfig(
            name="x", version=ver, model="m", provider="p", system="s", template="t"
        )
        registry.register(cfg, Path(f"x_{ver}.yaml"))

    assert registry.versions("x") == ["v1", "v2", "v3"]


def test_registry_diff() -> None:
    registry = PromptRegistry()
    cfg_a = PromptConfig(
        name="bot", version="v1", model="m", provider="p", system="Hello", template="t"
    )
    cfg_b = PromptConfig(
        name="bot", version="v2", model="m", provider="p", system="Goodbye", template="t"
    )
    registry.register(cfg_a, Path("a.yaml"))
    registry.register(cfg_b, Path("b.yaml"))

    result = registry.diff("bot", "v1", "v2")
    assert "Hello" in result
    assert "Goodbye" in result
    assert "---" in result  # unified diff header


def test_registry_from_directory(tmp_path: Path) -> None:
    prompt_text = """\
name: test_prompt
version: "1"
provider: echo
model: gpt-4o-mini
system: sys
template: "{{input}}"
"""
    (tmp_path / "test_v1.yaml").write_text(prompt_text)

    registry = PromptRegistry.from_directory(tmp_path)
    assert "test_prompt" in registry.names
    entry = registry.get("test_prompt", "1")
    assert entry is not None
    assert entry.config.system == "sys"
