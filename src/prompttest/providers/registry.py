"""Provider registry — maps provider names to classes."""

from __future__ import annotations

from prompttest.providers.base import LLMProvider

_BUILTIN: dict[str, str] = {
    "openai": "prompttest.providers.openai_provider:OpenAIProvider",
    "anthropic": "prompttest.providers.anthropic_provider:AnthropicProvider",
    "echo": "prompttest.providers.echo_provider:EchoProvider",
}


def get_provider(name: str) -> LLMProvider:
    """Resolve a provider name to an instantiated provider."""
    target = _BUILTIN.get(name)
    if target is None:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {', '.join(sorted(_BUILTIN))}"
        )
    module_path, class_name = target.rsplit(":", 1)
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()
