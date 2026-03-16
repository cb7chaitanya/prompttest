"""Provider registry — maps provider names to classes and resolves model strings."""

from __future__ import annotations

import importlib

from prompttest.providers.base import LLMProvider

_BUILTIN: dict[str, str] = {
    "openai": "prompttest.providers.openai_provider:OpenAIProvider",
    "anthropic": "prompttest.providers.anthropic_provider:AnthropicProvider",
    "local": "prompttest.providers.local_provider:LocalProvider",
    "echo": "prompttest.providers.echo_provider:EchoProvider",
}

# Model-prefix → provider name.  Order matters: first match wins.
_MODEL_PREFIXES: list[tuple[str, str]] = [
    # OpenAI
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    # Anthropic
    ("claude-", "anthropic"),
    # Local / self-hosted
    ("llama", "local"),
    ("mistral", "local"),
    ("phi", "local"),
    ("qwen", "local"),
    ("gemma", "local"),
]


def get_provider(name: str) -> LLMProvider:
    """Resolve a provider name to an instantiated provider."""
    target = _BUILTIN.get(name)
    if target is None:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {', '.join(sorted(_BUILTIN))}"
        )
    module_path, class_name = target.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def resolve_model(model: str) -> tuple[str, str]:
    """Map a model identifier to ``(provider_name, model_id)``.

    The model string is returned as-is for the model_id; the provider
    is inferred from well-known prefixes.

    Raises ``ValueError`` if no prefix matches.
    """
    lower = model.lower()
    for prefix, provider_name in _MODEL_PREFIXES:
        if lower.startswith(prefix):
            return provider_name, model
    raise ValueError(
        f"Cannot determine provider for model '{model}'. "
        f"Pass --provider explicitly or use a known model name "
        f"(gpt-*, claude-*, llama*, mistral*, etc.)."
    )


def list_providers() -> list[str]:
    """Return sorted list of registered provider names."""
    return sorted(_BUILTIN)
