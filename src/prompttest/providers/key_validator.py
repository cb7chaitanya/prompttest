"""API key validation for LLM providers."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Maps provider name → (env var name, optional package needed)
_PROVIDER_KEYS: dict[str, tuple[str, str | None]] = {
    "openai": ("OPENAI_API_KEY", "openai"),
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic"),
}


@dataclass
class KeyCheckResult:
    """Result of validating a single provider's API key."""

    provider: str
    env_var: str
    present: bool
    valid: bool | None = None  # None = not tested, True/False = live check result
    error: str = ""


def check_key_present(provider: str) -> KeyCheckResult | None:
    """Check whether the required env var is set for *provider*.

    Returns ``None`` for providers that don't need an API key (echo, local).
    """
    entry = _PROVIDER_KEYS.get(provider)
    if entry is None:
        return None

    env_var, _ = entry
    value = os.environ.get(env_var, "").strip()
    return KeyCheckResult(
        provider=provider,
        env_var=env_var,
        present=bool(value),
    )


def check_key_live(provider: str) -> KeyCheckResult | None:
    """Perform a lightweight API call to verify the key is valid.

    Returns ``None`` for providers that don't need a key.
    Falls back to presence-only check if the SDK is not installed.
    """
    result = check_key_present(provider)
    if result is None:
        return None
    if not result.present:
        return result

    env_var = result.env_var
    api_key = os.environ.get(env_var, "")

    if provider == "openai":
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            client.models.list()
            result.valid = True
        except ImportError:
            result.valid = None
            result.error = "openai package not installed"
        except Exception as exc:
            result.valid = False
            result.error = str(exc)

    elif provider == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Anthropic doesn't have a list-models endpoint that's as lightweight,
            # so we send a minimal message request that will fail fast on auth
            # but succeed cheaply otherwise.
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            result.valid = True
        except ImportError:
            result.valid = None
            result.error = "anthropic package not installed"
        except Exception as exc:
            exc_str = str(exc).lower()
            if "authentication" in exc_str or "401" in exc_str or "invalid" in exc_str:
                result.valid = False
                result.error = str(exc)
            else:
                # Non-auth errors (rate limit, network) mean the key itself is fine
                result.valid = True

    return result


def validate_provider_key(
    provider: str,
    *,
    live: bool = False,
) -> KeyCheckResult | None:
    """Validate the API key for *provider*.

    When *live* is ``True``, a lightweight API call is made to verify the key.
    Returns ``None`` for providers that don't require keys.
    """
    if live:
        return check_key_live(provider)
    return check_key_present(provider)
