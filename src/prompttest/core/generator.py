"""LLM-powered dataset generation for evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

_GENERATION_TYPES = {
    "edge": (
        "Generate edge case test inputs that test boundary conditions, unusual formats, "
        "empty inputs, very long inputs, special characters, and unexpected data types. "
        "These should stress-test the system's robustness."
    ),
    "adversarial": (
        "Generate adversarial test inputs designed to break or confuse the system. "
        "Include prompt injection attempts, misleading context, contradictory information, "
        "off-topic requests, and inputs that try to override system instructions."
    ),
    "paraphrase": (
        "Generate paraphrased variations of typical user inputs. Reword questions "
        "using different vocabulary, sentence structures, levels of formality, and "
        "phrasing styles while preserving the original intent."
    ),
    "domain": (
        "Generate domain-specific test inputs with realistic terminology, jargon, "
        "and scenarios that a real user in this domain would encounter. Cover common "
        "use cases, rare but valid scenarios, and domain-specific edge cases."
    ),
}


def _build_generation_prompt(
    gen_type: str,
    size: int,
    prompt_name: str,
    system_prompt: str,
    template: str,
    input_keys: list[str],
    existing_examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build the LLM prompt for generating test cases."""
    type_instruction = _GENERATION_TYPES.get(gen_type, _GENERATION_TYPES["domain"])

    keys_desc = ", ".join(f'"{k}"' for k in input_keys)
    examples_section = ""
    if existing_examples:
        examples_section = (
            "\n\nHere are some existing test cases for reference:\n"
            + json.dumps(existing_examples[:3], indent=2, ensure_ascii=False)
        )

    return (
        f"You are generating test cases for an LLM prompt evaluation system.\n\n"
        f"The prompt being tested is called '{prompt_name}'.\n"
        f"System prompt: {system_prompt or '(none)'}\n"
        f"User template: {template}\n"
        f"Input fields: {keys_desc}\n"
        f"{examples_section}\n\n"
        f"GENERATION TYPE: {gen_type}\n"
        f"{type_instruction}\n\n"
        f"Generate exactly {size} test cases. Each test case must have:\n"
        f"- \"input\": an object with keys {keys_desc}\n"
        f"- \"expected\": the expected output or a key phrase that should appear in the output\n"
        f"- \"tags\": a list of relevant tags (include \"{gen_type}\" as one tag)\n\n"
        f"Return ONLY a JSON array of test case objects. No other text."
    )


def generate_cases(
    prompt_name: str,
    system_prompt: str,
    template: str,
    input_keys: list[str],
    *,
    gen_type: str = "domain",
    size: int = 10,
    model: str | None = None,
    api_key: str | None = None,
    existing_examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Use an LLM to generate test cases.

    Returns a list of dicts with ``input``, ``expected``, and ``tags`` keys.
    """
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required: pip install prompttest[openai]")

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not set")

    mdl = model or os.environ.get("PROMPTTEST_GENERATE_MODEL", "gpt-4o-mini")

    prompt = _build_generation_prompt(
        gen_type, size, prompt_name, system_prompt, template,
        input_keys, existing_examples,
    )

    client = openai.OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=mdl,
        messages=[
            {"role": "system", "content": "You generate structured test data. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )

    raw = resp.choices[0].message.content or ""
    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    cases = json.loads(raw)
    if not isinstance(cases, list):
        raise ValueError(f"Expected a JSON array, got {type(cases).__name__}")

    # Validate and normalize each case
    validated: list[dict[str, Any]] = []
    for case in cases:
        inp = case.get("input", {})
        if isinstance(inp, str):
            inp = {input_keys[0]: inp} if input_keys else {"input": inp}
        expected = str(case.get("expected", ""))
        case_tags = case.get("tags", [gen_type])
        if gen_type not in case_tags:
            case_tags.append(gen_type)
        validated.append({
            "input": inp,
            "expected": expected,
            "tags": case_tags,
        })

    return validated


def build_dataset_yaml(
    prompt_ref: str,
    cases: list[dict[str, Any]],
    scoring: str = "contains",
) -> str:
    """Build a YAML string for an eval dataset from generated cases."""
    data = {
        "prompt": prompt_ref,
        "scoring": scoring,
        "tests": cases,
    }
    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


def list_generation_types() -> list[str]:
    """Return available generation type names."""
    return sorted(_GENERATION_TYPES)
