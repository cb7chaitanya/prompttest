# prompttest

A testing framework for LLM prompts that detects regressions when prompts change.

## Install

```bash
pip install -e ".[dev]"

# Optional providers
pip install -e ".[openai]"       # OpenAI support
pip install -e ".[anthropic]"    # Anthropic support
pip install -e ".[fuzzy]"        # rapidfuzz for fuzzy scoring
```

## Quick Start

```bash
# Initialize a project with example files
prompttest init

# Run all tests
prompttest run

# Run evaluation dataset
prompttest eval .prompttest/datasets/summarize-basics.yaml
```

This creates a `.prompttest/` directory with:

```
.prompttest/
  prompts/     # Versioned prompt templates (YAML)
  datasets/    # Test cases with expected outputs (YAML)
  results/     # Saved evaluation results
```

## Prompt Format

```yaml
name: support
version: "1"
provider: openai
model: gpt-4o-mini
system: You are a helpful support agent.
template: "Answer this question: {{question}}"
parameters:
  temperature: 0.3
```

## Dataset Format

```yaml
prompt: support_v1
scoring: contains
tests:
  - input:
      question: "What is your refund policy?"
    expected: "30 days"
    tags: [billing, critical]
  - input:
      question: "Do you offer support?"
    expected: "Yes"
    tags: [general]
```

## Commands

### `prompttest init [directory]`

Create a new project with example prompts and datasets.

### `prompttest run`

Run all datasets against their linked prompts.

### `prompttest eval <dataset.yaml>`

Run an evaluation dataset with full control over scoring, concurrency, and output.

```bash
# Basic
prompttest eval dataset.yaml

# Override model (auto-detects provider)
prompttest eval dataset.yaml --model gpt-4o

# Override scorer and pass threshold
prompttest eval dataset.yaml --scorer fuzzy --pass-threshold 0.8

# Async with concurrency control
prompttest eval dataset.yaml --async --max-concurrency 5 --rate-limit 10

# Filter by tags
prompttest eval dataset.yaml --tags billing,critical
prompttest eval dataset.yaml --tags billing,critical --match all

# Save results
prompttest eval dataset.yaml --output results.json
prompttest eval dataset.yaml --output results.csv
prompttest eval dataset.yaml --output-dir ./results

# Skip validation
prompttest eval dataset.yaml --no-strict --skip-key-check
```

### `prompttest list-prompts`

Show all registered prompts and their versions.

### `prompttest show-prompt <name>`

Display details of a specific prompt. Supports `name` (latest) or `name_vN`.

### `prompttest diff-prompts <name> <v1> <v2>`

Show a unified diff between two prompt versions.

```bash
prompttest diff-prompts summarize v1 v2
prompttest diff-prompts summarize v1 v2 --output diff.json
```

## Scoring

Scorers evaluate LLM output against expected values. All return a float score (0.0–1.0).

| Scorer | Type | Description |
|---|---|---|
| `contains` | Binary | Expected text appears in output (default) |
| `exact` | Binary | Normalized strings match exactly |
| `starts_with` | Binary | Output starts with expected text |
| `ends_with` | Binary | Output ends with expected text |
| `regex` | Binary | Expected field is a regex pattern |
| `fuzzy` | Continuous | Character-level similarity (rapidfuzz or SequenceMatcher) |
| `semantic` | Continuous | Cosine similarity via OpenAI embeddings |
| `llm_judge` | Continuous | LLM scores output 0.0–1.0 with reasoning |

The pass threshold defaults to `0.7` and can be set with `--pass-threshold`.

### Custom Scorers

```python
from prompttest.core.scoring import register_scorer

def my_scorer(output: str, expected: str) -> tuple[float, str]:
    score = ...  # 0.0 to 1.0
    return score, "reason"

register_scorer("my_scorer", my_scorer)
```

## Providers

| Provider | Env Var | Description |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | OpenAI API (GPT-4o, etc.) |
| `anthropic` | `ANTHROPIC_API_KEY` | Anthropic API (Claude) |
| `local` | `LOCAL_MODEL_URL` | OpenAI-compatible endpoint (Ollama, vLLM) |
| `echo` | — | Returns input as-is (testing) |

API keys are validated before evaluation. Use `--skip-key-check` to bypass.

## Output Formats

Results can be saved as JSON or CSV with `--output` or `--output-dir`.

JSON output includes metadata (timestamp, model, provider), summary (total, passed, accuracy, average score, pass threshold), and per-case results.

```bash
# Explicit file
prompttest eval dataset.yaml --output results.json

# Auto-timestamped in directory
prompttest eval dataset.yaml --output-dir ./results
# → ./results/support_v1_20260317T120000Z.json
```

## Using prompttest in CI/CD

prompttest is designed for CI pipelines. The `eval` command exits with a non-zero status code when tests fail, and `--fail-on-threshold` adds average-score gating.

### Exit codes

| Condition | Exit code |
|---|---|
| All tests pass | 0 |
| Any test case fails or errors | 1 |
| `--fail-on-threshold` and average score < threshold | 1 |

### GitHub Actions

An example workflow is included at `.github/workflows/prompt-eval.yml`:

```yaml
name: Prompt Evaluation

on:
  pull_request:
  push:
    branches: [main]
    paths:
      - ".prompttest/**"
      - "src/**"

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install prompttest
        run: pip install -e ".[dev]"

      - name: Initialize project
        run: prompttest init

      - name: Run prompt evaluation
        run: |
          prompttest eval .prompttest/datasets/summarize-basics.yaml \
            --pass-threshold 0.7 \
            --fail-on-threshold \
            --skip-key-check \
            --output results.json

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: results.json
```

> **Note:** The example above uses the built-in `echo` provider (created by `prompttest init`) which needs no API key. For real LLM evaluation, install the provider extra (`pip install -e ".[openai,dev]"`), set the API key secret, and remove `--skip-key-check`.

### Tips for CI

- Use `--fail-on-threshold` to gate on average score, not just individual failures.
- Use `--output results.json` to save artifacts for debugging failed runs.
- Use `--tags critical` to run only high-priority tests on every PR.
- Use `--skip-key-check` if key presence is validated elsewhere.
- Set `--pass-threshold` per environment (e.g. stricter in production).

## Development

```bash
make test    # Run tests
make lint    # Run linter
```

## License

MIT
