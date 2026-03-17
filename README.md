<p align="center">
  <h1 align="center">prompttest</h1>
  <p align="center">
    <strong>Catch prompt regressions before they hit production.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="#how-it-works">How It Works</a> &middot;
    <a href="#commands">Commands</a> &middot;
    <a href="#scoring">Scoring</a> &middot;
    <a href="#cicd">CI/CD</a>
  </p>
</p>

---

A testing framework for LLM prompts. Define expected outputs, run evaluations against any provider, and detect regressions when prompts change.

```
You "improved" your prompt.
prompttest caught 3 silent regressions in 2 seconds.
```

```
  prompttest diff-prompts summarizer v1 v2

  --- summarizer (v1)
  +++ summarizer (v2)

  system_prompt:
  - You are a precise summarizer. Always include specific numbers, names, and key facts.
  + You are a helpful assistant. Provide a brief high-level overview.

  user_template:
  - Summarize this text concisely:
  + Give a short overview:
```

```
  prompttest eval summarizer-eval.yaml --baseline v1-results.json

    #   Input                          Expected       Score   Result
  ──────────────────────────────────────────────────────────────────────
    1   Apple Q3 revenue report         $81.8B          1.00   PASS
    2   NASA Artemis II mission         Artemis         0.00   FAIL
    3   EU AI regulation                AI Act          1.00   PASS
    4*  Tesla Q3 deliveries             466,140         0.00   FAIL   CRITICAL
    5   Spotify subscriber growth       230 million     0.00   FAIL

  Results       2/5 passed | Accuracy 40%
  Avg Score     0.40 (threshold: 0.70)
  Regressions   3 detected (1 critical)

  PROMPT DEGRADED - DO NOT SHIP
```

---

## Install

```bash
pip install -e ".[dev]"

# Optional
pip install -e ".[openai]"       # OpenAI provider
pip install -e ".[anthropic]"    # Anthropic provider
pip install -e ".[fuzzy]"        # Faster fuzzy matching
pip install -e ".[report]"       # HTML reports
```

## Quick Start

```bash
prompttest init          # Scaffold project with examples
prompttest run           # Run all tests
prompttest eval dataset.yaml   # Full evaluation
```

This creates:

```
.prompttest/
  prompts/     Versioned prompt templates (YAML)
  datasets/    Test cases with expected outputs (YAML)
  results/     Saved evaluation results
  history/     Run history (auto-recorded)
```

## How It Works

**1. Define a prompt**

```yaml
# .prompttest/prompts/summarizer.yaml
name: summarizer
version: "1"
provider: openai
model: gpt-4o
system: You are a precise summarizer. Include specific numbers and key facts.
template: "Summarize this text:\n\n{{text}}"
parameters:
  temperature: 0.3
```

**2. Write test cases**

```yaml
# .prompttest/datasets/summarizer-eval.yaml
prompt: summarizer
scoring: contains
tests:
  - input:
      text: "Apple reported Q3 revenue of $81.8B, beating estimates."
    expected: "$81.8B"
    tags: [financial]
    critical: true

  - input:
      text: "NASA's Artemis II will send astronauts around the Moon."
    expected: "Artemis"
    tags: [science]
```

**3. Run evaluation**

```bash
prompttest eval .prompttest/datasets/summarizer-eval.yaml
```

**4. Catch regressions**

```bash
# Save baseline
prompttest eval dataset.yaml --output baseline.json

# After changing the prompt, compare
prompttest eval dataset.yaml --baseline baseline.json
```

## Commands

| Command | Description |
|---|---|
| `prompttest init` | Scaffold project with example files |
| `prompttest run` | Run all datasets against linked prompts |
| `prompttest eval <dataset>` | Full evaluation with all options |
| `prompttest eval-pipeline <dataset>` | Evaluate HTTP endpoints |
| `prompttest generate <output>` | Generate test cases with LLM |
| `prompttest watch` | Hot-reload on file changes |
| `prompttest history` | Show run history with trends |
| `prompttest list-prompts` | List registered prompts |
| `prompttest show-prompt <name>` | Show prompt details |
| `prompttest diff-prompts <name> <v1> <v2>` | Diff prompt versions |

### `eval` flags

```bash
# Model & provider
--model gpt-4o                    # Override model (auto-detects provider)
--provider openai                 # Override provider

# Scoring
--scorer fuzzy                    # Scoring function
--pass-threshold 0.8              # Minimum score to pass (default: 0.7)

# Concurrency
--async                           # Run cases concurrently
--max-concurrency 5               # Max parallel requests
--rate-limit 10                   # Requests per second
--max-retries 3                   # Retry on 429/5xx

# Filtering
--tags billing,critical           # Filter by tags
--match all                       # Require all tags (default: any)

# Output
--output results.json             # Save as JSON or CSV
--output-dir ./results            # Auto-timestamped files
--report report.html              # HTML report
--baseline baseline.json          # Compare against baseline

# CI/CD
--fail-on-threshold               # Exit 1 if avg score < threshold
--fail-on-critical                # Exit 1 if any critical test fails

# Validation
--no-strict                       # Warn instead of fail on placeholder mismatch
--skip-key-check                  # Skip API key validation

# Debug
--explain                         # LLM-powered failure explanations
```

### Generate test cases

```bash
prompttest generate tests.yaml --prompt summarizer --type edge --size 50
prompttest generate tests.yaml --prompt summarizer --type adversarial --size 20
prompttest generate tests.yaml --prompt summarizer --type paraphrase --size 30
prompttest generate tests.yaml --prompt summarizer --type domain --size 10
```

### Watch mode

```bash
prompttest watch                  # Re-run on file changes
prompttest watch --interval 2     # Poll every 2s
```

### Pipeline evaluation

```bash
# Test an HTTP endpoint
prompttest eval-pipeline dataset.yaml --endpoint http://localhost:8000/chat
```

```python
# Test a Python function
from prompttest.pipeline import evaluate, CallableTarget

def my_agent(inputs):
    return f"Answer: {inputs['question']}"

result = evaluate(CallableTarget(my_agent), "dataset.yaml")
```

## Scoring

8 built-in scorers, all pluggable:

| Scorer | Type | Description |
|---|---|---|
| `contains` | Binary | Expected text appears in output **(default)** |
| `exact` | Binary | Normalized exact match |
| `starts_with` | Binary | Output starts with expected |
| `ends_with` | Binary | Output ends with expected |
| `regex` | Binary | Expected field is a regex pattern |
| `fuzzy` | 0.0-1.0 | Character-level similarity |
| `semantic` | 0.0-1.0 | OpenAI embeddings + cosine similarity |
| `llm_judge` | 0.0-1.0 | LLM scores with reasoning |

**Custom scorers:**

```python
from prompttest.core.scoring import register_scorer

def my_scorer(output: str, expected: str) -> tuple[float, str]:
    score = ...  # 0.0 to 1.0
    return score, "reason"

register_scorer("my_scorer", my_scorer)
```

## Providers

| Provider | Env Var | Models |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `gpt-4o`, `gpt-4o-mini`, `o1`, ... |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-*` |
| `local` | `LOCAL_MODEL_URL` | Ollama, vLLM, llama.cpp |
| `echo` | - | Returns input as-is (testing) |

Model auto-detection: `--model gpt-4o` automatically selects the `openai` provider.

## Critical Tests

Mark must-pass test cases:

```yaml
tests:
  - input:
      question: "What is your refund policy?"
    expected: "30 days"
    critical: true
```

```bash
prompttest eval dataset.yaml --fail-on-critical
```

## Output & Reports

```bash
# JSON / CSV
prompttest eval dataset.yaml --output results.json
prompttest eval dataset.yaml --output results.csv

# Auto-timestamped
prompttest eval dataset.yaml --output-dir ./results

# HTML report (shareable)
prompttest eval dataset.yaml --report report.html

# Baseline comparison
prompttest eval dataset.yaml --baseline previous.json

# Run history
prompttest history
prompttest history --prompt summarizer --limit 10
```

<h2 id="cicd">CI/CD</h2>

prompttest exits with code 1 on failures, making it CI-ready out of the box.

| Condition | Exit code |
|---|---|
| All tests pass | 0 |
| Any test case fails | 1 |
| `--fail-on-threshold` and avg score < threshold | 1 |
| `--fail-on-critical` and any critical test fails | 1 |
| `--baseline` and regression detected | 1 |

### GitHub Actions

```yaml
name: Prompt Evaluation
on:
  pull_request:
  push:
    branches: [main]
    paths: [".prompttest/**", "src/**"]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - run: pip install -e ".[openai,dev]"

      - name: Evaluate prompts
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          prompttest eval .prompttest/datasets/eval.yaml \
            --fail-on-threshold \
            --fail-on-critical \
            --output results.json \
            --report report.html

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: eval-results
          path: |
            results.json
            report.html
```

## Development

```bash
make test    # Run tests (255 passing)
make lint    # Run linter
```

## License

MIT
