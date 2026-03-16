"""Integration test: init then run using the echo provider."""

import subprocess
import sys
from pathlib import Path


def test_init_and_run(tmp_path: Path):
    """End-to-end: init creates files, run executes with echo provider."""
    # init
    result = subprocess.run(
        [sys.executable, "-m", "prompttest.cli.main", "init", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / ".prompttest" / "prompts" / "summarize.yaml").exists()
    assert (tmp_path / ".prompttest" / "datasets" / "summarize-basics.yaml").exists()

    # run (echo provider — no API key needed, but 'contains' evaluator will pass
    # because the input text is echoed back and contains the expected substring)
    result = subprocess.run(
        [sys.executable, "-m", "prompttest.cli.main", "run", "--dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "PASS" in result.stdout or "pass" in result.stdout.lower()
