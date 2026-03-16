"""Load prompts and datasets from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from prompttest.core.models import Dataset, PromptConfig

PROMPTTEST_DIR = ".prompttest"


def project_root() -> Path:
    """Return the .prompttest directory in cwd."""
    return Path.cwd() / PROMPTTEST_DIR


def load_prompt(path: Path) -> PromptConfig:
    data = yaml.safe_load(path.read_text())
    return PromptConfig.from_dict(data)


def load_dataset(path: Path) -> Dataset:
    data = yaml.safe_load(path.read_text())
    return Dataset.from_dict(data)


def discover_prompts(root: Path | None = None) -> list[PromptConfig]:
    root = root or project_root()
    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        return []
    return [load_prompt(p) for p in sorted(prompts_dir.glob("*.yaml"))]


def discover_datasets(root: Path | None = None) -> list[Dataset]:
    root = root or project_root()
    datasets_dir = root / "datasets"
    if not datasets_dir.exists():
        return []
    return [load_dataset(d) for d in sorted(datasets_dir.glob("*.yaml"))]


def find_prompt_by_name(name: str, root: Path | None = None) -> PromptConfig | None:
    for p in discover_prompts(root):
        if p.name == name:
            return p
    return None
