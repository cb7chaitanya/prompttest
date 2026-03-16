"""Prompt registry with versioning, lookup, and diff support."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from prompttest.core.models import PromptConfig


@dataclass
class VersionInfo:
    """Parsed version identifier from a prompt filename or metadata."""

    raw: str
    major: int
    minor: int

    def __str__(self) -> str:
        return self.raw


def parse_version(raw: str) -> VersionInfo:
    """Parse a version string like 'v1', 'v2.3', or '1' into VersionInfo.

    Supported formats:
        - 'v1'   → major=1, minor=0
        - 'v2.3' → major=2, minor=3
        - '1'    → major=1, minor=0
        - '1.2'  → major=1, minor=2
    """
    cleaned = raw.lstrip("v")
    parts = cleaned.split(".", maxsplit=1)
    try:
        major = int(parts[0])
    except ValueError:
        major = 0
    minor = 0
    if len(parts) == 2:
        try:
            minor = int(parts[1])
        except ValueError:
            pass
    return VersionInfo(raw=raw, major=major, minor=minor)


def _version_sort_key(info: VersionInfo) -> tuple[int, int]:
    return (info.major, info.minor)


@dataclass
class PromptEntry:
    """A prompt config together with its source path and parsed version."""

    config: PromptConfig
    path: Path
    version_info: VersionInfo


@dataclass
class PromptRegistry:
    """In-memory registry that indexes prompts by name and version."""

    _entries: dict[str, dict[str, PromptEntry]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def register(self, config: PromptConfig, path: Path) -> None:
        """Add a single prompt to the registry."""
        vi = parse_version(config.version)
        entry = PromptEntry(config=config, path=path, version_info=vi)
        self._entries.setdefault(config.name, {})[config.version] = entry

    @classmethod
    def from_directory(cls, prompts_dir: Path) -> PromptRegistry:
        """Scan a directory of YAML prompt files and build a registry."""
        registry = cls()
        if not prompts_dir.is_dir():
            return registry
        for yaml_path in sorted(prompts_dir.glob("*.yaml")):
            data = yaml.safe_load(yaml_path.read_text())
            if data is None:
                continue
            config = PromptConfig.from_dict(data)
            registry.register(config, yaml_path)
        return registry

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """All registered prompt names, sorted."""
        return sorted(self._entries)

    def versions(self, name: str) -> list[str]:
        """Sorted version strings for a given prompt name."""
        bucket = self._entries.get(name, {})
        infos = [(v, parse_version(v)) for v in bucket]
        infos.sort(key=lambda t: _version_sort_key(t[1]))
        return [v for v, _ in infos]

    def get(self, name: str, version: str | None = None) -> PromptEntry | None:
        """Retrieve a specific prompt entry.

        If *version* is ``None``, the latest version is returned.
        Version lookup is flexible: 'v1' matches '1' and vice versa.
        """
        bucket = self._entries.get(name)
        if not bucket:
            return None
        if version is not None:
            entry = bucket.get(version)
            if entry is not None:
                return entry
            # Try alternate forms: v1 ↔ 1
            alt = version.lstrip("v") if version.startswith("v") else f"v{version}"
            return bucket.get(alt)
        # Return latest version
        latest_ver = self.versions(name)[-1]
        return bucket[latest_ver]

    def all_entries(self) -> list[PromptEntry]:
        """Flat list of every registered prompt entry, sorted by name then version."""
        entries: list[PromptEntry] = []
        for name in self.names:
            for ver in self.versions(name):
                entries.append(self._entries[name][ver])
        return entries

    # ------------------------------------------------------------------
    # Diffing
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_to_yaml(config: PromptConfig) -> str:
        """Serialize a PromptConfig to a YAML-like text block for diffing."""
        data = {
            "name": config.name,
            "version": config.version,
            "model": config.model,
            "provider": config.provider,
            "system_prompt": config.system,
            "user_template": config.template,
        }
        if config.parameters:
            data["parameters"] = config.parameters
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def diff(self, name: str, version_a: str, version_b: str) -> str:
        """Return a unified diff between two versions of the same prompt.

        Raises ``KeyError`` if either version is not found.
        """
        entry_a = self.get(name, version_a)
        entry_b = self.get(name, version_b)
        if entry_a is None:
            raise KeyError(f"Prompt '{name}' version '{version_a}' not found")
        if entry_b is None:
            raise KeyError(f"Prompt '{name}' version '{version_b}' not found")

        text_a = self._prompt_to_yaml(entry_a.config).splitlines(keepends=True)
        text_b = self._prompt_to_yaml(entry_b.config).splitlines(keepends=True)

        diff_lines = difflib.unified_diff(
            text_a,
            text_b,
            fromfile=f"{name} ({version_a})",
            tofile=f"{name} ({version_b})",
        )
        return "".join(diff_lines)
