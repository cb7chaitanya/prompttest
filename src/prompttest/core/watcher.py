"""File watcher: poll for changes in prompts/ and datasets/ directories."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class FileSnapshot:
    """Tracks file paths and their modification times."""

    _state: dict[Path, float] = field(default_factory=dict)

    def scan(self, *directories: Path, pattern: str = "*.yaml") -> None:
        """Record current mtime for all matching files."""
        self._state.clear()
        for d in directories:
            if d.is_dir():
                for p in d.glob(pattern):
                    self._state[p] = p.stat().st_mtime

    def diff(self, *directories: Path, pattern: str = "*.yaml") -> list[Path]:
        """Return files that changed, were added, or were removed since last scan."""
        current: dict[Path, float] = {}
        for d in directories:
            if d.is_dir():
                for p in d.glob(pattern):
                    current[p] = p.stat().st_mtime

        changed: list[Path] = []

        # Modified or new files
        for p, mtime in current.items():
            if p not in self._state or self._state[p] != mtime:
                changed.append(p)

        # Deleted files
        for p in self._state:
            if p not in current:
                changed.append(p)

        return changed


def watch_loop(
    directories: list[Path],
    on_change: Callable[[list[Path]], None],
    *,
    interval: float = 1.0,
    debounce: float = 0.5,
    stop_after: int = 0,
) -> None:
    """Poll *directories* for YAML changes and call *on_change* when detected.

    *interval* is seconds between polls.
    *debounce* is seconds to wait after detecting a change before acting
    (coalesces rapid saves).
    *stop_after* limits the number of change callbacks (0 = run forever).
    """
    snapshot = FileSnapshot()
    snapshot.scan(*directories)

    callbacks_fired = 0

    while True:
        time.sleep(interval)

        changed = snapshot.diff(*directories)
        if not changed:
            continue

        # Debounce: wait, then re-diff to coalesce rapid changes
        if debounce > 0:
            time.sleep(debounce)
            changed = snapshot.diff(*directories)

        if changed:
            snapshot.scan(*directories)
            on_change(changed)
            callbacks_fired += 1

            if stop_after > 0 and callbacks_fired >= stop_after:
                return
