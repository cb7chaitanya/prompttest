"""Tests for prompttest.core.watcher."""

from __future__ import annotations

import time
from pathlib import Path

from prompttest.core.watcher import FileSnapshot, watch_loop


# ---------------------------------------------------------------------------
# FileSnapshot
# ---------------------------------------------------------------------------

class TestFileSnapshot:
    def test_scan_records_files(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text("a")
        (tmp_path / "b.yaml").write_text("b")

        snap = FileSnapshot()
        snap.scan(tmp_path)
        assert len(snap._state) == 2

    def test_scan_ignores_non_yaml(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        snap = FileSnapshot()
        snap.scan(tmp_path)
        assert len(snap._state) == 1

    def test_diff_detects_modification(self, tmp_path: Path):
        f = tmp_path / "a.yaml"
        f.write_text("v1")

        snap = FileSnapshot()
        snap.scan(tmp_path)

        time.sleep(0.05)
        f.write_text("v2")

        changed = snap.diff(tmp_path)
        assert len(changed) == 1
        assert changed[0] == f

    def test_diff_detects_new_file(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text("a")

        snap = FileSnapshot()
        snap.scan(tmp_path)

        (tmp_path / "b.yaml").write_text("b")

        changed = snap.diff(tmp_path)
        assert any(p.name == "b.yaml" for p in changed)

    def test_diff_detects_deleted_file(self, tmp_path: Path):
        f = tmp_path / "a.yaml"
        f.write_text("a")

        snap = FileSnapshot()
        snap.scan(tmp_path)

        f.unlink()

        changed = snap.diff(tmp_path)
        assert len(changed) == 1
        assert changed[0] == f

    def test_diff_empty_when_unchanged(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text("a")

        snap = FileSnapshot()
        snap.scan(tmp_path)

        changed = snap.diff(tmp_path)
        assert changed == []

    def test_scan_clears_previous_state(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text("a")

        snap = FileSnapshot()
        snap.scan(tmp_path)
        assert len(snap._state) == 1

        (tmp_path / "a.yaml").unlink()
        snap.scan(tmp_path)
        assert len(snap._state) == 0

    def test_multiple_directories(self, tmp_path: Path):
        d1 = tmp_path / "prompts"
        d2 = tmp_path / "datasets"
        d1.mkdir()
        d2.mkdir()
        (d1 / "p.yaml").write_text("p")
        (d2 / "d.yaml").write_text("d")

        snap = FileSnapshot()
        snap.scan(d1, d2)
        assert len(snap._state) == 2

    def test_nonexistent_directory_ignored(self, tmp_path: Path):
        snap = FileSnapshot()
        snap.scan(tmp_path / "nonexistent")
        assert len(snap._state) == 0


# ---------------------------------------------------------------------------
# watch_loop
# ---------------------------------------------------------------------------

class TestWatchLoop:
    def test_fires_callback_on_change(self, tmp_path: Path):
        f = tmp_path / "a.yaml"
        f.write_text("v1")

        results = []

        def on_change(changed):
            results.append([p.name for p in changed])
            # Modify again so we can test stop_after
            time.sleep(0.05)
            f.write_text("v3")

        # Trigger change after initial scan
        import threading

        def delayed_write():
            time.sleep(0.3)
            f.write_text("v2")

        t = threading.Thread(target=delayed_write)
        t.start()

        watch_loop(
            [tmp_path],
            on_change,
            interval=0.1,
            debounce=0.1,
            stop_after=1,
        )

        t.join()
        assert len(results) == 1
        assert "a.yaml" in results[0]

    def test_stop_after_limits_callbacks(self, tmp_path: Path):
        f = tmp_path / "a.yaml"
        f.write_text("v1")

        count = 0

        def on_change(changed):
            nonlocal count
            count += 1
            time.sleep(0.05)
            f.write_text(f"v{count + 2}")

        import threading

        def writer():
            for i in range(5):
                time.sleep(0.2)
                f.write_text(f"v{i}")

        t = threading.Thread(target=writer)
        t.start()

        watch_loop(
            [tmp_path],
            on_change,
            interval=0.1,
            debounce=0.05,
            stop_after=2,
        )

        t.join()
        assert count == 2
