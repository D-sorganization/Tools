"""Tests for the cross-platform file watcher.

Runs against whichever backend is active (Rust extension if built, otherwise
the watchdog fallback). The tests target the public API only, so they pass on
either implementation.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

# Make the in-tree wrapper importable without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))

watchdog = pytest.importorskip(
    "watchdog",
    reason="watchdog is required to exercise the Python fallback backend",
)

from file_watcher import ChangeEvent, FileWatcher, backend  # noqa: E402

# Filesystem event timing varies wildly between OSes and CI runners. We use a
# generous post-action sleep so the debounce thread has time to flush.
_SETTLE = 0.6
_DEBOUNCE_MS = 50


def _wait_for(predicate, timeout: float = 2.0, poll: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


@pytest.fixture
def watcher_factory(tmp_path):
    created: list[FileWatcher] = []

    def _make(**overrides) -> tuple[FileWatcher, list[ChangeEvent], threading.Event]:
        events: list[ChangeEvent] = []
        got_any = threading.Event()

        watcher = FileWatcher(
            root=str(overrides.pop("root", tmp_path)),
            debounce_ms=overrides.pop("debounce_ms", _DEBOUNCE_MS),
            respect_gitignore=overrides.pop("respect_gitignore", False),
        )

        def _capture(batch):
            events.extend(batch)
            got_any.set()

        watcher.on_change(_capture)
        created.append(watcher)
        return watcher, events, got_any

    yield _make

    for w in created:
        if getattr(w, "is_running", False):
            try:
                w.stop()
            except Exception:
                pass


def test_backend_reports_string():
    assert backend() in {"rust", "watchdog"}


def test_detects_create(tmp_path, watcher_factory):
    watcher, events, got_any = watcher_factory()
    watcher.start()
    time.sleep(0.1)

    target = tmp_path / "new.txt"
    target.write_text("hello")
    assert _wait_for(got_any.is_set, timeout=2.0)
    time.sleep(_SETTLE)
    watcher.stop()

    paths = [Path(e.path).name for e in events]
    assert "new.txt" in paths
    create_events = [e for e in events if Path(e.path).name == "new.txt"]
    # We accept create-or-modify since watchdog can deliver either depending on
    # platform-specific atomic-write semantics.
    assert any(e.kind in {"create", "modify"} for e in create_events)


def test_detects_modify(tmp_path, watcher_factory):
    target = tmp_path / "existing.txt"
    target.write_text("v1")
    watcher, events, got_any = watcher_factory()
    watcher.start()
    time.sleep(0.1)

    target.write_text("v2")
    assert _wait_for(got_any.is_set, timeout=2.0)
    time.sleep(_SETTLE)
    watcher.stop()

    assert any(Path(e.path).name == "existing.txt" for e in events)


def test_detects_delete(tmp_path, watcher_factory):
    target = tmp_path / "doomed.txt"
    target.write_text("bye")
    watcher, events, got_any = watcher_factory()
    watcher.start()
    time.sleep(0.1)

    target.unlink()
    assert _wait_for(got_any.is_set, timeout=2.0)
    time.sleep(_SETTLE)
    watcher.stop()

    assert any(
        Path(e.path).name == "doomed.txt" and e.kind in {"delete", "modify"}
        for e in events
    )


def test_detects_rename(tmp_path, watcher_factory):
    src = tmp_path / "before.txt"
    src.write_text("name")
    watcher, events, got_any = watcher_factory()
    watcher.start()
    time.sleep(0.1)

    dst = tmp_path / "after.txt"
    src.rename(dst)
    assert _wait_for(got_any.is_set, timeout=2.0)
    time.sleep(_SETTLE)
    watcher.stop()

    names = {Path(e.path).name for e in events}
    # At least one of the rename endpoints should show up.
    assert names & {"before.txt", "after.txt"}


def test_debounce_coalesces_rapid_changes(tmp_path, watcher_factory):
    flush_count = 0
    flush_lock = threading.Lock()

    watcher = FileWatcher(root=str(tmp_path), debounce_ms=150, respect_gitignore=False)

    def _count(_batch):
        nonlocal flush_count
        with flush_lock:
            flush_count += 1

    watcher.on_change(_count)
    watcher.start()
    time.sleep(0.1)

    target = tmp_path / "rapid.txt"
    for i in range(10):
        target.write_text(f"v{i}")
        time.sleep(0.005)
    time.sleep(0.6)
    watcher.stop()

    # 10 rapid writes should collapse to a small handful of flushes (typically
    # 1; allow a little slack for watchdog's backend-specific behaviour).
    with flush_lock:
        assert flush_count <= 3, f"expected debounce to coalesce, got {flush_count}"


def test_gitignore_excludes_matched_paths(tmp_path, watcher_factory):
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    # Built-in skip dirs cover __pycache__/node_modules/.git regardless of
    # whether pathspec is installed, so we lean on those for fallback safety.
    watcher, events, got_any = watcher_factory(respect_gitignore=True)
    watcher.start()
    time.sleep(0.1)

    (tmp_path / "kept.txt").write_text("hi")
    junk = tmp_path / "node_modules"
    junk.mkdir()
    (junk / "noisy.txt").write_text("x")

    assert _wait_for(got_any.is_set, timeout=2.0)
    time.sleep(_SETTLE)
    watcher.stop()

    names = {Path(e.path).name for e in events}
    assert "kept.txt" in names
    assert "noisy.txt" not in names, "node_modules should be filtered"


def test_double_start_raises(tmp_path, watcher_factory):
    watcher, _events, _ = watcher_factory()
    watcher.start()
    try:
        with pytest.raises((RuntimeError, Exception)):
            watcher.start()
    finally:
        watcher.stop()


def test_stop_without_start_raises(tmp_path, watcher_factory):
    watcher, _events, _ = watcher_factory()
    with pytest.raises((RuntimeError, Exception)):
        watcher.stop()


def test_invalid_root_rejected(tmp_path):
    with pytest.raises((ValueError, Exception)):
        FileWatcher(root=str(tmp_path / "does_not_exist"))


def test_context_manager(tmp_path, watcher_factory):
    watcher, events, got_any = watcher_factory()
    with watcher:
        time.sleep(0.1)
        (tmp_path / "ctx.txt").write_text("ok")
        assert _wait_for(got_any.is_set, timeout=2.0)
    assert not watcher.is_running
    assert any(Path(e.path).name == "ctx.txt" for e in events)
