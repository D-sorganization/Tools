"""Unit tests for chat session-file helpers.

These helpers (_session_file_path, _read_shared_session_id, _write_shared_session_id)
have no PyQt6 dependency at runtime.  The module they live in (chat_dock_widget)
imports PyQt6 at the top, so PyQt6 must be installed for the import to succeed, but
the test logic itself needs only the standard library.

If PyQt6 is not installed in the environment the entire file is skipped via
``pytest.importorskip`` -- same as the original test_chat_dock_widget.py.  The value
of keeping these tests in a separate file is:

1. They are clearly labelled as headless-safe unit tests (no widget / display server
   needed at runtime).
2. Skip debt is visible and documented -- these tests SHOULD be dependency-complete
   once the session helpers are relocated to a pure-Python module (tracked separately).
3. The widget tests in test_chat_dock_widget.py are cleanly isolated from the
   pure-logic tests.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

# chat_dock_widget imports PyQt6 at the module level.  If PyQt6 is not installed
# the import fails.  Skip the file (not hide it) so the skip is visible in CI.
pytest.importorskip("PyQt6", reason="PyQt6 required to import chat_dock_widget")

from chat.chat_dock_widget import (  # noqa: E402
    _read_shared_session_id,
    _session_file_path,
    _write_shared_session_id,
)


@pytest.mark.unit
class TestSessionFileHelpers:
    """Unit tests for session-file persistence helpers.

    These tests exercise pure-Python logic -- no display server, no Qt event loop.
    """

    def test_session_file_path(self) -> None:
        """_session_file_path returns a path ending in active_chat_session.txt."""
        path = _session_file_path("my_app")
        assert path.name == "active_chat_session.txt"
        assert ".my_app" in str(path)

    def test_write_and_read(self, tmp_path: Path) -> None:
        """Written session ID is readable from the same path."""
        path = tmp_path / "session.txt"
        _write_shared_session_id("test-session-123", path)
        assert _read_shared_session_id(path) == "test-session-123"

    def test_read_missing_file(self, tmp_path: Path) -> None:
        """Reading a non-existent path returns None."""
        path = tmp_path / "nonexistent.txt"
        assert _read_shared_session_id(path) is None

    def test_read_empty_file(self, tmp_path: Path) -> None:
        """Reading an empty file returns None."""
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        assert _read_shared_session_id(path) is None

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Write creates intermediate directories as needed."""
        path = tmp_path / "deep" / "nested" / "session.txt"
        _write_shared_session_id("abc", path)
        assert path.exists()
        assert _read_shared_session_id(path) == "abc"

    def test_concurrent_writes_are_atomic(self, tmp_path: Path) -> None:
        """Tools issue #2753: concurrent writers must not corrupt the file.

        Four threads race to write distinct session IDs to the same path.
        With the module-level lock and atomic tmp+replace dance the file
        must contain *exactly one* valid session ID at the end (no torn
        writes, no exceptions, no empty file).
        """
        path = tmp_path / "shared_session.txt"
        candidates = [f"sid{i}" for i in range(4)]
        errors: list[BaseException] = []
        barrier = threading.Barrier(len(candidates))

        def writer(sid: str) -> None:
            try:
                barrier.wait(timeout=5.0)
                _write_shared_session_id(sid, path)
            except BaseException as exc:  # noqa: BLE001 - capture for assert
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(sid,)) for sid in candidates]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert errors == [], f"unexpected errors: {errors!r}"
        assert path.exists(), "file should exist after concurrent writes"
        # No leftover .tmp file after atomic replaces.
        assert not (path.parent / f"{path.name}.tmp").exists()
        final = path.read_text(encoding="utf-8")
        assert final in candidates, (
            f"expected exactly one of {candidates!r}, got {final!r}"
        )

    def test_atomic_write_leaves_no_tmp_file(self, tmp_path: Path) -> None:
        """Atomic write cleans up the .tmp file after replace."""
        path = tmp_path / "session.txt"
        _write_shared_session_id("clean-write", path)
        tmp = Path(str(path) + ".tmp")
        assert not tmp.exists(), ".tmp file should be gone after atomic replace"
        assert _read_shared_session_id(path) == "clean-write"


@pytest.mark.unit
class TestSharedSessionIdConcurrency:
    """Stress tests for the thread-safe _shared_session_id accessors.

    These require PyQt6.QtWebSockets for ChatDockWidget -- skipped if absent.
    """

    def test_concurrent_set_get_no_exception(self) -> None:
        """Concurrent _set_shared_session_id / _get_shared_session_id never raises."""
        pytest.importorskip(
            "PyQt6.QtWebSockets",
            reason="PyQt6.QtWebSockets required for ChatDockWidget",
            exc_type=ImportError,
        )
        from chat._chat_dock_widget_qt import ChatDockWidget  # noqa: PLC0415

        n_threads = 50
        errors: list[Exception] = []
        seen_values: list[str | None] = []
        lock = threading.Lock()
        ChatDockWidget._set_shared_session_id(None)

        def worker(idx: int) -> None:
            try:
                sid = f"session-{idx}"
                ChatDockWidget._set_shared_session_id(sid)
                val = ChatDockWidget._get_shared_session_id()
                with lock:
                    seen_values.append(val)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        valid = {f"session-{i}" for i in range(n_threads)}
        for val in seen_values:
            assert val in valid
        final = ChatDockWidget._get_shared_session_id()
        assert final in valid

    def test_first_writer_wins_when_already_set(self) -> None:
        """Pre-seeded session is never overwritten when already set."""
        pytest.importorskip(
            "PyQt6.QtWebSockets",
            reason="PyQt6.QtWebSockets required for ChatDockWidget",
            exc_type=ImportError,
        )
        from chat._chat_dock_widget_qt import ChatDockWidget  # noqa: PLC0415

        n_threads = 30
        errors: list[Exception] = []
        barrier = threading.Barrier(n_threads)
        ChatDockWidget._set_shared_session_id("pre-seeded-session")

        def worker() -> None:
            try:
                barrier.wait()
                if not ChatDockWidget._get_shared_session_id():
                    ChatDockWidget._set_shared_session_id("should-not-be-set")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert ChatDockWidget._get_shared_session_id() == "pre-seeded-session"
