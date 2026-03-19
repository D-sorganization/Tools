"""Unit tests for chat session-file helpers.

These helpers (_session_file_path, _read_shared_session_id, _write_shared_session_id)
have no PyQt6 dependency at runtime.  The module they live in (chat_dock_widget)
imports PyQt6 at the top, so PyQt6 must be installed for the import to succeed, but
the test logic itself needs only the standard library.

If PyQt6 is not installed in the environment the entire file is skipped via
``pytest.importorskip`` — same as the original test_chat_dock_widget.py.  The value
of keeping these tests in a separate file is:

1. They are clearly labelled as headless-safe unit tests (no widget / display server
   needed at runtime).
2. Skip debt is visible and documented — these tests SHOULD be dependency-complete
   once the session helpers are relocated to a pure-Python module (tracked separately).
3. The widget tests in test_chat_dock_widget.py are cleanly isolated from the
   pure-logic tests.
"""

from __future__ import annotations

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

    These tests exercise pure-Python logic — no display server, no Qt event loop.
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
