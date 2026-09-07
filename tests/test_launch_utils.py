"""TDD / DbC tests for tools.launch_utils — issue #930.

Tests cover:
  1. DbC pre-conditions on all public functions
  2. _stream_reader unit test (extracted helper — SRP decomposition)
  3. validate_and_sanitize_path security and error paths
  4. launch_tool dispatch logic
  5. Error hierarchy contract
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from contracts import PreconditionError
from tools.launch_utils import (
    LaunchError,
    PlatformError,
    SecurityError,
    ToolNotFoundError,
    _stream_reader,
    launch_batch_tool,
    launch_python_tool,
    launch_tool,
    validate_and_sanitize_path,
)

# ─────────────────────────────────────────────────────────────────────────────
# Error hierarchy
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorHierarchy:
    """Contract: exception hierarchy must be consistent."""

    def test_tool_not_found_is_launch_error(self) -> None:
        assert issubclass(ToolNotFoundError, LaunchError)

    def test_security_error_is_launch_error(self) -> None:
        assert issubclass(SecurityError, LaunchError)

    def test_platform_error_is_launch_error(self) -> None:
        assert issubclass(PlatformError, LaunchError)


# ─────────────────────────────────────────────────────────────────────────────
# _stream_reader — unit tests for extracted SRP helper
# ─────────────────────────────────────────────────────────────────────────────


class TestStreamReader:
    """Issue #930: _stream_reader extracted from nested closure for testability."""

    def test_forwards_lines_to_log_func(self) -> None:
        messages: list[str] = []
        stream = io.StringIO("hello\nworld\n")
        _stream_reader(stream, "[OUT]", messages.append)
        assert messages == ["[OUT] hello", "[OUT] world"]

    def test_applies_prefix_correctly(self) -> None:
        messages: list[str] = []
        stream = io.StringIO("line1\n")
        _stream_reader(stream, "[ERR]", messages.append)
        assert messages[0].startswith("[ERR]")

    def test_empty_stream_calls_no_log(self) -> None:
        messages: list[str] = []
        stream = io.StringIO("")
        _stream_reader(stream, "[OUT]", messages.append)
        assert messages == []

    def test_closes_stream_on_completion(self) -> None:
        stream = io.StringIO("data\n")
        _stream_reader(stream, "[OUT]", lambda _: None)
        assert stream.closed

    def test_closes_stream_on_error(self) -> None:
        """Stream must be closed even if read raises OSError."""

        class BrokenStream:
            def __iter__(self):
                raise OSError("broken pipe")

            def close(self):
                self.was_closed = True

            was_closed = False

        broken = BrokenStream()
        errors: list[str] = []
        _stream_reader(broken, "[ERR]", errors.append)
        assert broken.was_closed
        assert any("Error" in e for e in errors)

    def test_strips_trailing_newlines(self) -> None:
        messages: list[str] = []
        stream = io.StringIO("result\n")
        _stream_reader(stream, "[OUT]", messages.append)
        assert messages[0] == "[OUT] result"


# ─────────────────────────────────────────────────────────────────────────────
# validate_and_sanitize_path — DbC + security
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateAndSanitizePath:
    """Issue #930: contract tests for path validation."""

    def test_rejects_empty_path_str(self, tmp_path: Path) -> None:
        with pytest.raises((PreconditionError, ValueError)):
            validate_and_sanitize_path("", tmp_path)

    def test_rejects_non_path_repo_root(self) -> None:
        with pytest.raises((PreconditionError, ValueError, TypeError)):
            validate_and_sanitize_path("some/file.py", "/not/a/Path/object")

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityError):
            validate_and_sanitize_path("../../etc/passwd", tmp_path)

    def test_raises_tool_not_found_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ToolNotFoundError):
            validate_and_sanitize_path("does_not_exist.py", tmp_path)

    def test_accepts_valid_file(self, tmp_path: Path) -> None:
        (tmp_path / "tool.py").write_text("# tool\n")
        result = validate_and_sanitize_path("tool.py", tmp_path)
        assert result.name == "tool.py"
        assert result.is_absolute()

    def test_raises_tool_not_found_for_directory(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        with pytest.raises(ToolNotFoundError):
            validate_and_sanitize_path("subdir", tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# launch_python_tool — DbC
# ─────────────────────────────────────────────────────────────────────────────


class TestLaunchPythonToolContracts:
    def test_rejects_non_path(self) -> None:
        with pytest.raises((PreconditionError, ValueError)):
            launch_python_tool("/not/a/Path", "tool")

    def test_rejects_empty_tool_name(self, tmp_path: Path) -> None:
        f = tmp_path / "tool.py"
        f.write_text("")
        with pytest.raises((PreconditionError, ValueError)):
            launch_python_tool(f, "")

    @patch("subprocess.Popen")
    def test_launches_in_normal_mode(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        f = tmp_path / "tool.py"
        f.write_text("")
        mock_popen.return_value = MagicMock()
        launch_python_tool(f, "MyTool", is_debug=False)
        mock_popen.assert_called_once()

    @patch("tools.launch_utils._spawn_and_reap")
    def test_normal_mode_uses_reaper(
        self, mock_spawn_and_reap: MagicMock, tmp_path: Path
    ) -> None:
        f = tmp_path / "tool.py"
        f.write_text("")
        launch_python_tool(f, "MyTool", is_debug=False)
        mock_spawn_and_reap.assert_called_once()

    @patch("tools.launch_utils.threading.Thread")
    @patch("subprocess.Popen")
    def test_debug_mode_starts_reap_thread(
        self,
        mock_popen: MagicMock,
        mock_thread: MagicMock,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "tool.py"
        f.write_text("")
        process = MagicMock()
        process.pid = 123
        process.stdout = io.StringIO("")
        process.stderr = io.StringIO("")
        mock_popen.return_value = process
        mock_thread.return_value = MagicMock()

        launch_python_tool(f, "MyTool", is_debug=True)
        assert mock_thread.call_count >= 3
        thread_targets = [
            kwargs.get("target") for _, kwargs in mock_thread.call_args_list
        ]
        assert any(
            target is not None and target.__name__ == "_reap_process"
            for target in thread_targets
        )

    @patch("subprocess.Popen")
    def test_log_func_called_with_tool_name(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        f = tmp_path / "tool.py"
        f.write_text("")
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_proc.stdout = None
        mock_proc.stderr = None
        mock_popen.return_value = mock_proc
        messages: list[str] = []
        launch_python_tool(f, "MyTool", is_debug=True, log_func=messages.append)
        assert any("MyTool" in m for m in messages)


# ─────────────────────────────────────────────────────────────────────────────
# launch_batch_tool — platform contract
# ─────────────────────────────────────────────────────────────────────────────


class TestLaunchBatchToolContracts:
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_rejects_non_bat_file(self, tmp_path: Path) -> None:
        f = tmp_path / "tool.py"
        f.write_text("")
        with pytest.raises(SecurityError):
            launch_batch_tool(f, "tool")

    @pytest.mark.skipif(sys.platform == "win32", reason="Non-Windows only")
    def test_raises_platform_error_on_non_windows(self, tmp_path: Path) -> None:
        f = tmp_path / "tool.bat"
        f.write_text("")
        with pytest.raises(PlatformError):
            launch_batch_tool(f, "tool")


# ─────────────────────────────────────────────────────────────────────────────
# launch_tool — dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestLaunchToolDispatch:
    def test_raises_for_missing_path_in_tool_info(self, tmp_path: Path) -> None:
        with pytest.raises(LaunchError):
            launch_tool({"name": "X", "type": "python"}, tmp_path)

    def test_raises_for_unknown_type(self, tmp_path: Path) -> None:
        (tmp_path / "tool.xyz").write_text("")
        with pytest.raises(LaunchError):
            launch_tool({"name": "X", "path": "tool.xyz", "type": "unknown"}, tmp_path)

    def test_rejects_non_dict_tool_info(self, tmp_path: Path) -> None:
        with pytest.raises((PreconditionError, ValueError, TypeError)):
            launch_tool("not-a-dict", tmp_path)

    @patch("tools.launch_utils.launch_python_tool")
    def test_dispatches_to_launch_python_tool(
        self, mock_launch: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "my_tool.py").write_text("")
        launch_tool(
            {"name": "T", "path": "my_tool.py", "type": "python"},
            tmp_path,
        )
        mock_launch.assert_called_once()

    @patch("webbrowser.open")
    def test_dispatches_html_type_to_browser(
        self, mock_open: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "index.html").write_text("")
        launch_tool(
            {"name": "Web", "path": "index.html", "type": "html"},
            tmp_path,
        )
        mock_open.assert_called_once()


class TestLaunchToolLifecycle:
    """Process lifecycle tests for launcher helpers."""

    def test_file_launch_uses_spawn_and_reap(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("x")
        with (
            patch("tools.launch_utils._spawn_and_reap") as mock_spawn_and_reap,
            patch("os.startfile", create=True) as mock_startfile,
        ):
            # On platforms with startfile, verify startfile is invoked
            launch_tool(
                {"name": "Doc", "path": "notes.txt", "type": "file"},
                tmp_path,
            )
            mock_startfile.assert_called_once()
            mock_spawn_and_reap.assert_not_called()

        with (
            patch("tools.launch_utils._spawn_and_reap") as mock_spawn_and_reap,
            patch("sys.platform", "linux"),
        ):
            # On POSIX without startfile, verify _spawn_and_reap is invoked
            original_startfile = getattr(os, "startfile", None)
            if hasattr(os, "startfile"):
                delattr(os, "startfile")
            try:
                launch_tool(
                    {"name": "Doc", "path": "notes.txt", "type": "file"},
                    tmp_path,
                )
                mock_spawn_and_reap.assert_called_once()
            finally:
                if original_startfile is not None:
                    os.startfile = original_startfile  # type: ignore[attr-defined]
