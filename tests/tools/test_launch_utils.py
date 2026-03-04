"""Comprehensive TDD suite for launch_utils.py.

Tests cover validate_and_sanitize_path, launch_tool dispatch,
and all DbC contract violations (PreconditionError).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.contracts import PreconditionError
from src.tools.launch_utils import (
    LaunchError,
    PlatformError,
    SecurityError,
    ToolNotFoundError,
    validate_and_sanitize_path,
    launch_browser_tool,
    launch_tool,
)


# ─── validate_and_sanitize_path ────────────────────────────────


def test_validate_path_success(tmp_path):
    f = tmp_path / "tool.py"
    f.write_text("x = 1")
    result = validate_and_sanitize_path("tool.py", tmp_path)
    assert result == f.resolve()


def test_validate_path_traversal_blocked(tmp_path):
    with pytest.raises(SecurityError, match="traversal"):
        validate_and_sanitize_path("../escape.py", tmp_path)


def test_validate_path_missing_file(tmp_path):
    with pytest.raises(ToolNotFoundError):
        validate_and_sanitize_path("nonexistent.py", tmp_path)


def test_validate_path_directory_rejected(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    with pytest.raises(ToolNotFoundError, match="not a file"):
        validate_and_sanitize_path("subdir", tmp_path)


def test_validate_path_dbc_empty_string(tmp_path):
    with pytest.raises(PreconditionError):
        validate_and_sanitize_path("", tmp_path)


def test_validate_path_dbc_non_path_root(tmp_path):
    f = tmp_path / "tool.py"
    f.write_text("x = 1")
    with pytest.raises(PreconditionError):
        validate_and_sanitize_path("tool.py", str(tmp_path))  # type: ignore[arg-type]


def test_validate_path_dbc_relative_root():
    with pytest.raises(PreconditionError, match="absolute"):
        validate_and_sanitize_path("tool.py", Path("relative/path"))


# ─── launch_browser_tool ───────────────────────────────────────


@patch("webbrowser.open")
def test_launch_browser_tool_success(mock_open, tmp_path):
    f = tmp_path / "index.html"
    f.write_text("<html/>")
    logs = []
    launch_browser_tool(f, log_func=logs.append)
    assert mock_open.called
    assert any("browser" in msg.lower() or "opened" in msg.lower() for msg in logs)


def test_launch_browser_tool_dbc_non_path():
    with pytest.raises(PreconditionError):
        launch_browser_tool("not_a_path.html")  # type: ignore[arg-type]


# ─── launch_tool dispatch ──────────────────────────────────────


def test_launch_tool_missing_path_key():
    with pytest.raises(LaunchError, match="missing 'path'"):
        launch_tool({"name": "foo", "type": "python"}, Path.cwd())


def test_launch_tool_unknown_type(tmp_path):
    f = tmp_path / "tool.py"
    f.write_text("x = 1")
    with pytest.raises(LaunchError, match="Unknown tool type"):
        launch_tool({"name": "foo", "path": "tool.py", "type": "unknown"}, tmp_path)


@patch("subprocess.Popen")
def test_launch_tool_python_dispatch(mock_popen, tmp_path):
    f = tmp_path / "tool.py"
    f.write_text("x = 1")
    mock_proc = MagicMock()
    mock_proc.pid = 1234
    mock_popen.return_value = mock_proc
    logs = []
    launch_tool(
        {"name": "Tool", "path": "tool.py", "type": "python"},
        tmp_path,
        log_func=logs.append,
    )
    assert mock_popen.called


def test_launch_tool_dbc_rejects_non_dict():
    with pytest.raises(PreconditionError):
        launch_tool("not a dict", Path.cwd())  # type: ignore[arg-type]


def test_launch_tool_dbc_rejects_non_path():
    with pytest.raises(PreconditionError):
        launch_tool({"name": "x"}, "/not/a/Path/object")  # type: ignore[arg-type]


# ─── Platform errors ───────────────────────────────────────────


@pytest.mark.skipif(
    sys.platform == "win32", reason="Batch tools only fail on non-Windows"
)
def test_launch_batch_tool_fails_on_non_windows(tmp_path):
    from src.tools.launch_utils import launch_batch_tool

    f = tmp_path / "script.bat"
    f.write_text("echo hello")
    with pytest.raises(PlatformError):
        launch_batch_tool(f, "script")
