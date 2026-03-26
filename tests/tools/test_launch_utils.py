"""Comprehensive TDD suite for launch_utils.py.

Tests cover validate_and_sanitize_path, launch_tool dispatch,
and all DbC contract violations (PreconditionError).
"""

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
    get_repo_root,
    launch_batch_tool,
    launch_browser_tool,
    launch_matlab_tool,
    launch_octave_tool,
    launch_python_tool,
    launch_tool,
    validate_and_sanitize_path,
)

# ─── get_repo_root ─────────────────────────────────────────────


def test_get_repo_root_returns_path():
    """get_repo_root must return an absolute Path."""
    root = get_repo_root()
    assert isinstance(root, Path)
    assert root.is_absolute()


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


# ─── launch_python_tool ────────────────────────────────────────


@patch("subprocess.Popen")
def test_launch_python_tool_normal(mock_popen, tmp_path):
    f = tmp_path / "tool.py"
    f.write_text("x = 1")
    mock_popen.return_value = MagicMock(pid=1234)
    logs: list[str] = []
    launch_python_tool(f, "MyTool", is_debug=False, log_func=logs.append)
    assert mock_popen.called
    assert any("Process started" in m for m in logs)


@patch("subprocess.Popen")
def test_launch_python_tool_debug_mode(mock_popen, tmp_path):
    f = tmp_path / "tool.py"
    f.write_text("x = 1")
    mock_proc = MagicMock()
    mock_proc.pid = 999
    mock_proc.stdout = None
    mock_proc.stderr = None
    mock_popen.return_value = mock_proc
    logs: list[str] = []
    launch_python_tool(f, "MyTool", is_debug=True, log_func=logs.append)
    assert mock_popen.called


def test_launch_python_tool_dbc_non_path(tmp_path):
    with pytest.raises(PreconditionError):
        launch_python_tool("not_a_path.py", "T")  # type: ignore[arg-type]


def test_launch_python_tool_dbc_empty_name(tmp_path):
    f = tmp_path / "tool.py"
    f.write_text("x = 1")
    with pytest.raises(PreconditionError):
        launch_python_tool(f, "")


# ─── launch_matlab_tool ────────────────────────────────────────


@patch("subprocess.Popen")
def test_launch_matlab_tool_not_found_opens_file(mock_popen, tmp_path):
    """When MATLAB is not installed, falls back to opening in editor."""
    f = tmp_path / "script.m"
    f.write_text("disp('hello')")
    mock_popen.side_effect = FileNotFoundError("matlab not found")
    logs: list[str] = []
    # Should not raise — it catches FileNotFoundError and logs warning
    with patch("os.startfile", create=True):
        launch_matlab_tool(f, "Script", log_func=logs.append)
    assert any("not found" in m.lower() or "editor" in m.lower() for m in logs)


def test_launch_matlab_tool_dbc_non_path(tmp_path):
    with pytest.raises(PreconditionError):
        launch_matlab_tool("not_a_path.m", "T")  # type: ignore[arg-type]


# ─── launch_octave_tool ────────────────────────────────────────


@patch("subprocess.Popen")
def test_launch_octave_tool_not_found_opens_file(mock_popen, tmp_path):
    """When Octave is not installed, falls back to opening in editor."""
    f = tmp_path / "script.m"
    f.write_text("disp('hello')")
    mock_popen.side_effect = FileNotFoundError("octave not found")
    logs: list[str] = []
    with patch("os.startfile", create=True):
        launch_octave_tool(f, "Script", log_func=logs.append)
    assert any("not found" in m.lower() or "editor" in m.lower() for m in logs)


def test_launch_octave_tool_dbc_non_path():
    with pytest.raises(PreconditionError):
        launch_octave_tool("not_a_path.m", "T")  # type: ignore[arg-type]


# ─── launch_browser_tool ───────────────────────────────────────


@patch("webbrowser.open")
def test_launch_browser_tool_success(mock_open, tmp_path):
    f = tmp_path / "index.html"
    f.write_text("<html/>")
    logs: list[str] = []
    launch_browser_tool(f, log_func=logs.append)
    assert mock_open.called
    assert any("browser" in msg.lower() or "opened" in msg.lower() for msg in logs)


def test_launch_browser_tool_dbc_non_path():
    with pytest.raises(PreconditionError):
        launch_browser_tool("not_a_path.html")  # type: ignore[arg-type]


# ─── launch_batch_tool ─────────────────────────────────────────


@pytest.mark.skipif(sys.platform != "win32", reason="Batch tools only run on Windows")
@patch("subprocess.Popen")
def test_launch_batch_tool_success_windows(mock_popen, tmp_path):
    f = tmp_path / "script.bat"
    f.write_text("echo hello")
    mock_popen.return_value = MagicMock()
    logs: list[str] = []
    launch_batch_tool(f, "script", log_func=logs.append)
    assert mock_popen.called


@pytest.mark.skipif(sys.platform == "win32", reason="Batch tools only fail on non-Windows")
def test_launch_batch_tool_fails_on_non_windows(tmp_path):
    f = tmp_path / "script.bat"
    f.write_text("echo hello")
    with pytest.raises(PlatformError):
        launch_batch_tool(f, "script")


@pytest.mark.skipif(sys.platform != "win32", reason="Extension check only on Windows")
def test_launch_batch_tool_wrong_extension_windows(tmp_path):
    f = tmp_path / "script.exe"
    f.write_text("...")
    with pytest.raises(SecurityError):
        launch_batch_tool(f, "script")


def test_launch_batch_tool_dbc_non_path():
    with pytest.raises(PreconditionError):
        launch_batch_tool("not_a_path.bat", "script")  # type: ignore[arg-type]


def test_launch_batch_tool_dbc_empty_name(tmp_path):
    f = tmp_path / "script.bat"
    f.write_text("echo")
    with pytest.raises(PreconditionError):
        launch_batch_tool(f, "")


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
    logs: list[str] = []
    launch_tool(
        {"name": "Tool", "path": "tool.py", "type": "python"},
        tmp_path,
        log_func=logs.append,
    )
    assert mock_popen.called


@patch("webbrowser.open")
def test_launch_tool_html_dispatch(mock_open, tmp_path):
    f = tmp_path / "index.html"
    f.write_text("<html/>")
    launch_tool({"name": "App", "path": "index.html", "type": "html"}, tmp_path)
    assert mock_open.called


@patch("webbrowser.open")
def test_launch_tool_web_type(mock_open, tmp_path):
    f = tmp_path / "index.html"
    f.write_text("<html/>")
    launch_tool({"name": "App", "path": "index.html", "type": "web"}, tmp_path)
    assert mock_open.called


@patch("subprocess.Popen")
def test_launch_tool_file_type_non_windows(mock_popen, tmp_path):
    """On non-Windows, 'file' type uses xdg-open or open."""
    f = tmp_path / "data.csv"
    f.write_text("a,b,c")
    mock_popen.return_value = MagicMock()
    if sys.platform != "win32":
        launch_tool({"name": "csv", "path": "data.csv", "type": "file"}, tmp_path)
        assert mock_popen.called


def test_launch_tool_dbc_rejects_non_dict():
    with pytest.raises(PreconditionError):
        launch_tool("not a dict", Path.cwd())  # type: ignore[arg-type]


def test_launch_tool_dbc_rejects_non_path():
    with pytest.raises(PreconditionError):
        launch_tool({"name": "x"}, "/not/a/Path/object")  # type: ignore[arg-type]
