"""Shared utilities for launching tools."""

import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from src.shared.python.contracts import require


def get_repo_root() -> Path:
    """Get the absolute path to the repository root.

    Delegates to the canonical implementation in upstream_drift_tools.utils.paths.
    """
    try:
        from upstream_drift_tools.utils.paths import get_repo_root as _get_repo_root

        return Path(_get_repo_root())
    except ImportError:
        # Minimal fallback -- uses same markers as canonical implementation
        current = Path(__file__).resolve().parent
        for _ in range(10):
            if any(
                (current / m).exists() for m in (".git", "pyproject.toml", "tools.json")
            ):
                return current
            parent = current.parent
            if parent == current:
                break
            current = parent
        return Path.cwd()


# Custom Exceptions
class LaunchError(Exception):
    """Base class for launch errors."""


class ToolNotFoundError(LaunchError):
    """Raised when a tool file is not found."""


class SecurityError(LaunchError):
    """Raised when a path is invalid or unsafe."""


class PlatformError(LaunchError):
    """Raised when a tool is not supported on the current platform."""


def validate_and_sanitize_path(path_str: str, repo_root: Path) -> Path:
    """
    Validate and sanitize tool path to prevent path traversal attacks.

    Args:
        path_str: Path string from tool_info.
        repo_root: Absolute path to the repository root.

    Returns:
        Validated and sanitized Path object.

    Raises:
        SecurityError: If path is invalid or outside repository.
        ToolNotFoundError: If file does not exist.
    """
    require(
        isinstance(path_str, str) and path_str, "path_str must be a non-empty string"
    )
    require(isinstance(repo_root, Path), "repo_root must be a Path")
    require(repo_root.is_absolute(), f"repo_root must be absolute, got: {repo_root}")

    try:
        path = Path(path_str)
    except (TypeError, ValueError) as e:
        raise SecurityError(f"Invalid path format: {path_str}") from e

    # Resolve to absolute path
    try:
        full_path = (repo_root / path).resolve()
    except (OSError, RuntimeError) as e:
        raise SecurityError(f"Cannot resolve path: {path_str}") from e

    # Ensure path is within repository root
    try:
        full_path.relative_to(repo_root)
    except ValueError as err:
        raise SecurityError(f"Path traversal attempt: {full_path}") from err

    if not full_path.exists():
        raise ToolNotFoundError(f"Tool file not found: {full_path}")

    if not full_path.is_file():
        raise ToolNotFoundError(f"Path is not a file: {full_path}")

    return full_path


def _stream_reader(
    stream: IO[str], prefix: str, log_func: Callable[[str], None]
) -> None:
    """Read lines from *stream* and forward them to *log_func* with *prefix*.

    Extracted from ``launch_python_tool`` to satisfy SRP and enable unit testing.
    Issue #930: decompose long orchestration functions in the tools package.
    """
    try:
        for line in stream:
            log_func(f"{prefix} {line.strip()}")
    except (OSError, ValueError) as e:
        log_func(f"Error reading stream: {e}")
    finally:
        stream.close()


def launch_python_tool(
    path: Path,
    tool_name: str,
    is_debug: bool = False,
    log_func: Callable[[str], None] | None = None,
) -> None:
    """Launch a Python tool.

    In debug mode, stdout/stderr are piped and forwarded to *log_func* via
    daemon threads.  In normal mode the subprocess is fire-and-forget.

    Issue #930: the stream-reading closure has been extracted to the
    module-level ``_stream_reader`` helper for testability.
    """
    require(isinstance(path, Path), "path must be a Path")
    require(
        bool(isinstance(tool_name, str) and tool_name),
        "tool_name must be a non-empty string",
    )
    if log_func:
        log_func(f"Launching Python tool: {tool_name}")

    args = [sys.executable, str(path)]
    try:
        if is_debug:
            process = subprocess.Popen(
                args,
                cwd=path.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if log_func:
                log_func(f"✅ Process started (PID: {process.pid})")

                import threading

                _logger: Callable[[str], None] = log_func  # capture for threads
                if process.stdout:
                    threading.Thread(
                        target=_stream_reader,
                        args=(process.stdout, "[OUT]", _logger),
                        daemon=True,
                    ).start()
                if process.stderr:
                    threading.Thread(
                        target=_stream_reader,
                        args=(process.stderr, "[ERR]", _logger),
                        daemon=True,
                    ).start()
        else:
            subprocess.Popen(
                args,
                cwd=path.parent,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if log_func:
                log_func("✅ Process started")

    except (PermissionError, OSError) as e:
        raise LaunchError(f"Failed to start Python process: {e}") from e


def launch_matlab_tool(
    path: Path,
    tool_name: str,
    is_debug: bool = False,
    log_func: Callable[[str], None] | None = None,
) -> None:
    """Launch a MATLAB tool."""
    require(isinstance(path, Path), "path must be a Path")
    require(
        bool(isinstance(tool_name, str) and tool_name),
        "tool_name must be a non-empty string",
    )
    if log_func:
        log_func(f"Launching MATLAB tool: {tool_name}")

    sanitized_path = str(path).replace("'", "''")
    matlab_script = f"run('{sanitized_path}');"
    cmd_list = ["matlab", "-nosplash", "-nodesktop", "-r", matlab_script]

    try:
        process = subprocess.Popen(
            cmd_list,
            cwd=path.parent,
            stdout=subprocess.DEVNULL if not is_debug else None,
            stderr=subprocess.DEVNULL if not is_debug else None,
        )
        if log_func:
            log_func(f"✅ MATLAB command sent (PID: {process.pid})")

    except FileNotFoundError:
        if log_func:
            log_func("⚠️ MATLAB not found, opening script in editor")
        try:
            if hasattr(os, "startfile"):
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except (PermissionError, OSError) as e:
            raise LaunchError(f"Could not open file in editor: {e}") from e


def launch_octave_tool(
    path: Path,
    tool_name: str,
    is_debug: bool = False,
    log_func: Callable[[str], None] | None = None,
) -> None:
    """Launch an Octave tool."""
    require(isinstance(path, Path), "path must be a Path")
    require(
        bool(isinstance(tool_name, str) and tool_name),
        "tool_name must be a non-empty string",
    )
    if log_func:
        log_func(f"Launching Octave tool: {tool_name}")

    sanitized_path = str(path).replace("'", "''")
    octave_script = f"run('{sanitized_path}');"
    octave_executable = os.environ.get("OCTAVE_EXECUTABLE", "octave")
    cmd_list = [octave_executable, "--quiet", "--eval", octave_script]

    try:
        process = subprocess.Popen(
            cmd_list,
            cwd=path.parent,
            stdout=subprocess.DEVNULL if not is_debug else None,
            stderr=subprocess.DEVNULL if not is_debug else None,
        )
        if log_func:
            log_func(f"✅ Octave command sent (PID: {process.pid})")
    except FileNotFoundError:
        if log_func:
            log_func("⚠️ Octave not found, opening script in editor")
        try:
            if hasattr(os, "startfile"):
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except (PermissionError, OSError) as e:
            raise LaunchError(f"Could not open file in editor: {e}") from e


def launch_browser_tool(
    path: Path, log_func: Callable[[str], None] | None = None
) -> None:
    """Launch a browser tool."""
    require(isinstance(path, Path), "path must be a Path")
    try:
        uri = path.as_uri()
        webbrowser.open(uri)
        if log_func:
            log_func("✅ Opened in default browser")
    except (PermissionError, OSError) as e:
        raise LaunchError(f"Failed to open browser: {e}") from e


def launch_batch_tool(
    path: Path,
    tool_name: str,
    is_debug: bool = False,
    log_func: Callable[[str], None] | None = None,
) -> None:
    """Launch a batch script."""
    require(isinstance(path, Path), "path must be a Path")
    require(
        bool(isinstance(tool_name, str) and tool_name),
        "tool_name must be a non-empty string",
    )
    if sys.platform == "win32":
        if path.suffix.lower() not in [".bat", ".cmd"]:
            raise SecurityError("File must be .bat or .cmd to execute as batch script")

        if log_func:
            log_func(f"Launching batch tool: {tool_name}")

        try:
            subprocess.Popen(
                ["cmd.exe", "/c", str(path)],
                cwd=path.parent,
                stdout=subprocess.DEVNULL if not is_debug else None,
                stderr=subprocess.DEVNULL if not is_debug else None,
            )
            if log_func:
                log_func("✅ Batch script executed")
        except (PermissionError, OSError) as e:
            raise LaunchError(f"Failed to execute batch script: {e}") from e
    else:
        raise PlatformError("Batch scripts are only supported on Windows")


def launch_tool(
    tool_info: dict[str, Any],
    repo_root: Path,
    is_debug: bool = False,
    log_func: Callable[[str], None] | None = None,
) -> None:
    """
    Main entry point to launch a tool.

    Args:
        tool_info: Tool configuration dictionary.
        repo_root: Repository root path.
        is_debug: Whether to run in debug mode.
        log_func: Optional callback for logging messages.

    Raises:
        LaunchError, ToolNotFoundError, SecurityError, PlatformError
    """
    require(isinstance(tool_info, dict), "tool_info must be a dict")
    require(isinstance(repo_root, Path), "repo_root must be a Path")

    name = tool_info.get("name", "Unknown")
    path_str = tool_info.get("path")
    tool_type = tool_info.get("type")

    if not path_str:
        raise LaunchError("Tool configuration missing 'path'")

    path = validate_and_sanitize_path(path_str, repo_root)

    if tool_type == "python":
        launch_python_tool(path, name, is_debug, log_func)
    elif tool_type == "matlab":
        launch_matlab_tool(path, name, is_debug, log_func)
    elif tool_type == "octave":
        launch_octave_tool(path, name, is_debug, log_func)
    elif tool_type in ("web", "browser", "html"):
        launch_browser_tool(path, log_func)
    elif tool_type == "bat":
        launch_batch_tool(path, name, is_debug, log_func)
    elif tool_type == "file":
        if hasattr(os, "startfile"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)], cwd=path.parent)
        else:
            subprocess.Popen(["xdg-open", str(path)], cwd=path.parent)
        if log_func:
            log_func(f"✅ Opened file: {name}")
    else:
        raise LaunchError(f"Unknown tool type: {tool_type}")
