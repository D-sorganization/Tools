"""Process-tree termination helpers for subscription CLI adapters (#5727 / #6023)."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time

DEFAULT_GRACE_KILL_SECONDS = 2.0


def cli_subprocess_pid_from_error(exc: BaseException | None) -> int | None:
    """Return a child PID from CLI runner exceptions, walking ``__cause__``."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        pid = getattr(current, "pid", None)
        if isinstance(pid, int) and pid > 0:
            return pid
        current = current.__cause__
    return None


def terminate_cli_process_group(pid: int | None) -> bool:
    """Best-effort kill of the CLI process group on timeout or cancellation."""
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        return _windows_kill_process_tree(pid)
    if not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
        return True
    except OSError:
        return False


def terminate_cli_subprocess_after_error(
    exc: BaseException | None,
    *,
    proc: subprocess.Popen[bytes] | None = None,
    grace_kill_seconds: float = DEFAULT_GRACE_KILL_SECONDS,
) -> bool:
    """Kill a CLI child using a live ``Popen`` handle and/or exception-bound PID."""
    killed = False
    if proc is not None and proc.poll() is None:
        _graceful_then_forced_kill(proc, grace_kill_seconds=grace_kill_seconds)
        killed = True
    pid = cli_subprocess_pid_from_error(exc)
    if pid is not None and (proc is None or proc.poll() is not None):
        killed = terminate_cli_process_group(pid) or killed
    return killed


def _graceful_then_forced_kill(
    proc: subprocess.Popen[bytes], *, grace_kill_seconds: float
) -> None:
    if proc.poll() is not None:
        return
    pid = proc.pid
    if os.name == "nt":
        if grace_kill_seconds > 0:
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T"],
                    check=False,
                    capture_output=True,
                    timeout=grace_kill_seconds,
                )
        deadline = time.monotonic() + grace_kill_seconds
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if proc.poll() is None:
            _windows_kill_process_tree(pid, timeout=max(grace_kill_seconds, 2.0))
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    deadline = time.monotonic() + grace_kill_seconds
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if proc.poll() is None:
        terminate_cli_process_group(pid)


def _windows_kill_process_tree(pid: int, *, timeout: float = 2.0) -> bool:
    with contextlib.suppress(Exception):
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    return True
