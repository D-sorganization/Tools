"""Helpers for robust nightly system dependency installation."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

_DPKG_LOCK_MARKERS = (
    "Could not get lock",
    "Unable to acquire the dpkg frontend lock",
    "is another process using it",
)


def _is_dpkg_lock_failure(stderr: str) -> bool:
    return any(marker in stderr for marker in _DPKG_LOCK_MARKERS)


def run_with_lock_retries(
    command: list[str],
    *,
    attempts: int = 5,
    delay_seconds: float = 15.0,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> subprocess.CompletedProcess[str]:
    """Run an apt/dpkg command, retrying only transient lock failures."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    runner = run or (
        lambda current_command: subprocess.run(
            current_command,
            check=False,
            text=True,
            capture_output=True,
        )
    )
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, attempts + 1):
        result = runner(command)
        if result.returncode == 0:
            return result
        last_result = result
        if not _is_dpkg_lock_failure(result.stderr) or attempt == attempts:
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )
        sleep(delay_seconds)
    if last_result is None:
        raise RuntimeError("command runner did not execute")
    raise subprocess.CalledProcessError(
        last_result.returncode,
        last_result.args,
        output=last_result.stdout,
        stderr=last_result.stderr,
    )
