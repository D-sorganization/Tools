"""Regression tests for Codex findings on scripts/ci_bounded_apt_install.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "ci_bounded_apt_install.py"
)
_SPEC = importlib.util.spec_from_file_location("ci_bounded_apt_install", _MODULE_PATH)
assert _SPEC and _SPEC.loader
ci_apt = importlib.util.module_from_spec(_SPEC)
sys.modules["ci_bounded_apt_install"] = ci_apt
_SPEC.loader.exec_module(ci_apt)


def test_passwordless_sudo_returns_false_when_sudo_missing() -> None:
    def missing_sudo(_cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("sudo")

    assert ci_apt._passwordless_sudo(missing_sudo) is False


def test_sigterm_cleans_up_command_process_group(tmp_path: Path) -> None:
    import os
    import signal

    child_pid_file = tmp_path / "child.pid"
    parent_script = (
        "import subprocess, sys, time, pathlib\n"
        "cmd = [sys.executable, '-c', 'import time; time.sleep(60)']\n"
        "child = subprocess.Popen(cmd)\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )

    def deliver_sigterm() -> None:
        time.sleep(0.25)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=deliver_sigterm, daemon=True).start()
    with pytest.raises(SystemExit):
        ci_apt.run_bounded_command(
            [sys.executable, "-c", parent_script],
            timeout_sec=30.0,
            kill_grace_sec=0.2,
        )

    child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
    for _ in range(10):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
