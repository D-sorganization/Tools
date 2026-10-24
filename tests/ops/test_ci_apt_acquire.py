"""Tests for bounded CI apt acquisition (#5717)."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import ci_apt_acquire as apt

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SCRIPT = REPO_ROOT / ".github" / "scripts" / "install-ci-system-deps.sh"


def test_rendered_install_step_uses_bounded_lock_and_apt_timeouts() -> None:
    body = apt.render_install_system_dependencies_step(("libgl1", "xvfb"))

    assert f"flock -w {apt.LOCK_WAIT_SECONDS}" in body
    assert f"timeout {apt.APT_NETWORK_TIMEOUT_SECONDS}" in body
    assert "rm -f /var/lib/dpkg/lock" not in body
    assert "wait_for_apt_locks" in body


def test_install_script_exists_and_is_executable_contract() -> None:
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "Timed out after" in text
    assert "flock -w" in text
    assert "timeout" in text
    assert "rm -f /var/lib/dpkg/lock" not in text


def test_wait_for_apt_locks_times_out_in_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stalled lock wait must fail within the configured budget."""
    monkeypatch.setattr(apt, "LOCK_WAIT_SECONDS", 2)
    script = f"""#!/bin/bash
set -euo pipefail
{apt.render_wait_for_apt_locks()}
SECONDS=0
fuser() {{ return 0; }}
if wait_for_apt_locks; then exit 0; else exit 1; fi
"""
    start = time.monotonic()
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    elapsed = time.monotonic() - start
    assert result.returncode == 1
    assert "Timed out" in result.stderr + result.stdout
    assert elapsed < 7


def test_run_with_lock_retries_succeeds_on_second_attempt() -> None:
    from scripts.install_nightly_system_deps import run_with_lock_retries

    calls = 0

    def runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="Could not get lock /var/lib/dpkg/lock"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    result = run_with_lock_retries(
        ["apt-get", "update"],
        attempts=3,
        delay_seconds=0,
        run=runner,
        sleep=lambda _s: None,
    )
    assert result.returncode == 0
    assert calls == 2


def test_run_with_lock_retries_fails_on_non_lock_error() -> None:
    from scripts.install_nightly_system_deps import run_with_lock_retries

    def runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 100, stdout="", stderr="E: broken")

    with pytest.raises(subprocess.CalledProcessError):
        run_with_lock_retries(
            ["apt-get", "install"],
            attempts=2,
            delay_seconds=0,
            run=runner,
            sleep=MagicMock(),
        )
