"""Hermetic tests for scripts/ci_bounded_apt_install.py (#5717)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
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


def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class ScriptedRunner:
    """Replay canned subprocess results in order."""

    def __init__(self, results: Sequence[subprocess.CompletedProcess[str]]) -> None:
        self._results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if not self._results:
            raise AssertionError(f"no scripted result for {cmd}")
        return self._results.pop(0)


def _idle_lock_results() -> list[subprocess.CompletedProcess[str]]:
    return [_completed(returncode=1, stdout="") for _ in range(3)]


def test_successful_install_runs_update_then_install() -> None:
    runner = ScriptedRunner(
        [
            _completed(stdout="0\n"),  # id -u
            *_idle_lock_results(),
            _completed(),  # sanitize bash
            _completed(),  # apt update
            _completed(),  # apt install
        ]
    )
    ci_apt.install_packages(
        ["libgl1", "xvfb"],
        runner=runner,
        sleep=lambda _s: None,
        lock_wait_budget_sec=30.0,
        apt_command_budget_sec=60.0,
        uid=0,
        poll_interval_sec=0.0,
    )
    joined = " ".join(" ".join(call) for call in runner.calls)
    assert "apt-get update" in joined or "apt-get" in joined and "update" in joined
    assert "libgl1" in joined


def test_lock_contention_waits_then_succeeds() -> None:
    busy = _completed(returncode=0, stdout="1234\n")
    idle = _completed(returncode=1, stdout="")
    runner = ScriptedRunner(
        [
            _completed(stdout="0\n"),
            busy,
            busy,
            idle,
            idle,
            idle,
            _completed(),
            _completed(),
            _completed(),
        ]
    )
    ci_apt.install_packages(
        ["libgl1"],
        runner=runner,
        sleep=lambda _s: None,
        lock_wait_budget_sec=30.0,
        apt_command_budget_sec=60.0,
        uid=0,
        poll_interval_sec=0.0,
    )


def test_stalled_locks_fail_with_actionable_error() -> None:
    busy = _completed(returncode=0, stdout="999\n")
    calls = 0

    def runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _completed(stdout="0\n")
        return busy

    with pytest.raises(ci_apt.CiAptInstallError) as exc:
        ci_apt.install_packages(
            ["libgl1"],
            runner=runner,
            sleep=lambda _s: None,
            lock_wait_budget_sec=0.0,
            apt_command_budget_sec=60.0,
            uid=0,
            poll_interval_sec=0.0,
            monotonic=lambda: 0.0,
        )
    assert "lock" in str(exc.value).lower()


def test_missing_privilege_skips_without_mutating() -> None:
    runner = ScriptedRunner(
        [_completed(stdout="1000\n"), _completed(returncode=1)]  # id, sudo -n true
    )
    result = ci_apt.install_packages(
        ["libgl1"],
        runner=runner,
        sleep=lambda _s: None,
        lock_wait_budget_sec=30.0,
        apt_command_budget_sec=60.0,
        uid=1000,
        sudo_check=lambda: False,
    )
    assert result.skipped is True
    assert "passwordless sudo" in result.detail.lower()


def test_apt_command_failure_surfaces_stderr() -> None:
    runner = ScriptedRunner(
        [
            _completed(stdout="0\n"),
            *_idle_lock_results(),
            _completed(),
            _completed(returncode=100, stderr="E: fetch failed"),
        ]
    )
    with pytest.raises(ci_apt.CiAptInstallError) as exc:
        ci_apt.install_packages(
            ["libgl1"],
            runner=runner,
            sleep=lambda _s: None,
            lock_wait_budget_sec=30.0,
            apt_command_budget_sec=60.0,
            uid=0,
            poll_interval_sec=0.0,
        )
    assert "fetch failed" in str(exc.value)


def test_workflow_does_not_delete_package_manager_lock_files() -> None:
    text = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "ci-standard.yml"
    ).read_text(encoding="utf-8")
    install_blocks = text.split("- name: Install System Dependencies")[1:]
    assert len(install_blocks) >= 2
    for block in install_blocks:
        assert "rm -f /var/lib/apt/lists/lock" not in block
        assert "rm -f /var/lib/dpkg/lock" not in block
