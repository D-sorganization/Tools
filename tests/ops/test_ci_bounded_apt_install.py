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


def test_passwordless_sudo_returns_false_when_sudo_missing() -> None:
    def missing_sudo(_cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("sudo")

    assert ci_apt._passwordless_sudo(missing_sudo) is False


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


def test_wait_for_package_locks_enforces_deadline_across_slow_probes() -> None:
    """Three free-lock fuser calls advancing 200s each must fail a 180s budget."""
    now = 0.0

    def clock() -> float:
        return now

    def slow_free_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        nonlocal now
        now += 200.0
        return _completed(returncode=1, stdout="")

    with pytest.raises(ci_apt.CiAptInstallError, match="Timed out"):
        ci_apt.wait_for_package_locks(
            slow_free_runner,
            sleep=lambda _s: None,
            lock_wait_budget_sec=180.0,
            poll_interval_sec=0.0,
            monotonic=clock,
        )
    assert now <= 200.0


def test_outer_lock_contention_exits_within_budget_without_deleting_or_stealing_lock(
    tmp_path: Path,
) -> None:
    """Contention on an isolated test lock fails within budget and keeps the lock."""
    import fcntl

    lock_file = tmp_path / "d-sorg-apt-install.lock"
    holder = lock_file.open("w", encoding="utf-8")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = ScriptedRunner([_completed(stdout="0\n")])
        with pytest.raises(ci_apt.CiAptInstallError, match="outer lock"):
            ci_apt.install_packages(
                ["libgl1"],
                runner=runner,
                sleep=lambda _s: None,
                lock_wait_budget_sec=0.05,
                apt_command_budget_sec=1.0,
                poll_interval_sec=0.01,
                uid=0,
                outer_lock_path=lock_file,
            )
        assert lock_file.exists()
        # Verify original holder still holds the lock exclusively
        probe = lock_file.open("w", encoding="utf-8")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            probe.close()
    finally:
        holder.close()


def test_sigterm_cleans_up_command_process_group(tmp_path: Path) -> None:
    """SIGTERM to the helper must not leave the detached command group running."""
    import os
    import signal
    import threading
    import time

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

    assert child_pid_file.is_file()
    child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
    for _ in range(10):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_command_timeout_terminates_and_reaps_descendant_process_group(
    tmp_path: Path,
) -> None:
    """A timed-out command must terminate and reap its spawned child processes."""
    import os
    import time

    child_pid_file = tmp_path / "child.pid"
    parent_script = (
        "import subprocess, sys, time, pathlib\n"
        "cmd = [sys.executable, '-c', 'import time; time.sleep(60)']\n"
        "child = subprocess.Popen(cmd)\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(ci_apt.CiAptInstallError, match="timed out"):
        ci_apt.run_bounded_command(
            [sys.executable, "-c", parent_script],
            timeout_sec=0.3,
            kill_grace_sec=0.2,
        )

    assert child_pid_file.is_file()
    child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
    # Give the kernel a brief moment to finalize process table reaping
    for _ in range(10):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
