"""Bounded apt system-dependency install for CI shared runners (#5717)."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

LOCK_FILES = (
    "/var/lib/dpkg/lock",
    "/var/lib/apt/lists/lock",
    "/var/cache/apt/archives/lock",
)

DEFAULT_LOCK_WAIT_BUDGET_SEC = 180.0
DEFAULT_APT_COMMAND_BUDGET_SEC = 900.0
DEFAULT_POLL_INTERVAL_SEC = 2.0
DEFAULT_KILL_GRACE_SEC = 5.0


class CiAptInstallError(RuntimeError):
    """Install failed or timed out with diagnostics for CI logs."""


@dataclass(frozen=True)
class InstallResult:
    """Outcome when install is skipped (no root/sudo)."""

    skipped: bool
    detail: str


def _terminate_process_group(
    proc: subprocess.Popen[str], *, kill_grace_sec: float
) -> None:
    """Terminate and reap the dedicated process group led by ``proc``."""
    pgid = proc.pid
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=kill_grace_sec)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        proc.communicate(timeout=kill_grace_sec)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise CiAptInstallError(
            f"Process group {pgid} cleanup status unknown after timeout"
        ) from exc
    raise CiAptInstallError(
        f"Process group {pgid} survived SIGKILL after command timeout"
    )


def run_bounded_command(
    cmd: Sequence[str],
    *,
    timeout_sec: float,
    kill_grace_sec: float = DEFAULT_KILL_GRACE_SEC,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` in its own process group, killing all descendants on timeout."""
    if timeout_sec <= 0:
        raise CiAptInstallError(
            f"Command timed out before start (budget {timeout_sec:.2f}s): {' '.join(cmd)}"
        )
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(proc, kill_grace_sec=kill_grace_sec)
        raise CiAptInstallError(
            f"Command timed out after {timeout_sec:.1f}s: {' '.join(cmd)}"
        ) from exc
    except BaseException:
        _terminate_process_group(proc, kill_grace_sec=kill_grace_sec)
        raise
    return subprocess.CompletedProcess(
        args=list(cmd),
        returncode=proc.returncode if proc.returncode is not None else 1,
        stdout=stdout,
        stderr=stderr,
    )


@contextlib.contextmanager
def _acquire_outer_lock(
    lock_path: Path | None,
    *,
    deadline: float,
    poll_interval_sec: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> Iterator[None]:
    """Acquire an optional flock file within ``deadline`` without deleting it."""
    if lock_path is None:
        yield
        return
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if monotonic() >= deadline:
                    raise CiAptInstallError(
                        f"Timed out waiting for outer lock {lock_path}"
                    ) from None
                sleep(poll_interval_sec)
        yield
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _lock_in_use(run: CommandRunner, lock_path: str) -> bool:
    proc = run(["fuser", lock_path])
    return proc.returncode == 0 and bool(proc.stdout.strip())


def wait_for_package_locks(
    run: CommandRunner,
    *,
    sleep: Callable[[float], None],
    lock_wait_budget_sec: float,
    poll_interval_sec: float,
    monotonic: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
) -> None:
    """Wait until apt/dpkg locks are free, failing closed after the budget."""
    if lock_wait_budget_sec < 0:
        raise ValueError("lock_wait_budget_sec must be non-negative")
    effective_deadline = (
        deadline if deadline is not None else monotonic() + lock_wait_budget_sec
    )
    last_held: list[str] = []
    while monotonic() < effective_deadline:
        busy = False
        exceeded = False
        for path in LOCK_FILES:
            if monotonic() >= effective_deadline:
                exceeded = True
                break
            in_use = _lock_in_use(run, path)
            if monotonic() > effective_deadline:
                exceeded = True
                break
            if in_use:
                last_held = [path]
                busy = True
                break
        if exceeded:
            break
        if not busy:
            return
        if monotonic() >= effective_deadline:
            break
        sleep(poll_interval_sec)
    if not last_held and monotonic() <= effective_deadline:
        last_held = [path for path in LOCK_FILES if _lock_in_use(run, path)]
    raise CiAptInstallError(
        "Timed out after "
        f"{lock_wait_budget_sec:.0f}s waiting for apt/dpkg locks to clear "
        f"(still held: {', '.join(last_held) or 'unknown'}). "
        "Another job may be installing packages; retry later."
    )


def _sanitize_untrusted_apt_sources(run: CommandRunner) -> None:
    """Drop known-bad Google Chrome list entries without touching lock files."""
    run(
        [
            "bash",
            "-lc",
            "rm -f /etc/apt/sources.list.d/*chrome* /etc/apt/sources.list.d/*google*; "
            "sed -i '/dl\\.google\\.com/d' /etc/apt/sources.list 2>/dev/null || true",
        ]
    )


def _apt_get(run: CommandRunner, subcommand: str, packages: Sequence[str] = ()) -> None:
    cmd = [
        "env",
        "DEBIAN_FRONTEND=noninteractive",
        "apt-get",
        "-o",
        "DPkg::Lock::Timeout=120",
        subcommand,
        "--fix-missing",
    ]
    if subcommand == "install":
        cmd.extend(["-y", *packages])
    proc = run(cmd)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise CiAptInstallError(f"apt-get {subcommand} failed: {detail}")


def _install_under_flock(
    run: CommandRunner,
    packages: Sequence[str],
    *,
    lock_run: CommandRunner | None = None,
    sleep: Callable[[float], None],
    lock_wait_budget_sec: float,
    apt_command_budget_sec: float,
    poll_interval_sec: float,
    monotonic: Callable[[], float],
    deadline: float | None = None,
) -> None:
    del apt_command_budget_sec
    wait_for_package_locks(
        lock_run or run,
        sleep=sleep,
        lock_wait_budget_sec=lock_wait_budget_sec,
        poll_interval_sec=poll_interval_sec,
        monotonic=monotonic,
        deadline=deadline,
    )
    _sanitize_untrusted_apt_sources(run)
    _apt_get(run, "update")
    _apt_get(run, "install", packages)


def install_packages(
    packages: Sequence[str],
    *,
    runner: CommandRunner | None = None,
    sleep: Callable[[float], None] = time.sleep,
    lock_wait_budget_sec: float = DEFAULT_LOCK_WAIT_BUDGET_SEC,
    apt_command_budget_sec: float = DEFAULT_APT_COMMAND_BUDGET_SEC,
    poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
    uid: int | None = None,
    sudo_check: Callable[[], bool] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    outer_lock_path: Path | str | None = None,
) -> InstallResult:
    """Install Debian packages under flock with bounded lock and apt budgets."""
    if not packages:
        raise ValueError("packages must be non-empty")
    base_run = runner or (
        lambda cmd: run_bounded_command(cmd, timeout_sec=apt_command_budget_sec)
    )

    if uid is None:
        uid = int(base_run(["id", "-u"]).stdout.strip() or "1")

    if uid != 0:
        can_sudo = (
            sudo_check() if sudo_check is not None else _passwordless_sudo(base_run)
        )
        if not can_sudo:
            return InstallResult(
                skipped=True,
                detail="Passwordless sudo unavailable; using pre-provisioned deps.",
            )

    def invoke(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if uid == 0:
            return base_run(list(cmd))
        return base_run(["sudo", "-n", *cmd])

    lock_deadline = monotonic() + lock_wait_budget_sec

    def invoke_lock_probe(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if runner is not None:
            return invoke(cmd)
        remaining = lock_deadline - monotonic()
        target = list(cmd) if uid == 0 else ["sudo", "-n", *cmd]
        return run_bounded_command(target, timeout_sec=remaining)

    resolved_outer = Path(outer_lock_path) if outer_lock_path is not None else None
    with _acquire_outer_lock(
        resolved_outer,
        deadline=lock_deadline,
        poll_interval_sec=poll_interval_sec,
        sleep=sleep,
        monotonic=monotonic,
    ):
        _install_under_flock(
            invoke,
            packages,
            lock_run=invoke_lock_probe,
            sleep=sleep,
            lock_wait_budget_sec=lock_wait_budget_sec,
            apt_command_budget_sec=apt_command_budget_sec,
            poll_interval_sec=poll_interval_sec,
            monotonic=monotonic,
            deadline=lock_deadline,
        )
    return InstallResult(skipped=False, detail="installed")


def _passwordless_sudo(run: CommandRunner) -> bool:
    proc = run(["sudo", "-n", "true"])
    return proc.returncode == 0


def _parse_packages(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packages",
        required=True,
        help="Comma-separated apt package names",
    )
    parser.add_argument(
        "--lock-wait-budget-sec",
        type=float,
        default=DEFAULT_LOCK_WAIT_BUDGET_SEC,
    )
    parser.add_argument(
        "--apt-command-budget-sec",
        type=float,
        default=DEFAULT_APT_COMMAND_BUDGET_SEC,
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = install_packages(
            _parse_packages(args.packages),
            lock_wait_budget_sec=args.lock_wait_budget_sec,
            apt_command_budget_sec=args.apt_command_budget_sec,
        )
    except CiAptInstallError as exc:
        logger.error("%s", exc)
        return 1
    if result.skipped:
        logger.warning("%s", result.detail)
        return 0
    logger.info("%s", result.detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
