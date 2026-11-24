"""#6167 / #5727: Windows taskkill failures must be surfaced, not swallowed."""

from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.python.ai.adapters import cli_subprocess_termination as term

pytestmark = pytest.mark.unit

_MODULE = "shared.python.ai.adapters.cli_subprocess_termination"


def _completed(returncode: int) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=["taskkill"], returncode=returncode, stdout=b"", stderr=b"denied"
    )


def test_taskkill_success_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(term.subprocess, "run", lambda *a, **k: _completed(0))
    assert term._windows_kill_process_tree(1234) is True


def test_taskkill_process_already_gone_counts_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """taskkill exits 128 when the PID no longer exists: the child is gone."""
    monkeypatch.setattr(term.subprocess, "run", lambda *a, **k: _completed(128))
    assert term._windows_kill_process_tree(1234) is True


def test_taskkill_nonzero_exit_returns_false_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(term.subprocess, "run", lambda *a, **k: _completed(1))
    with caplog.at_level(logging.WARNING, logger=term.__name__):
        assert term._windows_kill_process_tree(1234) is False
    assert any("taskkill" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "exc",
    [
        subprocess.TimeoutExpired(cmd="taskkill", timeout=2.0),
        FileNotFoundError("taskkill"),
        OSError("boom"),
    ],
)
def test_taskkill_exception_returns_false_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exc: BaseException,
) -> None:
    def _raise(*_a: object, **_k: object) -> None:
        raise exc

    monkeypatch.setattr(term.subprocess, "run", _raise)
    with caplog.at_level(logging.WARNING, logger=term.__name__):
        assert term._windows_kill_process_tree(1234) is False
    assert any("taskkill" in r.getMessage() for r in caplog.records)


def test_terminate_process_group_propagates_windows_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(term, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(term.subprocess, "run", lambda *a, **k: _completed(5))
    assert term.terminate_cli_process_group(4321) is False


def test_forced_kill_warns_when_child_survives(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """After the forced kill the child exit is verified; a survivor is logged."""
    monkeypatch.setattr(term, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(term.subprocess, "run", lambda *a, **k: _completed(0))
    proc = MagicMock()
    proc.pid = 777
    proc.poll.return_value = None  # never exits
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=0.01)
    with caplog.at_level(logging.WARNING, logger=term.__name__):
        term._graceful_then_forced_kill(proc, grace_kill_seconds=0.0)
    assert any("still running" in r.getMessage() for r in caplog.records)


def test_forced_kill_quiet_when_child_exits(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(term, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(term.subprocess, "run", lambda *a, **k: _completed(0))
    proc = MagicMock()
    proc.pid = 778
    proc.poll.return_value = None
    proc.wait.return_value = 1
    with caplog.at_level(logging.WARNING, logger=term.__name__):
        term._graceful_then_forced_kill(proc, grace_kill_seconds=0.0)
    assert not caplog.records
