# ruff: noqa: E501
"""Tests for the PTY-backed terminal backend protocol (UpstreamDrift #5617)."""

from __future__ import annotations

import platform
import sys
import time

import pytest


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _is_posix() -> bool:
    return platform.system() in {"Linux", "Darwin"}


@pytest.fixture
def fallback_backend():  # noqa: ANN201
    """Yield a :class:`SubprocessFallbackBackend` running ``python -u``."""
    from upstream_drift_tools.ui.tools_sidebar.os_terminal import (
        SubprocessFallbackBackend,
    )

    # Run a minimal Python loop that echoes lines back: cross-platform and
    # avoids requiring an interactive shell on the CI runner.
    code = (
        "import sys\n"
        "for line in sys.stdin: sys.stdout.write(line); sys.stdout.flush()\n"
    )
    backend = SubprocessFallbackBackend(
        command=(sys.executable, "-u", "-c", code), cwd=None
    )
    backend.start()
    try:
        yield backend
    finally:
        backend.terminate()


def test_fallback_backend_starts_and_reports_running(
    fallback_backend,
) -> None:  # noqa: ANN001
    """``is_running`` is True once a child process has been spawned."""
    assert fallback_backend.is_running is True


def test_fallback_backend_round_trip(fallback_backend) -> None:  # noqa: ANN001
    """Writing bytes and reading produces the echoed payload."""
    fallback_backend.write(b"hello\n")

    deadline = time.time() + 5.0
    seen = b""
    while time.time() < deadline:
        seen += fallback_backend.read(timeout=0.1)
        if b"hello" in seen:
            break
    assert b"hello" in seen


def test_fallback_backend_write_rejects_empty() -> None:
    """``write(b"")`` raises ``ValueError`` (DbC precondition)."""
    from upstream_drift_tools.ui.tools_sidebar.os_terminal import (
        SubprocessFallbackBackend,
    )

    backend = SubprocessFallbackBackend(command=(sys.executable, "-V"), cwd=None)
    backend.start()
    try:
        with pytest.raises(ValueError):
            backend.write(b"")
    finally:
        backend.terminate()


def test_backend_read_requires_running_process() -> None:
    """Reading a never-started backend raises a clear ``RuntimeError``."""
    from upstream_drift_tools.ui.tools_sidebar.os_terminal import (
        SubprocessFallbackBackend,
    )

    backend = SubprocessFallbackBackend(command=(sys.executable, "-V"), cwd=None)
    # Never call .start() — must guard the precondition.
    with pytest.raises(RuntimeError):
        backend.read(timeout=0.1)


def test_fallback_backend_terminate_kills_process(
    fallback_backend,
) -> None:  # noqa: ANN001
    """``terminate`` flips ``is_running`` to ``False``."""
    fallback_backend.terminate()
    # Allow the process supervisor a brief moment to reap.
    for _ in range(20):
        if not fallback_backend.is_running:
            break
        time.sleep(0.05)
    assert fallback_backend.is_running is False


def test_resize_does_not_raise_on_fallback_backend(
    fallback_backend,
) -> None:  # noqa: ANN001
    """``resize`` is a no-op on the fallback backend (still satisfies protocol)."""
    fallback_backend.resize(rows=24, cols=80)


@pytest.mark.skipif(not _is_posix(), reason="ptyprocess requires POSIX")
def test_posix_pty_backend_round_trip_if_available() -> None:  # pragma: no cover
    """Smoke-test the real PTY backend when ``ptyprocess`` is installed."""
    pytest.importorskip("ptyprocess")
    from upstream_drift_tools.ui.tools_sidebar.os_terminal import PosixPtyBackend

    backend = PosixPtyBackend(command=("/bin/sh",), cwd=None)
    backend.start()
    try:
        backend.write(b"echo hello\n")
        deadline = time.time() + 5.0
        seen = b""
        while time.time() < deadline:
            seen += backend.read(timeout=0.1)
            if b"hello" in seen:
                break
        assert b"hello" in seen
    finally:
        backend.terminate()


@pytest.mark.skipif(not _is_windows(), reason="pywinpty requires Windows")
def test_windows_pty_backend_round_trip_if_available() -> None:  # pragma: no cover
    """Smoke-test the real ConPTY backend when ``pywinpty`` is installed."""
    pytest.importorskip("winpty")
    from upstream_drift_tools.ui.tools_sidebar.os_terminal import WindowsPtyBackend

    backend = WindowsPtyBackend(command=("cmd.exe",), cwd=None)
    backend.start()
    try:
        backend.write(b"echo hello\r\n")
        deadline = time.time() + 5.0
        seen = b""
        while time.time() < deadline:
            seen += backend.read(timeout=0.1)
            if b"hello" in seen:
                break
        assert b"hello" in seen
    finally:
        backend.terminate()
