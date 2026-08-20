"""Tests for the shared web-app launcher process lifecycle (#3279).

Covers:
- KeyboardInterrupt during ``launch_web_app`` reaps the child
  (terminate -> wait -> kill on timeout) and returns a non-zero exit code.
- The readiness probe replaces the fixed ``time.sleep`` guess: the browser is
  opened only once the port accepts connections, and not at all on timeout.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import pytest

from gui_launcher import launcher_web

pytestmark = pytest.mark.unit


class _FakeProcess:
    """Minimal subprocess.Popen stand-in recording lifecycle calls."""

    def __init__(
        self,
        *,
        wait_raises_keyboardinterrupt: bool = False,
        wait_times_out_until_kill: bool = False,
    ) -> None:
        self._first_wait = True
        self._raise_ki = wait_raises_keyboardinterrupt
        self._times_out = wait_times_out_until_kill
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self._raise_ki and self._first_wait:
            self._first_wait = False
            raise KeyboardInterrupt
        if self._times_out and not self.killed:
            raise subprocess.TimeoutExpired(cmd="npm", timeout=timeout or 0)
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def test_reap_child_terminates_waits_and_returns_nonzero() -> None:
    proc = _FakeProcess()
    code = launcher_web._reap_child(proc)  # type: ignore[arg-type]
    assert proc.terminated is True
    assert proc.killed is False
    assert code == launcher_web._SIGINT_EXIT_CODE
    assert code != 0


def test_reap_child_escalates_to_kill_on_timeout() -> None:
    proc = _FakeProcess(wait_times_out_until_kill=True)
    code = launcher_web._reap_child(proc)  # type: ignore[arg-type]
    assert proc.terminated is True
    assert proc.killed is True
    assert code == launcher_web._SIGINT_EXIT_CODE


def test_keyboard_interrupt_reaps_child_and_reports_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """KeyboardInterrupt while waiting must reap the child and return 130."""
    web_dir = tmp_path / "web"
    (web_dir / "node_modules").mkdir(parents=True)

    proc = _FakeProcess(wait_raises_keyboardinterrupt=True)

    monkeypatch.setattr(launcher_web, "_npm_executable", lambda: "npm")
    monkeypatch.setattr(launcher_web.subprocess, "Popen", lambda *a, **k: proc)

    result = launcher_web.launch_web_app(
        "demo",
        web_dir,
        auto_open_browser=False,
    )

    assert result == launcher_web._SIGINT_EXIT_CODE
    assert result != 0
    assert proc.terminated is True
    # terminate() followed by at least one wait() — the child is reaped, not
    # left as an orphan with a misleading exit code 0.
    assert proc.wait_calls >= 2


def test_wait_for_port_returns_true_when_listening(tmp_path: Path) -> None:
    """A bound, listening port is detected by the readiness probe."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert launcher_web._wait_for_port(port, timeout=2.0) is True
    finally:
        server.close()


def test_wait_for_port_times_out_on_closed_port() -> None:
    """A closed port yields False within the bounded timeout (no infinite wait)."""
    # Bind then close to obtain a port that is (almost certainly) not listening.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("localhost", 0))
    port = probe.getsockname()[1]
    probe.close()

    assert launcher_web._wait_for_port(port, timeout=0.5, poll_interval=0.05) is False


def test_open_browser_later_skips_browser_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fixed sleep: a never-ready port means the browser is not opened."""
    opened: list[str] = []
    monkeypatch.setattr(launcher_web, "_wait_for_port", lambda *a, **k: False)
    monkeypatch.setattr(launcher_web.webbrowser, "open", lambda url: opened.append(url))

    launcher_web._open_browser_later(12345)

    assert opened == []


def test_open_browser_later_opens_browser_when_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(launcher_web, "_wait_for_port", lambda *a, **k: True)
    monkeypatch.setattr(launcher_web.webbrowser, "open", lambda url: opened.append(url))

    launcher_web._open_browser_later(5173)

    assert opened == ["http://localhost:5173"]


def test_gui_info_launcher_forwards_keyword_only_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        launcher_web,
        "launch_web_app",
        lambda **kwargs: captured.update(kwargs) or 0,
    )
    caller = tmp_path / "launch_web.py"
    result = launcher_web.launch_web_from_gui_info(
        {"name": "Demo", "web": {"path": "web"}},
        str(caller),
        env_vars={"SERVER_ONLY": "secret"},
    )
    assert result == 0
    assert captured["env_vars"] == {"SERVER_ONLY": "secret"}
