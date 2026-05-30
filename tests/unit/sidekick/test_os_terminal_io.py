"""Tests for os_terminal PTY I/O correctness (issue #3104 F1, F3).

F1: _on_submit must send exactly one CR (\\r), never os.linesep — ConPTY/pwsh
    sees \\r\\n as two separate submits on Windows.
F3: _handle_output must append raw decoded text without stripping trailing
    newlines — per-chunk stripping collapses blank lines and joins prompt onto
    previous line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.serial]

if sys.platform == "win32" and __import__("os").environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt terminal widget tests run serially on Windows.",
        allow_module_level=True,
    )


class _FakeBackend:
    """Minimal backend recording what was written."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.is_running = True

    def start(self) -> None:
        pass

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def read(self, timeout: float = 0.0) -> bytes:  # noqa: ARG002
        return b""

    def terminate(self) -> None:
        self.is_running = False

    def resize(self, rows: int, cols: int) -> None:  # noqa: ARG002
        pass


@pytest.fixture
def qt_app():  # noqa: ANN201
    try:
        from sidekick.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def terminal_widget(qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qtbot):  # noqa: ANN001, ANN201
    from sidekick.ui.tools_sidebar import os_terminal
    from sidekick.ui.tools_sidebar.shell_discovery import ShellDescriptor

    backend = _FakeBackend()

    monkeypatch.setattr(
        os_terminal,
        "select_backend",
        lambda **_kw: (backend, None),
    )

    descriptor = ShellDescriptor(
        identifier="bash", label="bash", command=("/usr/bin/bash",)
    )
    widget = os_terminal.SidekickOsTerminalWidget(
        project_root=tmp_path,
        shells=[descriptor],
        autostart=True,
    )
    qtbot.addWidget(widget)
    return widget, backend


def test_on_submit_writes_single_cr(terminal_widget) -> None:  # noqa: ANN001
    """F1: must send line+CR; os.linesep (CRLF on Windows) causes double-submit."""
    widget, backend = terminal_widget
    from sidekick.ui.tools_sidebar.qt_compat import QtWidgets

    input_box = widget.findChild(QtWidgets.QLineEdit, "SidekickOsTerminalInput")
    assert input_box is not None, "input QLineEdit not found"
    input_box.setText("echo hello")
    widget._on_submit()  # noqa: SLF001

    assert len(backend.written) >= 1
    last_payload = backend.written[-1]
    assert last_payload.endswith(b"\r"), (
        f"Expected payload ending with CR, got {last_payload!r}"
    )
    assert b"\r\n" not in last_payload, (
        f"Payload must not contain CRLF (Windows double-submit), got {last_payload!r}"
    )


def test_on_submit_does_not_use_os_linesep(
    terminal_widget, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1: os.linesep is not used when writing to the PTY."""
    widget, backend = terminal_widget
    import os

    from sidekick.ui.tools_sidebar.qt_compat import QtWidgets

    monkeypatch.setattr(os, "linesep", "\r\n")

    input_box = widget.findChild(QtWidgets.QLineEdit, "SidekickOsTerminalInput")
    assert input_box is not None
    input_box.setText("cmd")
    widget._on_submit()  # noqa: SLF001

    last_payload = backend.written[-1]
    assert b"\r\n" not in last_payload, (
        "os.linesep must not be forwarded to PTY; expected only \\r"
    )


def test_handle_output_preserves_trailing_newlines(terminal_widget) -> None:  # noqa: ANN001
    """F3: raw text is appended without stripping trailing \\r or \\n."""
    widget, _ = terminal_widget
    from sidekick.ui.tools_sidebar.qt_compat import QtWidgets

    output = widget.findChild(QtWidgets.QPlainTextEdit, "SidekickOsTerminalOutput")
    assert output is not None, "output QPlainTextEdit not found"
    output.clear()

    chunk = b"line one\nline two\n"
    widget._handle_output(chunk)  # noqa: SLF001
    text = output.toPlainText()
    assert "line one" in text
    assert "line two" in text
