"""Tests for the Sidekick OS Terminal widget (UpstreamDrift #5617)."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.serial

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt terminal widget tests run serially on Windows.",
        allow_module_level=True,
    )


@pytest.fixture
def qt_app() -> Generator[Any, None, None]:
    """Provide a singleton ``QApplication`` for widget tests."""
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make_descriptor(identifier: str = "bash") -> Any:
    from upstream_drift_tools.ui.tools_sidebar.shell_discovery import ShellDescriptor

    return ShellDescriptor(
        identifier=identifier,
        label=identifier,
        command=("/usr/bin/" + identifier,),
    )


def test_widget_shows_cwd_label(qt_app: Any, tmp_path: Path, qtbot: Any) -> None:
    """The widget exposes a cwd label initialised to the starting directory."""
    from upstream_drift_tools.ui.tools_sidebar.os_terminal import (
        SidekickOsTerminalWidget,
    )
    from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets

    widget = SidekickOsTerminalWidget(
        project_root=tmp_path,
        shells=[_make_descriptor("bash")],
        autostart=False,
    )
    qtbot.addWidget(widget)
    label = widget.findChild(QtWidgets.QLabel, "SidekickOsTerminalCwd")
    assert label is not None
    assert str(tmp_path) in label.text()


def test_widget_shell_dropdown_lists_discovered_shells(
    qt_app: Any,
    tmp_path: Path,
    qtbot: Any,
) -> None:
    """The shell selector exposes every discovered descriptor."""
    from upstream_drift_tools.ui.tools_sidebar.os_terminal import (
        SidekickOsTerminalWidget,
    )
    from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets

    shells = [_make_descriptor("bash"), _make_descriptor("zsh")]
    widget = SidekickOsTerminalWidget(
        project_root=tmp_path,
        shells=shells,
        autostart=False,
    )
    qtbot.addWidget(widget)
    combo = widget.findChild(QtWidgets.QComboBox, "SidekickOsTerminalShellSelector")
    assert combo is not None
    items = [combo.itemText(i) for i in range(combo.count())]
    assert "bash" in items
    assert "zsh" in items


def test_widget_updates_cwd_on_osc7(qt_app: Any, tmp_path: Path, qtbot: Any) -> None:
    """OSC 7 sequences update the live cwd label via a single slot."""
    from upstream_drift_tools.ui.tools_sidebar.os_terminal import (
        SidekickOsTerminalWidget,
    )
    from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets

    widget = SidekickOsTerminalWidget(
        project_root=tmp_path,
        shells=[_make_descriptor("bash")],
        autostart=False,
    )
    qtbot.addWidget(widget)
    other = tmp_path / "subdir"
    other.mkdir()

    osc7 = f"\x1b]7;file://hostname{other.as_posix()}\x1b\\"
    widget._handle_output(osc7.encode())  # noqa: SLF001 - direct slot test

    label = widget.findChild(QtWidgets.QLabel, "SidekickOsTerminalCwd")
    assert str(other) in label.text()


def test_widget_shows_install_hint_when_no_backend_available(
    qt_app: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: Any,
) -> None:
    """When PTY libraries are missing the widget shows a labelled fallback."""
    from upstream_drift_tools.ui.tools_sidebar import os_terminal

    # Force the backend factory to report unavailability.
    monkeypatch.setattr(
        os_terminal,
        "select_backend",
        lambda **_kwargs: (None, "pty backend missing"),
    )

    widget = os_terminal.SidekickOsTerminalWidget(
        project_root=tmp_path,
        shells=[_make_descriptor("bash")],
        autostart=True,
    )
    qtbot.addWidget(widget)

    from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets

    hint = widget.findChild(QtWidgets.QLabel, "SidekickOsTerminalInstallHint")
    assert hint is not None
    assert "pty" in hint.text().lower() or "install" in hint.text().lower()


def test_widget_switching_shell_terminates_previous(
    qt_app: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: Any,
) -> None:
    """Selecting a different shell stops the current backend before starting."""
    from upstream_drift_tools.ui.tools_sidebar import os_terminal

    started: list[str] = []
    terminated: list[str] = []

    class FakeBackend:
        def __init__(self, command: tuple[str, ...], cwd: object) -> None:
            self.command = command
            self.is_running = False

        def start(self) -> None:
            started.append(self.command[0])
            self.is_running = True

        def write(self, _data: bytes) -> None:  # pragma: no cover - unused
            return None

        def read(self, timeout: float = 0.0) -> bytes:  # noqa: ARG002
            return b""

        def terminate(self) -> None:
            terminated.append(self.command[0])
            self.is_running = False

        def resize(self, rows: int, cols: int) -> None:  # noqa: ARG002
            return None

    def fake_select(
        command: tuple[str, ...],
        cwd: object,
        **_kwargs: object,
    ) -> tuple[object, str | None]:
        return FakeBackend(command=command, cwd=cwd), None

    monkeypatch.setattr(os_terminal, "select_backend", fake_select)

    shells = [_make_descriptor("bash"), _make_descriptor("zsh")]
    widget = os_terminal.SidekickOsTerminalWidget(
        project_root=tmp_path,
        shells=shells,
        autostart=True,
    )
    qtbot.addWidget(widget)
    assert started == ["/usr/bin/bash"]

    widget.switch_shell("zsh")
    assert terminated == ["/usr/bin/bash"]
    assert started[-1] == "/usr/bin/zsh"


def test_widget_shutdown_terminates_backend(
    qt_app: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: Any,
) -> None:
    """Widget shutdown must not leave backend reader threads alive."""
    from upstream_drift_tools.ui.tools_sidebar import os_terminal

    terminated: list[str] = []

    class FakeBackend:
        def __init__(self, command: tuple[str, ...], cwd: object) -> None:
            self.command = command
            self.is_running = False

        def start(self) -> None:
            self.is_running = True

        def write(self, _data: bytes) -> None:  # pragma: no cover - unused
            return None

        def read(self, timeout: float = 0.0) -> bytes:  # noqa: ARG002
            return b""

        def terminate(self) -> None:
            terminated.append(self.command[0])
            self.is_running = False

        def resize(self, rows: int, cols: int) -> None:  # noqa: ARG002
            return None

    monkeypatch.setattr(
        os_terminal,
        "select_backend",
        lambda command, cwd, **_kwargs: (FakeBackend(command=command, cwd=cwd), None),
    )

    widget = os_terminal.SidekickOsTerminalWidget(
        project_root=tmp_path,
        shells=[_make_descriptor("bash")],
        autostart=True,
    )
    qtbot.addWidget(widget)

    widget.shutdown()

    assert terminated == ["/usr/bin/bash"]
    assert widget._backend is None  # noqa: SLF001


def test_widget_resets_cwd_on_shell_switch(
    qt_app: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: Any,
) -> None:
    """Switching shells resets the cwd label to the project root."""
    from upstream_drift_tools.ui.tools_sidebar import os_terminal
    from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets

    class FakeBackend:
        def __init__(self, command: tuple[str, ...], cwd: object) -> None:
            self.command = command
            self.is_running = False

        def start(self) -> None:
            self.is_running = True

        def write(self, _data: bytes) -> None:  # pragma: no cover - unused
            return None

        def read(self, timeout: float = 0.0) -> bytes:  # noqa: ARG002
            return b""

        def terminate(self) -> None:
            self.is_running = False

        def resize(self, rows: int, cols: int) -> None:  # noqa: ARG002
            return None

    monkeypatch.setattr(
        os_terminal,
        "select_backend",
        lambda command, cwd, **_kwargs: (FakeBackend(command=command, cwd=cwd), None),
    )

    shells = [_make_descriptor("bash"), _make_descriptor("zsh")]
    widget = os_terminal.SidekickOsTerminalWidget(
        project_root=tmp_path,
        shells=shells,
        autostart=True,
    )
    qtbot.addWidget(widget)

    # Simulate a cwd change away from the project root.
    other = tmp_path / "child"
    other.mkdir()
    widget._on_cwd_changed(other)  # noqa: SLF001

    widget.switch_shell("zsh")
    label = widget.findChild(QtWidgets.QLabel, "SidekickOsTerminalCwd")
    assert str(tmp_path) in label.text()
