"""Regression tests for the Python REPL widget rename (UpstreamDrift #5617).

``SidekickTerminalWidget`` was misnamed: it never ran an OS shell, only a
Python REPL. After the #5617 rename the class is
``SidekickPythonReplWidget`` and the tab id is ``python_repl``. The OS
terminal claims the original ``terminal`` tab id.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_python_repl_widget_is_renamed_export() -> None:
    """The renamed class lives at the canonical module path."""
    from upstream_drift_tools.ui.tools_sidebar.runtime_tabs import (
        SidekickPythonReplWidget,
    )

    assert SidekickPythonReplWidget.__name__ == "SidekickPythonReplWidget"


def test_python_repl_widget_evaluates_python(tmp_path: Path) -> None:
    """The renamed widget still evaluates Python and exports variables."""
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert sidebar.set_active_tab("python_repl") is True
    repl = sidebar.tabs.currentWidget()
    script = repl.findChild(QtWidgets.QPlainTextEdit, "SidekickTerminalInput")
    run = repl.findChild(QtWidgets.QPushButton, "SidekickTerminalRun")
    output = repl.findChild(QtWidgets.QPlainTextEdit, "SidekickTerminalOutput")
    assert script is not None
    assert run is not None
    assert output is not None

    script.setPlainText("answer = 21 * 2\nprint(answer)")
    run.click()

    assert "42" in output.toPlainText()
    assert sidebar.registry.get("answer") == 42


def test_terminal_tab_id_now_hosts_os_terminal(tmp_path: Path) -> None:
    """The ``terminal`` tab id now hosts the new OS terminal widget."""
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert sidebar.set_active_tab("terminal") is True
    terminal = sidebar.tabs.currentWidget()
    # The OS terminal widget exposes a cwd label; the renamed REPL does not.
    cwd_label = terminal.findChild(QtWidgets.QLabel, "SidekickOsTerminalCwd")
    assert cwd_label is not None
