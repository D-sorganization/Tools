"""MATLAB-home two-pane layout tests (UpstreamDrift #5616)."""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.serial

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt matlab home layout tests run serially on Windows.",
        allow_module_level=True,
    )

pytest.importorskip("PyQt6")


def test_layout_mode_enum_values() -> None:
    from upstream_drift_tools.ui.tools_sidebar.sidebar import LayoutMode

    assert LayoutMode.SIDEBAR.value == "sidebar"
    assert LayoutMode.MATLAB_HOME.value == "matlab_home"


def _build_home(qt_app, tmp_path, qtbot):
    from upstream_drift_tools.ui.tools_sidebar.sidebar import (
        LayoutMode,
        MatlabHomeWidget,
        UnifiedToolsSidebar,
    )

    sidebar = UnifiedToolsSidebar(project_root=tmp_path)
    qtbot.addWidget(sidebar)
    home = MatlabHomeWidget(sidebar=sidebar)
    qtbot.addWidget(home)
    return sidebar, home, LayoutMode


def test_renders_two_panes(qt_app, tmp_path, qtbot) -> None:
    _, home, _ = _build_home(qt_app, tmp_path, qtbot)
    assert home.command_window_widget() is not None
    assert home.workspace_widget() is not None


def test_command_window_shares_namespace_with_workspace_inspector(
    qt_app, tmp_path, qtbot
) -> None:
    sidebar, home, _ = _build_home(qt_app, tmp_path, qtbot)
    command_window = home.command_window_widget()

    command_window.execute("x = 42")

    assert sidebar.registry.get("x") == 42
    names = [row[0] for row in home.workspace_widget().row_data()]
    assert "x" in names


def test_layout_mode_persists_across_restart(qt_app, tmp_path, qtbot) -> None:
    from upstream_drift_tools.ui.tools_sidebar.sidebar import (
        LayoutMode,
        UnifiedToolsSidebar,
    )

    sidebar = UnifiedToolsSidebar(project_root=tmp_path)
    qtbot.addWidget(sidebar)
    sidebar.set_layout_mode(LayoutMode.MATLAB_HOME)

    snapshot = sidebar.snapshot_state()
    rehydrated = UnifiedToolsSidebar(project_root=tmp_path, state=snapshot)
    qtbot.addWidget(rehydrated)

    assert rehydrated.layout_mode() == LayoutMode.MATLAB_HOME
