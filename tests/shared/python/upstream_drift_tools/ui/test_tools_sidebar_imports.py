"""Import and optional Qt contract tests for the unified tools sidebar."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

QT_BINDINGS = {"PyQt6", "PySide6", "PyQt5", "PySide2"}


def test_tools_sidebar_backend_imports_without_qt() -> None:
    qt_modules_before = {
        name for name in sys.modules if name.partition(".")[0] in QT_BINDINGS
    }

    from upstream_drift_tools.ui.tools_sidebar import (
        SIDEKICK_TOKEN_NAMES,
        SidebarState,
        SidekickDesignTokens,
        WorkspaceRegistry,
    )

    assert SidebarState().active_tab == "files"
    assert WorkspaceRegistry().list() == []
    assert SidekickDesignTokens()["color.accent"] == "#2563eb"
    assert "color.background" in SIDEKICK_TOKEN_NAMES

    qt_modules_after = {
        name for name in sys.modules if name.partition(".")[0] in QT_BINDINGS
    }
    assert qt_modules_after == qt_modules_before


def test_tools_sidebar_backend_imports_without_qt_in_clean_python() -> None:
    env = os.environ.copy()
    pythonpath = [
        str(Path("src").resolve()),
        str(Path("src/shared/python").resolve()),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    script = """
import sys
from upstream_drift_tools.ui.tools_sidebar import (
    SidebarState,
    SidekickDesignTokens,
    WorkspaceRegistry,
)

assert SidebarState().active_tab == "files"
assert WorkspaceRegistry().list() == []
assert SidekickDesignTokens()["color.accent"] == "#2563eb"
loaded = [
    name for name in sys.modules
    if name.partition(".")[0] in {"PyQt6", "PySide6", "PyQt5", "PySide2"}
]
assert loaded == [], loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_tools_sidebar_public_widget_api_is_lazy() -> None:
    import upstream_drift_tools.ui.tools_sidebar as tools_sidebar

    assert "create_tools_sidebar" in tools_sidebar.__all__
    assert "install_tools_sidebar" in tools_sidebar.__all__
    assert "SidekickSidebar" in tools_sidebar.__all__
    assert "SidebarTabDefinition" in tools_sidebar.__all__
    assert "SidekickDesignTokens" in tools_sidebar.__all__
    assert "sidekick_qss" in tools_sidebar.__all__


def test_sidekick_token_contract_spans_pyqt_and_web_css() -> None:
    from upstream_drift_tools.ui.tools_sidebar import SidekickDesignTokens

    tokens = SidekickDesignTokens.from_sidekick_tokens(
        {
            "sidekick.color.canvas": "#0f172a",
            "sidekick.color.surface.elevated": "#111827",
            "sidekick.radius.lg": "10px",
        }
    )

    assert tokens["color.background"] == "#0f172a"
    assert tokens["color.surface.raised"] == "#111827"
    assert tokens["radius.panel"] == "10px"
    assert tokens.css_variables()["--sidekick-color-background"] == "#0f172a"
    assert tokens.qss_variables()["sidekick-color-background"] == "#0f172a"

    css_path = Path("src/shared/typescript/theme/theme-variables.css")
    css = css_path.read_text(encoding="utf-8")
    expected_aliases = {
        "--sidekick-color-canvas: var(--theme-bg);",
        "--sidekick-color-surface: var(--theme-group-bg);",
        "--sidekick-color-border: var(--theme-border);",
        "--sidekick-color-text: var(--theme-text);",
        "--sidekick-color-accent: var(--theme-accent);",
        "--sidekick-color-focus: var(--theme-focus);",
        "--sidekick-control-height: 28px;",
    }

    for alias in expected_aliases:
        assert alias in css


def test_tools_sidebar_widget_contract_when_qt_available(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import (
        SIDEKICK_DOCK_OBJECT_NAME,
        SIDEKICK_PROJECT_TREE_OBJECT_NAME,
        SIDEKICK_SIDEBAR_OBJECT_NAME,
        SIDEKICK_TAB_BAR_OBJECT_NAME,
        SIDEKICK_TABS_OBJECT_NAME,
        SidebarState,
        SidebarTabDefinition,
        SidekickDesignTokens,
        SidekickSidebar,
        UnifiedToolsSidebar,
        create_tools_sidebar,
        install_tools_sidebar,
        sidekick_qss,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    host = QtWidgets.QMainWindow()
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)
    dock = sidebar.install_as_dock(host, area="left")

    assert sidebar.objectName() == SIDEKICK_SIDEBAR_OBJECT_NAME
    assert sidebar.tabs.objectName() == SIDEKICK_TABS_OBJECT_NAME
    assert sidebar.tabs.tabBar().objectName() == SIDEKICK_TAB_BAR_OBJECT_NAME
    assert sidebar.styleSheet() == sidekick_qss()
    assert dock.objectName() == SIDEKICK_DOCK_OBJECT_NAME
    assert dock.widget() is sidebar
    assert sidebar.active_tab_id() == "files"
    assert (
        sidebar.findChild(QtWidgets.QTreeView, SIDEKICK_PROJECT_TREE_OBJECT_NAME)
        is not None
    )
    assert sidebar.set_active_tab("terminal") is True
    assert sidebar.snapshot_state().active_tab == "terminal"
    assert sidebar.set_active_tab("missing") is False

    sidebar.set_context_variable("case", {"id": 1})
    assert sidebar.registry.get("case") == {"id": 1}

    created = create_tools_sidebar(project_root=tmp_path, parent=host)
    assert isinstance(created, UnifiedToolsSidebar)
    assert SidekickSidebar is UnifiedToolsSidebar

    result = install_tools_sidebar(host, project_root=tmp_path)
    assert result.installed is True
    assert result.sidebar is not None
    assert result.dock_widget is not None
    assert result.dock_widget.widget() is result.sidebar

    state = SidebarState(
        dock_area="left",
        active_tab="notes",
        tab_order=["notes", "files"],
        hidden_tabs=["chat"],
    )
    configured = UnifiedToolsSidebar(project_root=tmp_path, state=state)
    assert configured.visible_tab_ids()[0] == "notes"
    assert "chat" in configured.hidden_tab_ids()

    assert configured.move_tab("files", 0) is True
    assert configured.visible_tab_ids()[0] == "files"
    configured.set_minimized(True)
    assert configured.snapshot_state().minimized is True
    configured.set_minimized(False)
    assert configured.set_dock_area("right") is True
    assert configured.snapshot_state().dock_area == "right"

    popped = configured.pop_out_tab("notes")
    assert popped is not None
    assert "notes" in configured.snapshot_state().popped_out_tabs
    assert configured.redock_tab("notes") is True
    assert "notes" in configured.visible_tab_ids()

    duplicate_id = configured.duplicate_tab("calculator")
    assert duplicate_id is not None
    assert duplicate_id in configured.visible_tab_ids()

    custom = UnifiedToolsSidebar(
        project_root=tmp_path,
        design_tokens=SidekickDesignTokens({"color.background": "#ffffff"}),
        tab_definitions=[
            SidebarTabDefinition(
                "scratch",
                "Scratch",
                lambda sidebar: QtWidgets.QLabel("scratch", sidebar),
                duplicate_enabled=True,
            )
        ],
    )
    assert custom.visible_tab_ids() == ["scratch"]

    installed = install_tools_sidebar(
        host,
        project_root=tmp_path,
        sidekick_tokens={"sidekick.color.canvas": "#0f172a"},
    )
    assert installed.sidebar is not None
    assert "#0f172a" in installed.sidebar.styleSheet()
    assert custom.duplicate_tab("scratch") == "scratch#1"
