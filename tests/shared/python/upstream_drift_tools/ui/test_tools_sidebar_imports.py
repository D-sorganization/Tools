"""Import and optional Qt contract tests for the unified tools sidebar."""

from __future__ import annotations

import pytest


def test_tools_sidebar_backend_imports_without_qt() -> None:
    from upstream_drift_tools.ui.tools_sidebar import SidebarState, WorkspaceRegistry

    assert SidebarState().active_tab == "files"
    assert WorkspaceRegistry().list() == []


def test_tools_sidebar_public_widget_api_is_lazy() -> None:
    import upstream_drift_tools.ui.tools_sidebar as tools_sidebar

    assert "create_tools_sidebar" in tools_sidebar.__all__
    assert "install_tools_sidebar" in tools_sidebar.__all__


def test_tools_sidebar_widget_contract_when_qt_available(tmp_path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import (
        UnifiedToolsSidebar,
        create_tools_sidebar,
        install_tools_sidebar,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    host = QtWidgets.QMainWindow()
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)
    dock = sidebar.install_as_dock(host, area="left")

    assert dock.widget() is sidebar
    assert sidebar.active_tab_id() == "files"
    assert sidebar.set_active_tab("terminal") is True
    assert sidebar.snapshot_state().active_tab == "terminal"
    assert sidebar.set_active_tab("missing") is False

    sidebar.set_context_variable("case", {"id": 1})
    assert sidebar.registry.get("case") == {"id": 1}

    created = create_tools_sidebar(project_root=tmp_path, parent=host)
    assert isinstance(created, UnifiedToolsSidebar)

    result = install_tools_sidebar(host, project_root=tmp_path)
    assert result.installed is True
    assert result.sidebar is not None
    assert result.dock_widget is not None
    assert result.dock_widget.widget() is result.sidebar
