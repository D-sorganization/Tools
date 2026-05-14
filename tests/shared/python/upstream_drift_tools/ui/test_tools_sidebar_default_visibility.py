"""Tests for persisted Sidekick default tab visibility."""

from __future__ import annotations

from pathlib import Path

import pytest


def _qt_widgets() -> object:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")
    return QtWidgets


def _tab_definition(tab_id: str, title: str, visible: bool = True) -> object:
    from upstream_drift_tools.ui.tools_sidebar import SidebarTabDefinition

    QtWidgets = _qt_widgets()
    return SidebarTabDefinition(
        tab_id,
        title,
        lambda sidebar: QtWidgets.QLabel(title, sidebar),
        visible=visible,
    )


def test_sidebar_state_persists_default_tab_visibility(tmp_path: Path) -> None:
    from upstream_drift_tools.ui.tools_sidebar import SidebarState

    path = tmp_path / "sidebar.json"
    state = SidebarState(
        default_visible_tabs=["files", "terminal", "files", ""],
        default_hidden_tabs=["chat", "chat"],
        hidden_tabs=["notes"],
    )

    state.save_json(path)
    loaded = SidebarState.load_json(path)

    assert loaded.default_visible_tabs == ["files", "terminal"]
    assert loaded.default_hidden_tabs == ["chat"]
    assert loaded.hidden_tabs == ["notes"]


def test_default_visibility_applies_before_active_tab_selection(
    tmp_path: Path,
) -> None:
    from upstream_drift_tools.ui.tools_sidebar import SidebarState, UnifiedToolsSidebar

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        state=SidebarState(
            active_tab="terminal",
            default_visible_tabs=["terminal", "notes"],
            default_hidden_tabs=["notes", "missing"],
        ),
        tab_definitions=[
            _tab_definition("files", "Files"),
            _tab_definition("terminal", "Terminal"),
            _tab_definition("notes", "Notes"),
        ],
    )

    assert sidebar.available_tab_ids() == ["files", "terminal", "notes"]
    assert sidebar.visible_tab_ids() == ["terminal"]
    assert sidebar.hidden_tab_ids() == ["files", "notes"]
    assert sidebar.active_tab_id() == "terminal"
    assert sidebar.snapshot_state().default_hidden_tabs == ["notes"]


def test_hidden_tabs_remain_available_and_override_defaults(tmp_path: Path) -> None:
    from upstream_drift_tools.ui.tools_sidebar import SidebarState, UnifiedToolsSidebar

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        state=SidebarState(
            default_visible_tabs=["files", "terminal", "notes"],
            hidden_tabs=["terminal", "stale"],
        ),
        tab_definitions=[
            _tab_definition("files", "Files"),
            _tab_definition("terminal", "Terminal"),
            _tab_definition("notes", "Notes"),
        ],
    )

    assert sidebar.available_tab_ids() == ["files", "terminal", "notes"]
    assert sidebar.visible_tab_ids() == ["files", "notes"]
    assert sidebar.hidden_tab_ids() == ["terminal"]
    assert sidebar.snapshot_state().hidden_tabs == ["terminal"]


def test_hiding_active_tab_selects_next_visible_and_keeps_recovery(
    tmp_path: Path,
) -> None:
    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        tab_definitions=[
            _tab_definition("files", "Files"),
            _tab_definition("terminal", "Terminal"),
        ],
    )

    assert sidebar.set_active_tab("files") is True
    assert sidebar.set_tab_visible("files", False) is True
    assert sidebar.active_tab_id() == "terminal"
    assert sidebar.set_tab_visible("terminal", False) is False
    assert sidebar.visible_tab_ids() == ["terminal"]


def test_unknown_default_visibility_updates_fail_clearly(tmp_path: Path) -> None:
    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        tab_definitions=[_tab_definition("files", "Files")],
    )

    with pytest.raises(ValueError, match="Unknown sidebar tab id: missing"):
        sidebar.set_default_tab_visible("missing", False)


def test_all_hidden_defaults_leave_a_recovery_tab_visible(tmp_path: Path) -> None:
    from upstream_drift_tools.ui.tools_sidebar import SidebarState, UnifiedToolsSidebar

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        state=SidebarState(default_hidden_tabs=["files", "terminal"]),
        tab_definitions=[
            _tab_definition("files", "Files"),
            _tab_definition("terminal", "Terminal"),
        ],
    )

    assert sidebar.visible_tab_ids() == ["files"]
