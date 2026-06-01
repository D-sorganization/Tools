from pathlib import Path

import pytest

pytest.importorskip("PyQt6")


def test_sidebar_private_aliases_stay_live_after_tab_reconfiguration(
    tmp_path: Path,
    qtbot,
) -> None:
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar
    from sidekick.ui.tools_sidebar.tab_definition import SidebarTabDefinition

    first = SidebarTabDefinition(
        tab_id="first",
        title="First",
        factory=lambda _sidebar: QtWidgets.QLabel("first"),
        duplicate_enabled=True,
        popout_enabled=True,
    )
    second = SidebarTabDefinition(
        tab_id="second",
        title="Second",
        factory=lambda _sidebar: QtWidgets.QLabel("second"),
    )
    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        tab_definitions=[first],
    )
    qtbot.addWidget(sidebar)

    sidebar.configure_tabs([first, second])

    assert sidebar._tab_definitions is sidebar._tab_collection._tab_definitions
    assert sidebar._tab_ids is sidebar._tab_collection._tab_ids
    assert sidebar._tab_widgets is sidebar._tab_collection._tab_widgets
    assert sidebar._tab_definitions["first"] is first

    duplicate_id = sidebar.duplicate_tab("first")

    assert duplicate_id == "first#1"
    assert duplicate_id in sidebar._tab_collection.all_ids()
    assert duplicate_id in sidebar.visible_tab_ids()
