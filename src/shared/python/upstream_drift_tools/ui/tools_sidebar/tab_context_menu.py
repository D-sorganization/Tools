"""Tab context-menu construction for the Sidekick sidebar."""

from __future__ import annotations

from typing import Any

from .qt_compat import QtWidgets


def show_tab_context_menu(sidebar: Any, pos: Any) -> None:
    """Show the context menu for a Sidekick tab bar position."""
    index = sidebar.tabs.tabBar().tabAt(pos)
    if index < 0 or index >= len(sidebar._tab_ids):
        return

    tab_id = sidebar._tab_ids[index]
    definition = sidebar._tab_definitions.get(tab_id)

    menu = QtWidgets.QMenu(sidebar)

    move_menu = menu.addMenu("Move Sidebar")
    move_menu.addAction("Left").triggered.connect(lambda: sidebar.set_dock_area("left"))
    move_menu.addAction("Right").triggered.connect(
        lambda: sidebar.set_dock_area("right")
    )

    menu.addSeparator()

    if definition and definition.popout_enabled:
        menu.addAction("Pop Out").triggered.connect(lambda: sidebar.pop_out_tab(tab_id))

    if definition and definition.duplicate_enabled:
        menu.addAction("Duplicate").triggered.connect(
            lambda: sidebar.duplicate_tab(tab_id)
        )

    rename_action = menu.addAction("Rename")
    rename_action.triggered.connect(lambda: sidebar._prompt_rename_tab(tab_id))
    if tab_id in sidebar._state.tab_display_names:
        menu.addAction("Reset Name").triggered.connect(
            lambda: sidebar.reset_tab_display_name(tab_id)
        )

    menu.addSeparator()

    menu.addAction("Close").triggered.connect(
        lambda: sidebar.set_tab_visible(tab_id, False)
    )

    menu.addSeparator()

    menu.addAction("Minimize Sidebar").triggered.connect(
        lambda: sidebar.set_minimized(True)
    )

    menu.exec(sidebar.tabs.tabBar().mapToGlobal(pos))
