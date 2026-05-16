"""Popout and duplicate tab helpers for the Sidekick sidebar."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from .qt_compat import QtWidgets


class TabPopoutMixin:
    """Mixin for popping out, redocking, and duplicating sidebar tabs."""

    _tab_definitions: dict[str, Any]
    _tab_ids: list[str]
    _tab_widgets: dict[str, QtWidgets.QWidget]
    _popout_windows: dict[str, QtWidgets.QMainWindow]
    _duplicate_counts: dict[str, int]
    _state: Any
    tabs: QtWidgets.QTabWidget

    def tab_display_name(self, tab_id: str) -> str:
        raise NotImplementedError

    def set_active_tab(self, tab_id: str) -> bool:
        raise NotImplementedError

    def _sync_tab_order_from_widget(self) -> None:
        raise NotImplementedError

    def _emit_context(self) -> None:
        raise NotImplementedError

    def _configure_tab_settings(self) -> None:
        raise NotImplementedError

    def _add_defined_tab(self, definition: Any) -> None:
        raise NotImplementedError

    def pop_out_tab(self, tab_id: str) -> QtWidgets.QMainWindow | None:
        definition = self._tab_definitions.get(tab_id)
        if (
            definition is None
            or not definition.popout_enabled
            or tab_id not in self._tab_ids
        ):
            return None
        index = self._tab_ids.index(tab_id)
        widget = self.tabs.widget(index)
        self.tabs.removeTab(index)
        self._tab_ids.pop(index)
        self._tab_widgets.pop(tab_id, None)

        window = QtWidgets.QMainWindow(self)
        window.setObjectName(f"SidekickPopout_{tab_id}")
        window.setWindowTitle(f"Sidekick - {self.tab_display_name(tab_id)}")
        window.setCentralWidget(widget)
        window.resize(max(self._state.width, 360), max(self._state.height, 360))
        window.closeEvent = self._redock_close_event(tab_id, window)
        self._popout_windows[tab_id] = window
        window.show()
        self._emit_context()
        return window

    def redock_tab(self, tab_id: str) -> bool:
        window = self._popout_windows.pop(tab_id, None)
        if window is None:
            return tab_id in self._tab_ids
        widget = window.centralWidget()
        window.setCentralWidget(None)
        window.hide()
        self._tab_ids.append(tab_id)
        self._tab_widgets[tab_id] = widget
        self.tabs.addTab(widget, self.tab_display_name(tab_id))
        self.set_active_tab(tab_id)
        self._sync_tab_order_from_widget()
        self._emit_context()
        return True

    def duplicate_tab(self, tab_id: str) -> str | None:
        definition = self._tab_definitions.get(tab_id)
        if definition is None or not definition.duplicate_enabled:
            return None
        count = self._duplicate_counts.get(tab_id, 0) + 1
        self._duplicate_counts[tab_id] = count
        duplicate_id = f"{tab_id}#{count}"
        duplicate = replace(
            definition,
            tab_id=duplicate_id,
            title=f"{definition.title} {count + 1}",
        )
        self._tab_definitions[duplicate_id] = duplicate
        self._configure_tab_settings()
        self._add_defined_tab(duplicate)
        self.set_active_tab(duplicate_id)
        self._sync_tab_order_from_widget()
        self._emit_context()
        return duplicate_id

    def _redock_close_event(
        self,
        tab_id: str,
        window: QtWidgets.QMainWindow,
    ) -> Callable[[Any], None]:
        def _handle_close(event: Any) -> None:
            self.redock_tab(tab_id)
            event.ignore()
            window.hide()

        return _handle_close
