"""Qt helpers for Sidekick tab settings panels."""

from __future__ import annotations

from typing import Any

from .qt_compat import QtCore, QtWidgets
from .settings import SidebarTabSettingsStore

SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME = "SidekickActiveTabSettings"


class TabSettingsMixin:
    """Methods that keep tab settings out of the main sidebar controller."""

    _settings_store: SidebarTabSettingsStore
    _settings_dialog: QtWidgets.QDialog | None
    _settings_button: QtWidgets.QToolButton

    def _configure_tab_settings(self) -> None:
        self._settings_store = SidebarTabSettingsStore(
            self._tab_definitions.values(),
            self._state,
        )
        self._refresh_settings_button()

    def tab_settings(self, tab_id: str) -> dict[str, Any]:
        """Return materialized settings for one tab instance."""
        return self._settings_store.settings_for(tab_id)

    def update_tab_settings(
        self,
        tab_id: str,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist settings for a known tab instance."""
        updated = self._settings_store.update_settings(tab_id, values)
        self._state.tab_settings = self._settings_store.raw_settings()
        self._emit_context()
        return updated

    def open_active_tab_settings(self) -> bool:
        """Open the selected tab settings panel when the tab declares one."""
        tab_id = self.active_tab_id()
        definition = self._tab_definitions.get(tab_id)
        if definition is None or definition.settings is None:
            return False
        factory = definition.settings.widget_factory
        content = factory(self, tab_id) if factory is not None else None
        self._settings_dialog = build_tab_settings_dialog(self, tab_id, content)
        self._settings_dialog.show()
        return True

    def _refresh_settings_button(self, *_args: Any) -> None:
        if not hasattr(self, "_settings_button"):
            return
        if not getattr(self, "_tab_ids", []):
            self._settings_button.setEnabled(False)
            return
        definition = self._tab_definitions.get(self.active_tab_id())
        enabled = definition is not None and definition.settings is not None
        self._settings_button.setEnabled(enabled)

    def _tab_settings_payload(self) -> dict[str, dict[str, Any]]:
        return self._settings_store.materialized_settings()


def build_tab_settings_toolbar(sidebar: Any) -> QtWidgets.QToolBar:
    """Build the compact selected-tab settings action surface."""
    toolbar = QtWidgets.QToolBar(sidebar)
    toolbar.setObjectName("SidekickSettingsToolbar")
    toolbar.setIconSize(QtCore.QSize(16, 16))
    button = QtWidgets.QToolButton(toolbar)
    button.setObjectName(SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME)
    button.setText("⚙")
    button.setToolTip("Active tab settings")
    button.clicked.connect(sidebar.open_active_tab_settings)
    toolbar.addWidget(button)
    sidebar.register_settings_button_widget(button)
    return toolbar


def build_tab_settings_dialog(
    sidebar: Any,
    tab_id: str,
    content: QtWidgets.QWidget | None,
) -> QtWidgets.QDialog:
    """Create a simple settings dialog for the active tab."""
    dialog = QtWidgets.QDialog(sidebar)
    dialog.setObjectName(f"SidekickTabSettingsDialog_{tab_id}")
    dialog.setWindowTitle(f"{sidebar.tab_display_name(tab_id)} Settings")
    layout = QtWidgets.QVBoxLayout(dialog)
    if content is None:
        payload = sidebar.tab_settings(tab_id)
        label = QtWidgets.QLabel(str(payload["values"]), dialog)
        label.setWordWrap(True)
        content = label
    layout.addWidget(content)
    return dialog
