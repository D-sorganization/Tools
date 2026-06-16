"""Persistence helpers for the P1AM desktop window layout."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QSettings

from p1am_control_system.desktop.tab_labels import TAB_ORDER

SETTINGS_ORG = "D-sorganization"
SETTINGS_APP = "P1AM_HMI"
WINDOW_GEOMETRY_KEY = "window/geometry"
WINDOW_STATE_KEY = "window/state"
TAB_VISIBLE_KEY_PREFIX = "tabs/visible/"


def make_hmi_settings() -> QSettings:
    """Create the org/app-scoped settings store for the P1AM HMI."""
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def _settings_bool(settings: QSettings, key: str, default: bool) -> bool:
    """Read a boolean QSettings value across native and INI backends."""
    value = settings.value(key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def read_tab_visibility(settings: QSettings, tab_key: str) -> bool:
    """Return the persisted visibility for a toggleable dashboard tab."""
    if tab_key not in TAB_ORDER[:-1]:
        raise ValueError(f"unknown toggleable tab: {tab_key}")
    return _settings_bool(settings, f"{TAB_VISIBLE_KEY_PREFIX}{tab_key}", True)


def write_tab_visibility(settings: QSettings, tab_visibility: dict[str, bool]) -> None:
    """Persist visibility for each toggleable dashboard tab."""
    for tab_key in TAB_ORDER[:-1]:
        settings.setValue(f"{TAB_VISIBLE_KEY_PREFIX}{tab_key}", tab_visibility[tab_key])


def restore_window_settings(window: Any, settings: QSettings | None = None) -> None:
    """Restore geometry, dock state, and tab visibility from QSettings."""
    settings = make_hmi_settings() if settings is None else settings
    geometry = settings.value(WINDOW_GEOMETRY_KEY)
    if geometry is not None:
        window.restoreGeometry(geometry)

    window_state = settings.value(WINDOW_STATE_KEY)
    if window_state is not None:
        window.restoreState(window_state)

    for tab_key in TAB_ORDER[:-1]:
        visible = read_tab_visibility(settings, tab_key)
        window.settings_tab.set_tab_visible(tab_key, visible, emit=False)
        window._handle_tab_visibility(tab_key, visible)


def persist_window_settings(window: Any, settings: QSettings | None = None) -> None:
    """Persist geometry, dock state, and tab visibility to QSettings."""
    settings = make_hmi_settings() if settings is None else settings
    settings.setValue(WINDOW_GEOMETRY_KEY, window.saveGeometry())
    settings.setValue(WINDOW_STATE_KEY, window.saveState())
    write_tab_visibility(settings, window.settings_tab.visible_tabs())
    settings.sync()
