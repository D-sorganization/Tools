"""Persistent, reorderable primary-navigation contracts."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings, Qt  # noqa: E402

from rate_of_closure.ui.pyqt6.main_window import (  # noqa: E402
    _DEFAULT_TAB_IDS,
    _NAVIGATION_SETTINGS_APP,
    _NAVIGATION_SETTINGS_ORG,
    _NAVIGATION_STATE_KEY,
    _REQUIRED_TAB_IDS,
    RateOfClosureMainWindow,
)
from rate_of_closure.ui.pyqt6.navigation_state import (  # noqa: E402
    DEFAULT_TAB_IDS,
    NAVIGATION_SETTINGS_APP,
    NAVIGATION_SETTINGS_ORG,
    NAVIGATION_STATE_KEY,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_main_window_reexports_the_focused_navigation_contract() -> None:
    """Keep compatibility imports bound to the extracted state contract."""
    assert _DEFAULT_TAB_IDS is DEFAULT_TAB_IDS
    assert _NAVIGATION_SETTINGS_ORG is NAVIGATION_SETTINGS_ORG
    assert _NAVIGATION_SETTINGS_APP is NAVIGATION_SETTINGS_APP
    assert _NAVIGATION_STATE_KEY is NAVIGATION_STATE_KEY


@pytest.fixture
def settings(tmp_path):  # type: ignore[no-untyped-def]
    path = tmp_path / "navigation.ini"
    return QSettings(str(path), QSettings.Format.IniFormat)


def test_primary_tabs_are_movable_and_have_stable_ids(qtbot, settings) -> None:  # type: ignore[no-untyped-def]
    window = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(window)
    try:
        assert window._tabs.isMovable()
        assert window._tabs.usesScrollButtons()
        assert window._tabs.tabBar().elideMode() == Qt.TextElideMode.ElideNone
        assert window.primary_tab_ids() == list(_DEFAULT_TAB_IDS)
        assert "drag" in window._tabs.tabBar().toolTip().lower()
    finally:
        window.close()


def test_reordered_tabs_and_active_view_round_trip(qtbot, settings) -> None:  # type: ignore[no-untyped-def]
    first = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(first)
    first._tabs.tabBar().moveTab(3, 0)
    first._tabs.setCurrentWidget(first._simulation_tab)
    expected = first.primary_tab_ids()
    first.close()

    second = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(second)
    try:
        assert second.primary_tab_ids() == expected
        assert second._tabs.currentWidget() is second._simulation_tab
    finally:
        second.close()


def test_corrupt_navigation_state_falls_back_to_defaults(qtbot, settings) -> None:  # type: ignore[no-untyped-def]
    settings.setValue(_NAVIGATION_STATE_KEY, "{not-json")
    window = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(window)
    try:
        assert window.primary_tab_ids() == list(_DEFAULT_TAB_IDS)
    finally:
        window.close()


def test_legacy_or_partial_order_is_sanitized(qtbot, settings) -> None:  # type: ignore[no-untyped-def]
    settings.setValue(
        _NAVIGATION_STATE_KEY,
        json.dumps(
            {
                "version": 1,
                "order": ["plots", "unknown", "plots", "simulation"],
                "active": "plots",
            }
        ),
    )
    window = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(window)
    try:
        assert window.primary_tab_ids()[:2] == ["plots", "simulation"]
        assert set(window.primary_tab_ids()) == set(_DEFAULT_TAB_IDS)
        assert window.visible_primary_tab_ids() == window.primary_tab_ids()
    finally:
        window.close()


def test_default_qsettings_namespace_is_application_specific() -> None:
    assert _NAVIGATION_SETTINGS_ORG == "D-sorganization"
    assert _NAVIGATION_SETTINGS_APP == "RateOfClosureImpactExplorer"


def test_launch_monitor_tab_is_registered_once(qtbot, settings) -> None:  # type: ignore[no-untyped-def]
    window = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(window)
    try:
        assert window.primary_tab_ids().count("launch_monitor_analytics") == 1
        assert window._tabs.indexOf(window._launch_monitor_analytics_tab) >= 0
        assert (
            window._launch_monitor_analytics_tab.outcome_combo.currentText()
            == "ball_speed"
        )
    finally:
        window.close()


def test_visibility_and_order_round_trip_with_deterministic_active_fallback(
    qtbot, settings
) -> None:  # type: ignore[no-untyped-def]
    first = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(first)
    first.set_primary_module_active("simulation")
    assert first.set_primary_module_visible("simulation", False)
    assert first.current_primary_module_id() == "clubhead"
    assert first.move_primary_module("plots", 1)
    expected_order = first.primary_tab_ids()
    first.close()

    second = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(second)
    try:
        assert second.primary_tab_ids() == expected_order
        assert "simulation" not in second.visible_primary_tab_ids()
        assert second.current_primary_module_id() == "clubhead"
    finally:
        second.close()


def test_required_module_cannot_be_hidden(qtbot, settings) -> None:  # type: ignore[no-untyped-def]
    window = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(window)
    try:
        required = _REQUIRED_TAB_IDS[0]
        assert not window.set_primary_module_visible(required, False)
        assert required in window.visible_primary_tab_ids()
        entry = next(
            item
            for item in window.primary_module_entries()
            if item.module_id == required
        )
        assert entry.required
    finally:
        window.close()


def test_restore_defaults_restores_visibility_order_and_active_module(
    qtbot, settings
) -> None:  # type: ignore[no-untyped-def]
    window = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(window)
    try:
        window.set_primary_module_visible("plots", False)
        window.move_primary_module("simulation", -1)
        window.set_primary_module_active("flight_explorer")

        window.restore_default_workspace()

        assert window.primary_tab_ids() == list(_DEFAULT_TAB_IDS)
        assert window.visible_primary_tab_ids() == list(_DEFAULT_TAB_IDS)
        assert window.current_primary_module_id() == _DEFAULT_TAB_IDS[0]
    finally:
        window.close()


def test_saved_visibility_ignores_unknown_ids_and_preserves_required_module(
    qtbot, settings
) -> None:  # type: ignore[no-untyped-def]
    settings.setValue(
        _NAVIGATION_STATE_KEY,
        json.dumps(
            {
                "version": 1,
                "order": ["future_module", "simulation"],
                "visible": ["future_module", "simulation"],
                "active": "future_module",
            }
        ),
    )
    window = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(window)
    try:
        assert set(window.visible_primary_tab_ids()) == {
            "simulation",
            *_REQUIRED_TAB_IDS,
        }
        assert "future_module" not in window.primary_tab_ids()
        assert window.current_primary_module_id() in window.visible_primary_tab_ids()
    finally:
        window.close()
