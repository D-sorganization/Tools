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
    RateOfClosureMainWindow,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


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
        analytics_tab = window._launch_monitor_analytics_tab
        if analytics_tab.catalog is None:
            assert analytics_tab.dataset_id == "demo"
            assert analytics_tab.outcome_combo.currentText() == "ball_speed"
        else:
            assert analytics_tab.dataset_id == "normalized"
            assert len(analytics_tab.frame) == 10_169
            assert analytics_tab.outcome_combo.currentText() == "observed_carry_m"
    finally:
        window.close()
