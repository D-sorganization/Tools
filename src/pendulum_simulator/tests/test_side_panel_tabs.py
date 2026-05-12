"""Tests for SidePanelTabs — the reusable side-panel tabbed container.

These tests intentionally come before the implementation (TDD red).
They lock down the public contract:

- Construction with a non-empty settings key
- add_panel() validates label and widget; returns ascending tab index
- add_panel() rejects empty labels and duplicates (DbC)
- panel_labels() returns insertion order
- active_tab_label() / set_active_tab() round-trip
- save_state() / restore_state() round-trip via QSettings
- Every added panel is wrapped in a QScrollArea (no clipping)
- The widget passed in is reachable through the wrapping scroll area
"""

from __future__ import annotations

from typing import Any

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QLabel, QScrollArea

from double_pendulum_golf.gui.side_panel_tabs import SidePanelTabs

_TEST_KEY = "test_side_panel_tabs/active_tab"


@pytest.fixture(autouse=True)
def _clear_settings() -> Any:
    """Each test starts with a clean QSettings slot for the test key."""
    QSettings("D-sorganization", "PendulumSimulator").remove(_TEST_KEY)
    yield
    QSettings("D-sorganization", "PendulumSimulator").remove(_TEST_KEY)


# ── Construction & DbC -----------------------------------------------------


def test_construct_requires_non_empty_settings_key(qapp) -> Any:
    with pytest.raises(ValueError, match="settings_key"):
        SidePanelTabs(settings_key="")


def test_construct_succeeds_with_valid_key(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    assert tabs.count() == 0
    assert tabs.panel_labels() == []


# ── add_panel ---------------------------------------------------------------


def test_add_panel_returns_ascending_index(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    idx0 = tabs.add_panel("Setup", QLabel("setup"))
    idx1 = tabs.add_panel("Plots", QLabel("plots"))
    idx2 = tabs.add_panel("Optimizer", QLabel("opt"))
    assert (idx0, idx1, idx2) == (0, 1, 2)
    assert tabs.count() == 3


def test_add_panel_records_labels_in_order(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    tabs.add_panel("Setup", QLabel("a"))
    tabs.add_panel("Mass Matrix", QLabel("b"))
    tabs.add_panel("Plots", QLabel("c"))
    assert tabs.panel_labels() == ["Setup", "Mass Matrix", "Plots"]


def test_add_panel_rejects_empty_label(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    with pytest.raises(ValueError, match="label"):
        tabs.add_panel("", QLabel("x"))
    with pytest.raises(ValueError, match="label"):
        tabs.add_panel("   ", QLabel("x"))


def test_add_panel_rejects_none_widget(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    with pytest.raises(ValueError, match="widget"):
        tabs.add_panel("Setup", None)  # type: ignore[arg-type]


def test_add_panel_rejects_duplicate_label(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    tabs.add_panel("Setup", QLabel("first"))
    with pytest.raises(ValueError, match="duplicate|already"):
        tabs.add_panel("Setup", QLabel("second"))


# ── Panel wrapping (LOD: don't reach into internals) -----------------------


def test_each_panel_is_wrapped_in_scroll_area(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    tabs.add_panel("Setup", QLabel("a"))
    tabs.add_panel("Plots", QLabel("b"))
    for i in range(tabs.count()):
        wrapper = tabs.widget(i)
        assert isinstance(
            wrapper, QScrollArea
        ), f"Tab {i} is {type(wrapper).__name__}, expected QScrollArea"


def test_added_widget_reachable_through_panel_widget(qapp) -> Any:
    """The original widget is accessible via panel_widget(label)."""
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    inner = QLabel("real content")
    tabs.add_panel("Setup", inner)
    assert tabs.panel_widget("Setup") is inner


def test_panel_widget_unknown_label_raises(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    tabs.add_panel("Setup", QLabel("x"))
    with pytest.raises(KeyError, match="Plots"):
        tabs.panel_widget("Plots")


# ── Active tab navigation ---------------------------------------------------


def test_active_tab_label_matches_selection(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    tabs.add_panel("Setup", QLabel("a"))
    tabs.add_panel("Plots", QLabel("b"))
    tabs.add_panel("Optimizer", QLabel("c"))
    assert tabs.active_tab_label() == "Setup"
    tabs.setCurrentIndex(2)
    assert tabs.active_tab_label() == "Optimizer"


def test_set_active_tab_switches_by_label(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    tabs.add_panel("Setup", QLabel("a"))
    tabs.add_panel("Plots", QLabel("b"))
    tabs.set_active_tab("Plots")
    assert tabs.active_tab_label() == "Plots"


def test_set_active_tab_unknown_label_raises(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    tabs.add_panel("Setup", QLabel("a"))
    with pytest.raises(KeyError, match="Plots"):
        tabs.set_active_tab("Plots")


def test_active_tab_label_empty_tabs_returns_empty_string(qapp) -> Any:
    """Querying an empty tab bar returns '' rather than crashing."""
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    assert tabs.active_tab_label() == ""


# ── Persistence -------------------------------------------------------------


def test_save_and_restore_state_round_trip(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    tabs.add_panel("Setup", QLabel("a"))
    tabs.add_panel("Plots", QLabel("b"))
    tabs.add_panel("Optimizer", QLabel("c"))
    tabs.set_active_tab("Optimizer")
    tabs.save_state()

    # Build a fresh widget with the same content; restore_state must
    # bring back the active tab.
    tabs2 = SidePanelTabs(settings_key=_TEST_KEY)
    tabs2.add_panel("Setup", QLabel("a"))
    tabs2.add_panel("Plots", QLabel("b"))
    tabs2.add_panel("Optimizer", QLabel("c"))
    tabs2.restore_state()
    assert tabs2.active_tab_label() == "Optimizer"


def test_restore_state_with_no_saved_value_is_noop(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    tabs.add_panel("Setup", QLabel("a"))
    tabs.add_panel("Plots", QLabel("b"))
    # Nothing saved → should not crash and should keep default selection
    tabs.restore_state()
    assert tabs.active_tab_label() == "Setup"


def test_restore_state_with_obsolete_label_falls_back(qapp) -> Any:
    """Saved label that no longer exists keeps the default tab."""
    QSettings("D-sorganization", "PendulumSimulator").setValue(
        _TEST_KEY, "ObsoleteLabel"
    )
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    tabs.add_panel("Setup", QLabel("a"))
    tabs.add_panel("Plots", QLabel("b"))
    tabs.restore_state()
    assert tabs.active_tab_label() == "Setup"


# ── Tooltip support (nice-to-have, but locked in by contract) --------------


def test_add_panel_accepts_tooltip(qapp) -> Any:
    tabs = SidePanelTabs(settings_key=_TEST_KEY)
    tabs.add_panel("Setup", QLabel("x"), tooltip="Configure simulation parameters")
    assert tabs.tabToolTip(0) == "Configure simulation parameters"
