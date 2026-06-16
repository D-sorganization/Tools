"""P1AM desktop tab label consistency regressions (#3359)."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QTabWidget, QWidget  # noqa: E402

from p1am_control_system.desktop.settings_tab import SettingsTab  # noqa: E402
from p1am_control_system.desktop.tab_labels import TAB_ORDER, TAB_TITLES  # noqa: E402


def test_settings_checkboxes_match_tab_titles(qapp) -> None:
    settings = SettingsTab()

    checkbox_text = {
        "mimic": settings.chk_mimic.text(),
        "trends": settings.chk_trends.text(),
        "control": settings.chk_control.text(),
        "routing": settings.chk_routing.text(),
        "history": settings.chk_history.text(),
    }

    assert checkbox_text == {
        key: TAB_TITLES[key]
        for key in ("mimic", "trends", "control", "routing", "history")
    }


def test_history_reinsert_order_precedes_settings(qapp) -> None:
    tab_widget = QTabWidget()
    tab_widgets = {key: QWidget() for key in TAB_ORDER}

    for key in TAB_ORDER:
        if key != "history":
            tab_widget.addTab(tab_widgets[key], TAB_TITLES[key])

    target_idx = 0
    for key in TAB_ORDER:
        if key == "history":
            break
        widget = tab_widgets[key]
        if tab_widget.indexOf(widget) != -1:
            target_idx = tab_widget.indexOf(widget) + 1
    tab_widget.insertTab(target_idx, tab_widgets["history"], TAB_TITLES["history"])

    assert tab_widget.indexOf(tab_widgets["history"]) == (
        tab_widget.indexOf(tab_widgets["routing"]) + 1
    )
    assert tab_widget.indexOf(tab_widgets["history"]) < tab_widget.indexOf(
        tab_widgets["settings"]
    )
