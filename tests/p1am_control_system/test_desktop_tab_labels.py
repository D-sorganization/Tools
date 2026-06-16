"""P1AM desktop tab label consistency regressions (#3359)."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QTabWidget, QWidget  # noqa: E402

from p1am_control_system.desktop.layout_settings import (  # noqa: E402
    read_tab_visibility,
    write_tab_visibility,
)
from p1am_control_system.desktop.main_window import HMIMainWindow  # noqa: E402
from p1am_control_system.desktop.settings_tab import SettingsTab  # noqa: E402
from p1am_control_system.desktop.tab_labels import TAB_ORDER, TAB_TITLES  # noqa: E402


def test_settings_checkboxes_match_tab_titles(qapp) -> None:
    """Settings checkboxes must reuse the tab-bar labels exactly."""
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


def test_settings_tab_reports_and_restores_visibility_without_emitting(qapp) -> None:
    """Programmatic restore updates checkbox state without emitting user signals."""
    settings = SettingsTab()
    emitted: list[tuple[str, bool]] = []
    settings.tabVisibilityChanged.connect(
        lambda key, visible: emitted.append((key, visible))
    )

    settings.set_tab_visible("history", False, emit=False)

    assert settings.chk_history.isChecked() is False
    assert settings.visible_tabs()["history"] is False
    assert emitted == []


def test_tab_visibility_round_trips_through_qsettings(tmp_path) -> None:
    """Tab visibility writes survive a fresh QSettings reader."""
    from PyQt6.QtCore import QSettings

    settings_path = tmp_path / "p1am.ini"
    writer = QSettings(str(settings_path), QSettings.Format.IniFormat)
    write_tab_visibility(
        writer,
        {
            "mimic": True,
            "trends": False,
            "control": True,
            "routing": False,
            "history": False,
        },
    )
    writer.sync()

    reader = QSettings(str(settings_path), QSettings.Format.IniFormat)

    assert read_tab_visibility(reader, "mimic") is True
    assert read_tab_visibility(reader, "trends") is False
    assert read_tab_visibility(reader, "routing") is False
    assert read_tab_visibility(reader, "history") is False


def test_main_window_restores_persisted_tab_visibility(
    tmp_path, qapp, qtbot, monkeypatch
) -> None:
    """HMIMainWindow applies persisted tab visibility during startup restore."""
    from PyQt6.QtCore import QSettings

    from p1am_control_system.desktop.event_logger import EventLogger

    settings_path = tmp_path / "p1am-window.ini"
    writer = QSettings(str(settings_path), QSettings.Format.IniFormat)
    write_tab_visibility(
        writer,
        {
            "mimic": True,
            "trends": True,
            "control": True,
            "routing": True,
            "history": False,
        },
    )
    writer.sync()

    monkeypatch.setattr(
        "p1am_control_system.desktop.main_window.make_hmi_settings",
        lambda: QSettings(str(settings_path), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr(
        "p1am_control_system.desktop.main_window.EventLogger",
        lambda: EventLogger(str(tmp_path / "events.db")),
    )
    monkeypatch.setattr(HMIMainWindow, "_load_routing_config", lambda _self: None)

    class _Signal:
        def connect(self, _callback) -> None:
            """Accept signal connections without starting Qt work."""
            return None

    class _FakeWebSocketThread:
        messageReceived = _Signal()
        connectionStatusChanged = _Signal()

        def __init__(self, _uri: str) -> None:
            """Record no state; the fake never connects to a socket."""
            return None

        def start(self) -> None:
            """Skip background startup."""
            return None

        def stop(self) -> None:
            """Skip background shutdown."""
            return None

        def wait(self) -> None:
            """Return immediately like an already-stopped thread."""
            return None

    monkeypatch.setattr(
        "p1am_control_system.desktop.main_window.WebSocketClientThread",
        _FakeWebSocketThread,
    )

    window = HMIMainWindow()
    qtbot.addWidget(window)

    assert window.settings_tab.chk_history.isChecked() is False
    assert window.tab_widget.indexOf(window.event_log_viewer) == -1


def test_history_reinsert_order_precedes_settings(qapp) -> None:
    """History tab reinsertion keeps canonical order before Settings."""
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
