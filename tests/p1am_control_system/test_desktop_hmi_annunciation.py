"""Qt-level regressions for the P1AM desktop HMI operator surface.

Covers:

* **#4012** — the ACK button's colour follows the *active* alarm set and its
  flashing follows the *unacknowledged* set.
* **#4019b** — the connection label is driven by the telemetry frame.
* **#4021** — clearing the E-stop is Admin-gated and modally confirmed, and a
  declined confirmation reverts the button to its tripped state.
* **#4022** — the History table is only requeried when that tab is on screen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import QMessageBox, QWidget  # noqa: E402

from p1am_control_system.desktop import header as header_module  # noqa: E402
from p1am_control_system.desktop import main_window as hmi_main_window  # noqa: E402
from p1am_control_system.desktop.header import HMIHeader  # noqa: E402
from p1am_control_system.desktop.main_window import HMIMainWindow  # noqa: E402


class _Interlock:
    def __init__(self, lolo: float, low: float, high: float, hihi: float) -> None:
        self.lolo_limit = lolo
        self.low_limit = low
        self.high_limit = high
        self.hihi_limit = hihi


class _RoutingConfig:
    def __init__(self, interlocks) -> None:
        self.interlocks = interlocks


# ---------------------------------------------------------------------------
# Header: alarm annunciation (#4012)
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_acknowledged_active_alarm_shows_steady_colour(qapp) -> None:
    """An acked-but-active alarm stays coloured and stops toggling."""
    header = HMIHeader()
    header.set_alarms_state(
        has_hl=False, has_hhll=True, unacked_hl=False, unacked_hhll=False
    )

    steady = header.ack_btn.styleSheet()
    assert "background-color" in steady, "acked active alarm must stay visible"

    header._toggle_flash()
    assert header.ack_btn.styleSheet() == steady
    header._toggle_flash()
    assert header.ack_btn.styleSheet() == steady


@pytest.mark.gui
def test_unacknowledged_alarm_still_flashes(qapp) -> None:
    header = HMIHeader()
    header.set_alarms_state(
        has_hl=False, has_hhll=True, unacked_hl=False, unacked_hhll=True
    )

    header._toggle_flash()
    first = header.ack_btn.styleSheet()
    header._toggle_flash()
    assert header.ack_btn.styleSheet() != first


@pytest.mark.gui
def test_repeated_state_pushes_do_not_pin_the_flash_on(qapp) -> None:
    """A telemetry frame must not restart the flash phase.

    ``_refresh_annunciator`` calls ``set_alarms_state`` on EVERY frame (~10 Hz).
    If that resets ``_flash_state`` to True, the timer's OFF phase is overwritten
    within ~100 ms and an unacknowledged alarm renders steady — silently erasing
    the flash-vs-steady distinction that tells an operator whether anyone has
    seen the alarm. Only a transition INTO unacknowledged may restart the phase.
    """
    header = HMIHeader()
    unacked = {
        "has_hl": False,
        "has_hhll": True,
        "unacked_hl": False,
        "unacked_hhll": True,
    }
    header.set_alarms_state(**unacked)

    # Drop to the OFF phase, then let the stream push the same state repeatedly.
    header._toggle_flash()
    off_phase = header.ack_btn.styleSheet()
    for _ in range(10):
        header.set_alarms_state(**unacked)
    assert header.ack_btn.styleSheet() == off_phase, (
        "an unchanged alarm state re-pushed at frame rate must not restart the "
        "flash cycle — the alarm would render effectively steady"
    )

    # The timer must still be able to alternate.
    header._toggle_flash()
    assert header.ack_btn.styleSheet() != off_phase


@pytest.mark.gui
def test_a_newly_unacknowledged_alarm_starts_on_the_visible_phase(qapp) -> None:
    """A fresh alarm must annunciate immediately, not wait for the timer."""
    header = HMIHeader()
    header.set_alarms_state(
        has_hl=False, has_hhll=True, unacked_hl=False, unacked_hhll=False
    )
    header._toggle_flash()  # acked+active is steady; phase is irrelevant here

    header.set_alarms_state(
        has_hl=False, has_hhll=True, unacked_hl=False, unacked_hhll=True
    )
    assert header._flash_state is True
    assert "background-color" in header.ack_btn.styleSheet()


@pytest.mark.gui
def test_no_active_alarm_clears_the_button(qapp) -> None:
    header = HMIHeader()
    header.set_alarms_state(
        has_hl=True, has_hhll=False, unacked_hl=True, unacked_hhll=False
    )
    header.set_alarms_state(
        has_hl=False, has_hhll=False, unacked_hl=False, unacked_hhll=False
    )
    assert "background-color" not in header.ack_btn.styleSheet()
    header._toggle_flash()
    assert "background-color" not in header.ack_btn.styleSheet()


# ---------------------------------------------------------------------------
# Header: E-stop clear guard (#4021)
# ---------------------------------------------------------------------------


def _trip_estop(header: HMIHeader) -> list[bool]:
    emitted: list[bool] = []
    header.estopTriggered.connect(emitted.append)
    header.estop_btn.setChecked(True)
    assert emitted == [True], "tripping must stay immediate and unguarded"
    emitted.clear()
    return emitted


@pytest.mark.gui
def test_estop_clear_is_rejected_for_operators(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    header = HMIHeader()
    header.set_role("Operator")
    emitted = _trip_estop(header)

    monkeypatch.setattr(header_module.QMessageBox, "critical", lambda *a, **k: None)
    monkeypatch.setattr(
        header_module.QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )

    header.estop_btn.setChecked(False)

    assert emitted == [], "an Operator must not be able to clear the E-stop"
    assert header.estop_btn.isChecked() is True


@pytest.mark.gui
def test_estop_clear_declined_reverts_the_button(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    header = HMIHeader()
    header.set_role("Admin")
    emitted = _trip_estop(header)

    titles: list[str] = []

    def _question(_parent, title, _text, *_a, **_k):
        titles.append(title)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(header_module.QMessageBox, "question", _question)

    header.estop_btn.setChecked(False)

    assert titles, "clearing the E-stop must be modally confirmed"
    assert emitted == []
    assert header.estop_btn.isChecked() is True


@pytest.mark.gui
def test_estop_clear_confirmed_by_admin_emits(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    header = HMIHeader()
    header.set_role("Admin")
    emitted = _trip_estop(header)

    monkeypatch.setattr(
        header_module.QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )

    header.estop_btn.setChecked(False)

    assert emitted == [False]


# ---------------------------------------------------------------------------
# Main window integration
# ---------------------------------------------------------------------------


class _FakeHeader(QWidget):
    roleChanged = pyqtSignal(str)
    estopTriggered = pyqtSignal(bool)
    alarmAcknowledgeClicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.alarm_states: list[tuple] = []
        self.connection_states: list[str] = []

    def apply_theme_styles(self, _theme_name: str) -> None:
        return None

    def set_alarms_state(
        self,
        has_hl: bool,
        has_hhll: bool,
        unacked_hl: bool | None = None,
        unacked_hhll: bool | None = None,
    ) -> None:
        self.alarm_states.append((has_hl, has_hhll, unacked_hl, unacked_hhll))

    def set_connection_status(self, status: str) -> None:
        self.connection_states.append(status)

    def set_role(self, _role: str) -> None:
        return None


class _FakeRoutingTab(QWidget):
    def set_routing_config(self, _config) -> None:
        return None

    def set_role(self, _role: str) -> None:
        return None


class _FakeControlTab(_FakeRoutingTab):
    def update_telemetry(self, _tags) -> None:
        return None


class _FakeMimicTab(QWidget):
    elementSelected = pyqtSignal(object)

    def update_telemetry(self, _tags) -> None:
        return None


class _FakeTrendsTab(QWidget):
    def add_telemetry_point(self, _timestamp: float, _tags) -> None:
        return None


class _FakeInspectorSidebar(QWidget):
    configUpdated = pyqtSignal()

    def select_element(self, _element) -> None:
        return None

    def set_routing_config(self, _config) -> None:
        return None

    def set_role(self, _role: str) -> None:
        return None


class _FakeSidekickSidebar(QWidget):
    def install_as_dock(self, _window, *, area: str = "right") -> None:
        return None

    def register_shortcuts(self, _window) -> None:
        return None


class _FakeEventLogViewer(QWidget):
    def __init__(self, _event_logger, parent=None) -> None:
        super().__init__(parent)
        self.refreshes = 0

    def apply_filters(self) -> None:
        self.refreshes += 1


class _Signal:
    def connect(self, _callback) -> None:
        return None


class _FakeWebSocketThread:
    messageReceived = _Signal()
    connectionStatusChanged = _Signal()

    def __init__(self, _uri: str) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def wait(self) -> None:
        return None


@pytest.fixture
def hmi_window(qapp, qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fully faked HMIMainWindow with no network, no theme churn, no plots."""
    from p1am_control_system.desktop.event_logger import EventLogger

    monkeypatch.setenv("EVENT_LOG_DB_PATH", str(tmp_path / "events.db"))
    monkeypatch.setattr(
        hmi_main_window, "EventLogger", lambda: EventLogger(str(tmp_path / "events.db"))
    )
    monkeypatch.setattr(HMIMainWindow, "_load_routing_config", lambda _self: None)
    monkeypatch.setattr(hmi_main_window, "HMIHeader", _FakeHeader)
    monkeypatch.setattr(hmi_main_window, "MimicTab", _FakeMimicTab)
    monkeypatch.setattr(hmi_main_window, "RoutingTab", _FakeRoutingTab)
    monkeypatch.setattr(hmi_main_window, "ControlTab", _FakeControlTab)
    monkeypatch.setattr(hmi_main_window, "TrendsTab", _FakeTrendsTab)
    monkeypatch.setattr(hmi_main_window, "InspectorSidebar", _FakeInspectorSidebar)
    monkeypatch.setattr(hmi_main_window, "UnifiedToolsSidebar", _FakeSidekickSidebar)
    monkeypatch.setattr(hmi_main_window, "EventLogViewerWidget", _FakeEventLogViewer)
    monkeypatch.setattr(hmi_main_window, "WebSocketClientThread", _FakeWebSocketThread)

    window = HMIMainWindow()
    qtbot.addWidget(window)
    yield window
    window.event_logger.close()


@pytest.mark.gui
def test_main_window_trips_hh_at_the_configured_hihi_limit(hmi_window) -> None:
    """#4019a: HMI severity must match the firmware's trip point."""
    hmi_window.routing_config = _RoutingConfig(
        {"TAG_0": _Interlock(3.0, 5.0, 95.0, 97.0)}
    )

    hmi_window._on_telemetry_update({"tags": [98.0]})

    assert (0, "HH") in hmi_window.alarm_state.active_alarms
    assert (0, "H") not in hmi_window.alarm_state.active_alarms
    assert hmi_window.header.alarm_states[-1][1] is True


@pytest.mark.gui
def test_main_window_keeps_acked_active_alarm_annunciated(hmi_window) -> None:
    """#4012a: acking to silence must not blind the operator."""
    hmi_window.routing_config = _RoutingConfig(
        {"TAG_0": _Interlock(3.0, 5.0, 95.0, 97.0)}
    )
    hmi_window._on_telemetry_update({"tags": [99.0]})

    hmi_window._on_alarm_acknowledged()
    hmi_window._on_telemetry_update({"tags": [99.0]})

    has_hl, has_hhll, unacked_hl, unacked_hhll = hmi_window.header.alarm_states[-1]
    assert has_hhll is True
    assert unacked_hhll is False


@pytest.mark.gui
def test_main_window_clears_alarm_from_both_sets(hmi_window) -> None:
    """#4012b: a long-cleared alarm must stop flashing the ACK button."""
    hmi_window.routing_config = _RoutingConfig(
        {"TAG_0": _Interlock(3.0, 5.0, 95.0, 97.0)}
    )
    hmi_window._on_telemetry_update({"tags": [99.0]})
    hmi_window._on_telemetry_update({"tags": [50.0]})

    assert hmi_window.alarm_state.active_alarms == set()
    assert hmi_window.alarm_state.unacknowledged_alarms == set()
    assert hmi_window.header.alarm_states[-1] == (False, False, False, False)


@pytest.mark.gui
def test_main_window_ack_does_not_swallow_a_racing_alarm(hmi_window) -> None:
    """Only the alarms that were on screen at render time get acknowledged."""
    interlock = _Interlock(3.0, 5.0, 95.0, 97.0)
    hmi_window.routing_config = _RoutingConfig({"TAG_0": interlock, "TAG_1": interlock})
    hmi_window._on_telemetry_update({"tags": [99.0, 50.0]})

    # A second alarm arrives after the header was painted but before the click.
    hmi_window.alarm_state.evaluate(1, 99.0, interlock)

    hmi_window._on_alarm_acknowledged()

    assert hmi_window.alarm_state.unacknowledged_alarms == {(1, "HH")}


@pytest.mark.gui
def test_main_window_rejects_inverted_interlock_config(
    hmi_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#4019a: a config whose limits are out of order fails loudly."""
    shown: list[str] = []
    monkeypatch.setattr(
        hmi_main_window.QMessageBox,
        "critical",
        lambda _p, title, _text, *a, **k: shown.append(title),
    )

    class _Cfg:
        interlocks = {"TAG_0": _Interlock(0.0, 5.0, 95.0, 90.0)}

        def __init__(self) -> None:
            return None

    monkeypatch.setattr(hmi_main_window, "RoutingConfig", _Cfg, raising=False)
    hmi_window._apply_routing_config(_Cfg())

    assert shown, "an invalid interlock set must raise a visible alarm"
    assert hmi_window.routing_config is None


@pytest.mark.gui
def test_main_window_derives_connection_status_from_the_frame(hmi_window) -> None:
    """#4019b: a live plant is never labelled 'Simulating' by default."""
    hmi_window._on_telemetry_update({"tags": [1.0]})
    assert hmi_window.header.connection_states[-1] == "Connected"

    hmi_window._on_telemetry_update({"tags": [1.0], "simulated": True})
    assert hmi_window.header.connection_states[-1] == "Simulating"


@pytest.mark.gui
def test_history_table_is_not_requeried_while_another_tab_is_shown(
    hmi_window,
) -> None:
    """#4022: no full requery + repopulate per alarm event on the GUI thread."""
    hmi_window.tab_widget.setCurrentWidget(hmi_window.mimic_tab)
    before = hmi_window.event_log_viewer.refreshes

    for _ in range(5):
        hmi_window.log_event("ALARM", "chattering thermocouple")

    assert hmi_window.event_log_viewer.refreshes == before


@pytest.mark.gui
def test_history_table_refreshes_when_its_tab_becomes_current(hmi_window) -> None:
    hmi_window.tab_widget.setCurrentWidget(hmi_window.mimic_tab)
    before = hmi_window.event_log_viewer.refreshes

    hmi_window.tab_widget.setCurrentWidget(hmi_window.event_log_viewer)

    assert hmi_window.event_log_viewer.refreshes > before


@pytest.mark.gui
def test_dithering_tag_logs_one_event_not_one_per_scan(hmi_window) -> None:
    """#4022: repeated identical alarm transitions are coalesced."""
    interlock = _Interlock(3.0, 5.0, 95.0, 97.0)
    hmi_window.routing_config = _RoutingConfig({"TAG_0": interlock})

    for _ in range(10):
        hmi_window._on_telemetry_update({"tags": [99.0]})
        hmi_window._on_telemetry_update({"tags": [50.0]})

    hmi_window.event_logger.flush_async(timeout=10.0)
    alarm_rows = hmi_window.event_logger.fetch_logs(event_type="alarm_trip")
    assert len(alarm_rows) == 1, "a dithering tag must not spam the event log"
