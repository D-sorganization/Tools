# mypy: ignore-errors
# ruff: noqa: E402, E501
import asyncio
import json
import logging
import os
import time
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QDockWidget,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Load properties from .env if present (satisfying python-dotenv instruction)
load_dotenv()

# Relative package imports
# Import Sidekick Unified Tools Sidebar
from p1am_control_system.desktop.alarm_state import (
    AlarmEventDebouncer,
    AlarmStateMachine,
    InterlockLimitError,
    interlock_for_index,
    validate_interlocks,
)
from p1am_control_system.desktop.connection_state import (
    CONNECTED,
    OFFLINE,
    derive_connection_status,
)
from p1am_control_system.desktop.control_tab import ControlTab
from p1am_control_system.desktop.event_logger import EventLogger, EventLogViewerWidget
from p1am_control_system.desktop.header import HMIHeader
from p1am_control_system.desktop.layout_settings import (
    make_hmi_settings,
    persist_window_settings,
    restore_window_settings,
)
from p1am_control_system.desktop.mimic_tab import MimicTab
from p1am_control_system.desktop.routing_tab import RoutingTab
from p1am_control_system.desktop.settings_tab import SettingsTab
from p1am_control_system.desktop.sidebar import InspectorSidebar
from p1am_control_system.desktop.tab_labels import TAB_ORDER, TAB_TITLES
from p1am_control_system.desktop.trends_tab import TrendsTab
from p1am_control_system.desktop.workers import HttpWorker, start_http_request
from shared.python.sidekick.ui.tools_sidebar import UnifiedToolsSidebar
from shared.python.theme.theme_manager import get_theme_manager

logger = logging.getLogger("p1am_control.desktop.main_window")

try:
    import websockets as _websockets_import
except ImportError:
    _websockets: Any | None = None
else:
    _websockets = _websockets_import


class WebSocketClientThread(QThread):
    """Asynchronous background thread to receive live telemetry stream from FastAPI middleware."""

    messageReceived = pyqtSignal(dict)
    connectionStatusChanged = pyqtSignal(str)  # "Connected", "Offline", "Simulating"

    def __init__(self, uri: str) -> None:
        super().__init__()
        self.uri = uri
        self.running = True
        self.loop = None

    def run(self) -> None:
        # Create new event loop for this QThread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._listen())

    async def _listen(self) -> None:
        self.connectionStatusChanged.emit(OFFLINE)
        if _websockets is None:
            logger.error("websockets dependency is not installed")
            return

        while self.running:
            try:
                logger.info(f"Connecting to WebSocket: {self.uri}")
                async with _websockets.connect(self.uri) as websocket:
                    # The link is up. Whether the values behind it are live or
                    # simulated is decided per-frame by
                    # ``derive_connection_status`` — the socket opening is not
                    # evidence of a simulation, and labelling a live plant
                    # "Simulating" is the dangerous direction (issue #4019).
                    self.connectionStatusChanged.emit(CONNECTED)
                    logger.info("WebSocket connection established.")
                    while self.running:
                        message = await websocket.recv()
                        payload = json.loads(message)
                        self.messageReceived.emit(payload)
            except Exception as e:
                logger.error(f"WebSocket client error: {e}")
                self.connectionStatusChanged.emit(OFFLINE)

            if not self.running:
                break
            await asyncio.sleep(2.0)  # Reconnect cooldown

    def stop(self) -> None:
        self.running = False
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)


class HMIMainWindow(QMainWindow):
    """Main HMI desktop window coordinating plant mimic, trends, PID loop tuning,
    alarm logic, event logging, and Sidekick docking.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("hmi_main_window")
        self.setWindowTitle("P1AM Control System HMI")
        self.resize(1400, 850)

        # Read environment configuration
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        ws_host = self.backend_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        self.ws_uri = f"{ws_host}/api/stream"

        # Current configurations
        self.routing_config = None
        self.user_role = "Operator"

        # Alarm annunciation. The state machine owns the active/unacknowledged
        # sets; the header's colour follows "active" and its flashing follows
        # "unacknowledged" (issue #4012).
        self.alarm_state = AlarmStateMachine()
        # Snapshot of what the header was last showing, so ACK acknowledges the
        # alarms the operator actually saw rather than blanket-clearing ones
        # that arrived between the render and the click.
        self._annunciated_alarms: frozenset = frozenset()
        # Coalesce chattering alarm transitions before they hit SQLite (#4022).
        self.alarm_event_debouncer = AlarmEventDebouncer(
            window_s=float(os.getenv("HMI_ALARM_EVENT_WINDOW_S", "5.0"))
        )

        # SQLite event database logger
        self.event_logger = EventLogger()

        self._init_ui()
        self.theme_manager = get_theme_manager(self)
        self.theme_manager.apply_theme_to_window(self)
        self.theme_manager.themeChanged.connect(self._on_theme_changed)
        self._on_theme_changed(self.theme_manager.get_current_theme_name())

        restore_window_settings(self, make_hmi_settings())
        self._purge_expired_events()
        self._load_routing_config()

        # Release coalesced alarm-event summaries even when the plant goes
        # quiet, so a burst's tail is never lost.
        self.alarm_flush_timer = QTimer(self)
        self.alarm_flush_timer.setInterval(1000)
        self.alarm_flush_timer.timeout.connect(self._flush_alarm_events)
        self.alarm_flush_timer.start()

        # Start WebSocket client stream thread.
        self.ws_thread = WebSocketClientThread(self.ws_uri)
        self.ws_thread.messageReceived.connect(self._on_telemetry_update)
        self.ws_thread.connectionStatusChanged.connect(
            self._on_connection_status_changed
        )
        self.ws_thread.start()

        # Log application startup
        self.log_event("INFO", "HMI Application started. Connecting to backend...")

    def _init_ui(self) -> None:
        # 1. Main layout container
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        # 2. Header
        self.header = HMIHeader(self)
        self.header.roleChanged.connect(self._on_role_changed)
        self.header.estopTriggered.connect(self._on_estop_triggered)
        self.header.alarmAcknowledgeClicked.connect(self._on_alarm_acknowledged)
        main_layout.addWidget(self.header)

        # 3. Central tabbed panel
        self.tab_widget = QTabWidget(self)
        main_layout.addWidget(self.tab_widget)

        # Create the sub-tabs
        self.mimic_tab = MimicTab(self)
        self.trends_tab = TrendsTab(self)
        self.control_tab = ControlTab(self)
        self.routing_tab = RoutingTab(self)
        self.event_log_viewer = EventLogViewerWidget(self.event_logger, self)
        self.settings_tab = SettingsTab(self)

        # Store references for dynamic hiding/showing
        self.tab_widgets = {
            "mimic": self.mimic_tab,
            "trends": self.trends_tab,
            "control": self.control_tab,
            "routing": self.routing_tab,
            "history": self.event_log_viewer,
            "settings": self.settings_tab,
        }
        self.tab_titles = TAB_TITLES

        # Connect settings tab visibility toggles
        self.settings_tab.tabVisibilityChanged.connect(self._handle_tab_visibility)

        # The History table is only requeried when it is actually on screen
        # (issue #4022); refresh it the moment it becomes current instead.
        self.tab_widget.currentChanged.connect(self._on_current_tab_changed)

        # Add tabs initially
        self.tab_widget.addTab(self.mimic_tab, self.tab_titles["mimic"])
        self.tab_widget.addTab(self.trends_tab, self.tab_titles["trends"])
        self.tab_widget.addTab(self.control_tab, self.tab_titles["control"])
        self.tab_widget.addTab(self.routing_tab, self.tab_titles["routing"])
        self.tab_widget.addTab(self.event_log_viewer, self.tab_titles["history"])
        self.tab_widget.addTab(self.settings_tab, self.tab_titles["settings"])

        # 4. Bottom Event Log
        self.log_list = QListWidget(self)
        self.log_list.setMaximumHeight(120)
        self.log_list.setStyleSheet("font-family: Consolas, monospace;")
        main_layout.addWidget(self.log_list)

        # 5. Inspector Sidebar Dock (Left)
        self.inspector_dock = QDockWidget("DCS Inspector Panel", self)
        self.inspector_sidebar = InspectorSidebar(self)
        self.inspector_sidebar.configUpdated.connect(self._load_routing_config)
        self.inspector_dock.setWidget(self.inspector_sidebar)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.inspector_dock)

        # Connect Mimic selection to Inspector
        self.mimic_tab.elementSelected.connect(self.inspector_sidebar.select_element)

        # 6. Sidekick (UnifiedToolsSidebar) Dock (Right)
        # Toggled by Ctrl+B / Ctrl+Shift+B registered shortcut
        self.sidekick_sidebar = UnifiedToolsSidebar(parent=self)
        self.sidekick_sidebar.install_as_dock(self, area="right")
        self.sidekick_sidebar.register_shortcuts(self)

    @pyqtSlot(str)
    def _on_theme_changed(self, theme_name: str) -> None:
        """Re-apply HMI widget-local styles when the shared theme changes."""
        if hasattr(self, "header"):
            self.header.apply_theme_styles(theme_name)
        if hasattr(self, "log_list"):
            self.log_list.setStyleSheet("font-family: Consolas, monospace;")

    def _load_routing_config(self) -> None:
        """Fetch current PLC configuration routing parameters from FastAPI."""
        worker = HttpWorker(
            "GET", f"{self.backend_url}/api/routing", timeout=1.5, parent=self
        )
        worker.success.connect(self._on_load_routing_config_success)
        worker.error.connect(self._on_load_routing_config_error)
        start_http_request(self, "routing_worker", worker)

    def _on_load_routing_config_success(self, data):
        from models import RoutingConfig

        self._apply_routing_config(RoutingConfig(**data))

    def _apply_routing_config(self, config) -> None:
        """Validate and adopt a freshly fetched routing configuration.

        Precondition: every interlock satisfies ``lolo <= low <= high <= hihi``.
        A configuration that violates it cannot be mapped onto the firmware's
        four-tier trip points, so it is rejected loudly (critical dialog +
        ALARM event) rather than silently annunciating the wrong severity
        (issue #4019).
        """
        try:
            validate_interlocks(getattr(config, "interlocks", {}))
        except (InterlockLimitError, TypeError) as exc:
            logger.error("Rejected routing configuration: %s", exc)
            self.log_event("ALARM", f"Invalid interlock configuration rejected: {exc}")
            QMessageBox.critical(
                self,
                "Invalid interlock configuration",
                "The PLC returned alarm setpoints that are not ordered "
                "lolo <= low <= high <= hihi, so the HMI cannot annunciate "
                f"severity consistently with the firmware.\n\n{exc}",
            )
            return

        self.routing_config = config

        # Pass configuration to sub-widgets
        self.routing_tab.set_routing_config(self.routing_config)
        self.control_tab.set_routing_config(self.routing_config)
        self.inspector_sidebar.set_routing_config(self.routing_config)

        logger.info("Successfully fetched and loaded PLC Routing configuration.")

    def _on_load_routing_config_error(self, err_msg):
        logger.error(
            f"Could not reach backend to load routing configuration: {err_msg}"
        )

    @pyqtSlot(str, bool)
    def _handle_tab_visibility(self, tab_key: str, visible: bool) -> None:
        widget = self.tab_widgets[tab_key]
        title = self.tab_titles[tab_key]

        if visible:
            if self.tab_widget.indexOf(widget) == -1:
                # Insert at appropriate index keeping order
                target_idx = 0
                for key in TAB_ORDER:
                    if key == tab_key:
                        break
                    w = self.tab_widgets.get(key)
                    if w and self.tab_widget.indexOf(w) != -1:
                        target_idx = self.tab_widget.indexOf(w) + 1
                self.tab_widget.insertTab(target_idx, widget, title)
        else:
            idx = self.tab_widget.indexOf(widget)
            if idx != -1:
                self.tab_widget.removeTab(idx)

    @pyqtSlot(dict)
    def _on_telemetry_update(self, payload: dict) -> None:
        """Processes real-time Modbus telemetry packet (10Hz)."""
        # Connectivity is a property of the frame, not of the socket handshake
        # (issue #4019); derive it before the empty-frame early return.
        self._on_connection_status_changed(derive_connection_status(payload))

        tags = payload.get("tags", [])
        if not tags:
            return

        # 1. Update views
        self.mimic_tab.update_telemetry(tags)
        self.control_tab.update_telemetry(tags)

        timestamp = time.time()
        self.trends_tab.add_telemetry_point(timestamp, tags)

        # 2. Evaluate Alarm/Interlocks logic against the deployed four-tier
        #    trip points, then repaint the annunciator.
        if self.routing_config:
            self._evaluate_alarms(tags)
            self._refresh_annunciator()

    def _evaluate_alarms(self, tags) -> None:
        """Fold one telemetry frame into the alarm state machine."""
        interlocks = getattr(self.routing_config, "interlocks", None)
        if not interlocks:
            return

        for tag_id, val in enumerate(tags):
            interlock = interlock_for_index(interlocks, tag_id)
            if interlock is None:
                continue
            try:
                transitions = self.alarm_state.evaluate(tag_id, val, interlock)
            except (TypeError, ValueError) as exc:
                # ValueError == non-finite reading (sensor fault). Skipping the
                # tag is the fail-safe outcome: `evaluate` validates before it
                # clears anything, so an alarm already latched for this tag stays
                # latched instead of being resolved by a NaN that compares False
                # against every limit.
                logger.error("Skipping tag %s with unusable value: %s", tag_id, exc)
                continue
            for transition in transitions:
                self._record_alarm_transition(transition)

    def _record_alarm_transition(self, transition) -> None:
        """Log an alarm edge, coalescing repeats from a chattering tag."""
        if transition.kind == "raised":
            level = "ALARM"
            message = f"CRITICAL: {transition.message}"
        else:
            level = "CLEAR"
            message = transition.message

        key = (transition.tag_id, transition.alarm_type, transition.kind)
        for event_level, event_message in self.alarm_event_debouncer.submit(
            key, level, message
        ):
            self.log_event(event_level, event_message)

    def _flush_alarm_events(self) -> None:
        """Release coalesced alarm-event summaries whose window has expired."""
        for event_level, event_message in self.alarm_event_debouncer.flush():
            self.log_event(event_level, event_message)

    def _refresh_annunciator(self) -> None:
        """Repaint the header from the alarm state and snapshot what it shows."""
        state = self.alarm_state.annunciator_state()
        self._annunciated_alarms = frozenset(self.alarm_state.unacknowledged_alarms)
        self.header.set_alarms_state(
            has_hl=state.has_hl,
            has_hhll=state.has_hhll,
            unacked_hl=state.unacked_hl,
            unacked_hhll=state.unacked_hhll,
        )

    def _purge_expired_events(self) -> None:
        """Apply the event-log retention window at startup."""
        try:
            retention_days = int(os.getenv("EVENT_LOG_RETENTION_DAYS", "90"))
            removed = self.event_logger.purge_older_than(retention_days)
        except Exception as exc:
            logger.error("Event-log retention purge failed: %s", exc)
            return
        if removed:
            logger.info(
                "Purged %d event-log rows older than %d days", removed, retention_days
            )

    @pyqtSlot(int)
    def _on_current_tab_changed(self, _index: int) -> None:
        """Refresh the History table only when it becomes the visible tab."""
        if self.tab_widget.currentWidget() is self.event_log_viewer:
            self._refresh_event_log_viewer()

    def _refresh_event_log_viewer(self) -> None:
        """Requery the History table, flushing queued writes first."""
        try:
            self.event_logger.flush_async(timeout=2.0)
            self.event_log_viewer.apply_filters()
        except Exception as exc:
            logger.error("Failed to refresh the event-log viewer: %s", exc)

    def log_event(self, level: str, msg: str) -> None:
        """Formats and appends a color-coded event string to the bottom list widget and SQLite db."""
        timestamp = time.strftime("%H:%M:%S")
        item_text = f"[{timestamp}] [{level}] {msg}"
        item = QListWidgetItem(item_text)

        # Color schemes matching severity
        severity = "INFO"
        event_type = "info"
        if level == "ALARM":
            item.setForeground(QColor(Qt.GlobalColor.red))
            severity = "CRITICAL"
            event_type = "alarm_trip"
        elif level == "CLEAR":
            item.setForeground(QColor(Qt.GlobalColor.darkGreen))
            severity = "INFO"
            event_type = "alarm_clear"
        elif level == "ACTION":
            item.setForeground(QColor(Qt.GlobalColor.blue))
            severity = "INFO"
            event_type = "user_action"
        elif level == "WARN":
            item.setForeground(QColor(Qt.GlobalColor.darkYellow))
            severity = "WARNING"
            event_type = "warning"
        else:
            item.setForeground(self.palette().color(QPalette.ColorRole.WindowText))
            severity = "INFO"
            event_type = "info"

        self.log_list.addItem(item)
        self.log_list.scrollToBottom()

        # Database logging. Queued onto the writer thread so the Qt GUI thread
        # never blocks on an fsync-backed commit while an alarm is active
        # (issue #4022), and the History table is only requeried when the
        # operator is actually looking at it.
        try:
            self.event_logger.log_event_async(
                event_type=event_type,
                severity=severity,
                operator=self.user_role,
                description=msg,
            )
        except Exception as e:
            logger.error(f"Failed to log event to database: {e}")
            return

        if self.tab_widget.currentWidget() is self.event_log_viewer:
            self._refresh_event_log_viewer()

    @pyqtSlot(str)
    def _on_role_changed(self, role: str) -> None:
        self.user_role = role
        # Propagate role settings to sidebar, routing matrix & control tab
        self.inspector_sidebar.set_role(role)
        self.routing_tab.set_role(role)
        self.control_tab.set_role(role)
        self.log_event("ACTION", f"User logged in role switched to: {role}")

    @pyqtSlot(bool)
    def _on_estop_triggered(self, active: bool) -> None:
        if active:
            worker = HttpWorker(
                "POST", f"{self.backend_url}/api/estop", timeout=0.5, parent=self
            )
            worker.success.connect(self._on_estop_success)
            worker.error.connect(self._on_estop_error)
            start_http_request(self, "estop_worker", worker)
        else:
            worker = HttpWorker(
                "POST",
                f"{self.backend_url}/api/estop/clear",
                timeout=0.5,
                parent=self,
            )
            worker.success.connect(self._on_estop_clear_success)
            worker.error.connect(self._on_estop_clear_error)
            start_http_request(self, "estop_clear_worker", worker)

    def _on_estop_clear_success(self, data) -> None:
        # Only now is the plant confirmed released; let the header go green.
        self.header.confirm_estop_cleared()
        self.log_event("ACTION", "E-STOP state cleared. Normal operations resumed.")

    def _on_estop_clear_error(self, err_msg) -> None:
        # PLC did not acknowledge the reset: keep the header tripped (red) so the
        # operator is not misled into thinking the plant was released.
        self.header.revert_estop_to_tripped()
        self.log_event("ALARM", f"Failed to clear E-STOP state: {err_msg}")

    def _on_estop_success(self, data):
        self.log_event("ALARM", "EMERGENCY STOP SHUTDOWN COMMAND SENT TO PLC.")

    def _on_estop_error(self, err_msg):
        self.log_event("ALARM", f"Failed to send E-STOP command: {err_msg}")

    @pyqtSlot()
    def _on_alarm_acknowledged(self) -> None:
        """Acknowledge exactly the alarms the header was showing.

        Acknowledging silences the flash but leaves every still-active alarm
        annunciated (steady), and an alarm that arrived after the last repaint
        stays unacknowledged so it is never silently swallowed (issue #4012).
        """
        acknowledged = self.alarm_state.acknowledge(self._annunciated_alarms)
        self._refresh_annunciator()
        if acknowledged:
            summary = ", ".join(f"Tag {tag} {kind}" for tag, kind in acknowledged)
            self.log_event("ACTION", f"Alarms acknowledged by operator: {summary}.")
        else:
            self.log_event(
                "ACTION", "Alarm acknowledge pressed; nothing to acknowledge."
            )

    @pyqtSlot(str)
    def _on_connection_status_changed(self, status: str) -> None:
        # Map websocket state to industrial header connection state
        # In mock middleware, if WebSocket connects it simulates modbus state.
        self.header.set_connection_status(status)
        logger.info(f"HMI Connection State Changed: {status}")

    def closeEvent(self, event) -> None:
        persist_window_settings(self, make_hmi_settings())
        # Stop background thread on close
        self.ws_thread.stop()
        self.ws_thread.wait()
        # Stop the periodic flush before tearing anything down: a timeout that
        # fires after the widgets are gone would touch deleted C++ objects.
        timer = getattr(self, "alarm_flush_timer", None)
        if timer is not None:
            timer.stop()
        # Release any coalesced alarm summaries, then drain the writer thread so
        # no queued event is lost on shutdown.
        try:
            self._flush_alarm_events()
            self.event_logger.close()
        except Exception as exc:  # pragma: no cover - shutdown safety net
            logger.error("Failed to drain the event log on close: %s", exc)
        super().closeEvent(event)


# Alias for gui launcher / launcher loading compatibility
P1AMMainWindow = HMIMainWindow
