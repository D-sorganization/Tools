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


from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QDockWidget,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Load properties from .env if present (satisfying python-dotenv instruction)
load_dotenv()

# Relative package imports
# Import Sidekick Unified Tools Sidebar
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
        self.connectionStatusChanged.emit("Offline")
        if _websockets is None:
            logger.error("websockets dependency is not installed")
            return

        while self.running:
            try:
                logger.info(f"Connecting to WebSocket: {self.uri}")
                async with _websockets.connect(self.uri) as websocket:
                    self.connectionStatusChanged.emit(
                        "Simulating"
                    )  # Default simulated state, updated on active Modbus
                    logger.info("WebSocket connection established.")
                    while self.running:
                        message = await websocket.recv()
                        payload = json.loads(message)
                        self.messageReceived.emit(payload)
            except Exception as e:
                logger.error(f"WebSocket client error: {e}")
                self.connectionStatusChanged.emit("Offline")

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

        # Alarm tracking state
        # Set of active alarms: (tag_id, alarm_type)
        self.active_alarms = set()
        # Set of active unacknowledged alarms: (tag_id, alarm_type)
        self.unacknowledged_alarms = set()

        # SQLite event database logger
        self.event_logger = EventLogger()

        self._init_ui()
        self.theme_manager = get_theme_manager(self)
        self.theme_manager.apply_theme_to_window(self)
        self.theme_manager.themeChanged.connect(self._on_theme_changed)
        self._on_theme_changed(self.theme_manager.get_current_theme_name())

        restore_window_settings(self, make_hmi_settings())
        self._load_routing_config()

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

        self.routing_config = RoutingConfig(**data)

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
        tags = payload.get("tags", [])
        if not tags:
            return

        # 1. Update views
        self.mimic_tab.update_telemetry(tags)
        self.control_tab.update_telemetry(tags)

        timestamp = time.time()
        self.trends_tab.add_telemetry_point(timestamp, tags)

        # 2. Evaluate Alarm/Interlocks logic
        if self.routing_config:
            for tag_id, val in enumerate(tags):
                if tag_id >= len(self.routing_config.interlocks):
                    break

                interlock = self.routing_config.interlocks[tag_id]
                low_limit = interlock.low_limit
                high_limit = interlock.high_limit

                # HH: High-High threshold (exceeds high limit + 5.0 units)
                # LL: Low-Low threshold (drops below low limit - 5.0 units)
                hh_thresh = high_limit + 5.0
                ll_thresh = low_limit - 5.0

                # Check LL
                if val <= ll_thresh:
                    self._trigger_alarm(
                        tag_id,
                        "LL",
                        f"Tag {tag_id} Low-Low limit violation ({val:.2f} <= {ll_thresh:.2f})",
                    )
                # Check HH
                elif val >= hh_thresh:
                    self._trigger_alarm(
                        tag_id,
                        "HH",
                        f"Tag {tag_id} High-High limit violation ({val:.2f} >= {hh_thresh:.2f})",
                    )
                # Check L
                elif val <= low_limit:
                    self._trigger_alarm(
                        tag_id,
                        "L",
                        f"Tag {tag_id} Low limit violation ({val:.2f} <= {low_limit:.2f})",
                    )
                # Check H
                elif val >= high_limit:
                    self._trigger_alarm(
                        tag_id,
                        "H",
                        f"Tag {tag_id} High limit violation ({val:.2f} >= {high_limit:.2f})",
                    )
                else:
                    # Clear active alarm if value returns to normal range
                    for alarm_type in ["LL", "HH", "L", "H"]:
                        alarm_key = (tag_id, alarm_type)
                        if alarm_key in self.active_alarms:
                            self.active_alarms.remove(alarm_key)
                            self.log_event(
                                "CLEAR", f"Tag {tag_id} alarm {alarm_type} cleared."
                            )

            # Update header alarm button flashing state
            # Flashing yellow for H/L, flashing red for HH/LL
            self.header.set_alarms_state(
                has_hl=len(
                    [a for a in self.unacknowledged_alarms if a[1] in ["H", "L"]]
                )
                > 0,
                has_hhll=len(
                    [a for a in self.unacknowledged_alarms if a[1] in ["HH", "LL"]]
                )
                > 0,
            )

    def _trigger_alarm(self, tag_id: int, alarm_type: str, message: str) -> None:
        alarm_key = (tag_id, alarm_type)
        if alarm_key not in self.active_alarms:
            self.active_alarms.add(alarm_key)
            self.unacknowledged_alarms.add(alarm_key)
            self.log_event("ALARM", f"CRITICAL: {message}")

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

        # Database Logging
        try:
            self.event_logger.log_event(
                event_type=event_type,
                severity=severity,
                operator=self.user_role,
                description=msg,
            )
            # Update viewer if tab is active
            if hasattr(self, "event_log_viewer"):
                self.event_log_viewer.apply_filters()
        except Exception as e:
            logger.error(f"Failed to log event to database: {e}")

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
        # Acknowledge all currently active alarms
        self.unacknowledged_alarms.clear()
        self.header.set_alarms_state(False, False)
        self.log_event("ACTION", "All active alarms acknowledged by operator.")

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
        super().closeEvent(event)


# Alias for gui launcher / launcher loading compatibility
P1AMMainWindow = HMIMainWindow
