# mypy: ignore-errors
# ruff: noqa: E501
import logging
import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .workers import HttpWorker, start_http_request

logger = logging.getLogger("p1am_control.desktop.sidebar")


class InspectorSidebar(QWidget):
    """Inspector sidebar panel. Displays parameters of the currently selected tag/element.
    Manual overrides, alarm limits (Admin only), PID gains, and routing configuration.
    """

    # Emitted when config is saved and routing configurations need reload
    configUpdated = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("inspector_sidebar")

        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

        # Selected element info
        self.selected_tag_id = -1
        self.selected_type = ""  # "reactor" or "valve"
        self.selected_name = ""
        self.selected_description = ""

        # User role
        self.user_role = "Operator"

        # Baseline values of the safety-limit / PID spin boxes as last loaded
        # from the routing config. Used to detect whether the user edited a
        # read-only control (QDoubleSpinBox has no isModified(); issue #3320).
        self._baseline_low_limit: float = 0.0
        self._baseline_pid_setpoint: float = 0.0

        # Current system configuration reference
        self.routing_config = None

        self._init_ui()

    def _init_ui(self) -> None:
        # Layout scrollable so it fits on small screens
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # 1. Header Information
        self.lbl_header = QLabel("No Selected Element", self)
        self.lbl_header.setStyleSheet("font-weight: bold; font-size: 12pt;")
        self.lbl_header.setWordWrap(True)
        layout.addWidget(self.lbl_header)

        # 2. General Details Card
        self.details_group = QGroupBox("General Details", self)
        details_grid = QGridLayout(self.details_group)

        details_grid.addWidget(QLabel("Tag ID:", self), 0, 0)
        self.lbl_tag_id = QLabel("--", self)
        details_grid.addWidget(self.lbl_tag_id, 0, 1)

        details_grid.addWidget(QLabel("Name:", self), 1, 0)
        self.txt_name = QLineEdit(self)
        self.txt_name.setReadOnly(True)
        details_grid.addWidget(self.txt_name, 1, 1)

        details_grid.addWidget(QLabel("Description:", self), 2, 0)
        self.txt_desc = QLineEdit(self)
        self.txt_desc.setReadOnly(True)
        details_grid.addWidget(self.txt_desc, 2, 1)

        layout.addWidget(self.details_group)

        # 3. Manual Override Card
        self.override_group = QGroupBox("Manual Force Override", self)
        override_grid = QGridLayout(self.override_group)

        self.chk_force_active = QCheckBox("Active Manual Force", self)
        self.chk_force_active.toggled.connect(self._on_force_toggled)
        override_grid.addWidget(self.chk_force_active, 0, 0, 1, 2)

        override_grid.addWidget(QLabel("Force Value:", self), 1, 0)
        self.spin_force_val = QDoubleSpinBox(self)
        self.spin_force_val.setRange(-9999.0, 9999.0)
        self.spin_force_val.setEnabled(False)
        override_grid.addWidget(self.spin_force_val, 1, 1)

        layout.addWidget(self.override_group)

        # 4. Safety Alarm Interlocks Card (Admin only)
        self.alarms_group = QGroupBox("Alarm Interlock Limits", self)
        alarms_grid = QGridLayout(self.alarms_group)

        # A limit of ``None`` (side disabled, the backend default for every
        # unrouted tag) is shown as the spin box's range end (the low box says
        # "disabled" there); reading that value back yields ``None`` again.
        alarms_grid.addWidget(QLabel("Low Limit (LL/L):", self), 0, 0)
        self.spin_low_limit = QDoubleSpinBox(self)
        self.spin_low_limit.setRange(-9999.0, 9999.0)
        self.spin_low_limit.setSpecialValueText("disabled")
        alarms_grid.addWidget(self.spin_low_limit, 0, 1)

        alarms_grid.addWidget(QLabel("High Limit (H/HH):", self), 1, 0)
        self.spin_high_limit = QDoubleSpinBox(self)
        self.spin_high_limit.setRange(-9999.0, 9999.0)
        alarms_grid.addWidget(self.spin_high_limit, 1, 1)

        layout.addWidget(self.alarms_group)

        # 5. PID Configuration Card (Admin only, only visible if tag is in loop)
        self.pid_group = QGroupBox("Associated PID Loop Config", self)
        pid_grid = QGridLayout(self.pid_group)

        pid_grid.addWidget(QLabel("Loop Setpoint:", self), 0, 0)
        self.spin_pid_sp = QDoubleSpinBox(self)
        self.spin_pid_sp.setRange(0.0, 1000.0)
        pid_grid.addWidget(self.spin_pid_sp, 0, 1)

        pid_grid.addWidget(QLabel("Gain (Kp):", self), 1, 0)
        self.spin_pid_kp = QDoubleSpinBox(self)
        self.spin_pid_kp.setDecimals(3)
        self.spin_pid_kp.setRange(0.0, 100.0)
        pid_grid.addWidget(self.spin_pid_kp, 1, 1)

        pid_grid.addWidget(QLabel("Integral (Ki):", self), 2, 0)
        self.spin_pid_ki = QDoubleSpinBox(self)
        self.spin_pid_ki.setDecimals(3)
        self.spin_pid_ki.setRange(0.0, 100.0)
        pid_grid.addWidget(self.spin_pid_ki, 2, 1)

        pid_grid.addWidget(QLabel("Derivative (Kd):", self), 3, 0)
        self.spin_pid_kd = QDoubleSpinBox(self)
        self.spin_pid_kd.setDecimals(3)
        self.spin_pid_kd.setRange(0.0, 100.0)
        pid_grid.addWidget(self.spin_pid_kd, 3, 1)

        layout.addWidget(self.pid_group)
        self.pid_group.setVisible(False)  # Hidden unless active loop selected

        # 6. Apply Button at bottom
        self.btn_apply = QPushButton("Apply Inspector Changes", self)
        self.btn_apply.setStyleSheet("font-weight: bold; height: 30px;")
        self.btn_apply.clicked.connect(self._apply_changes)
        layout.addWidget(self.btn_apply)

        layout.addStretch()

        # Initial Role Setup
        self.set_role(self.user_role)

    def set_role(self, role: str) -> None:
        self.user_role = role
        is_admin = role == "Admin"

        # Force overrides are allowed for Operator/Admin, but safety limits & PID is Admin only
        self.spin_low_limit.setEnabled(is_admin)
        self.spin_high_limit.setEnabled(is_admin)

        self.spin_pid_sp.setEnabled(is_admin)
        self.spin_pid_kp.setEnabled(is_admin)
        self.spin_pid_ki.setEnabled(is_admin)
        self.spin_pid_kd.setEnabled(is_admin)

    def set_routing_config(self, config) -> None:
        self.routing_config = config
        self._refresh_values()

    def select_element(
        self, tag_id: int, element_type: str, name: str, description: str
    ) -> None:
        """Loads a clicked mimic element's details and active config limits."""
        self.selected_tag_id = tag_id
        self.selected_type = element_type
        self.selected_name = name
        self.selected_description = description

        self.lbl_header.setText(f"{name} ({element_type.upper()})")
        self.lbl_tag_id.setText(str(tag_id))
        self.txt_name.setText(name)
        self.txt_desc.setText(description)

        # Reset override controls
        self.chk_force_active.setChecked(False)
        self.spin_force_val.setValue(0.0)
        self.spin_force_val.setEnabled(False)

        self._refresh_values()

    def _refresh_values(self) -> None:
        if self.selected_tag_id < 0 or not self.routing_config:
            return

        tag_id = self.selected_tag_id

        # 1. Load Interlock limits
        if tag_id < len(self.routing_config.interlocks):
            interlock = self.routing_config.interlocks[tag_id]
            self.spin_low_limit.setValue(
                self.spin_low_limit.minimum()
                if interlock.low_limit is None
                else interlock.low_limit
            )
            self.spin_high_limit.setValue(
                self.spin_high_limit.maximum()
                if interlock.high_limit is None
                else interlock.high_limit
            )
            self._baseline_low_limit = self.spin_low_limit.value()

        # 2. Check associated PID loop (either PV or CV tag matches)
        pid_found = None
        self.pid_loop_index = -1
        for idx, pid in enumerate(self.routing_config.pids):
            if pid.pv_tag_id == tag_id or pid.cv_tag_id == tag_id:
                pid_found = pid
                self.pid_loop_index = idx
                break

        if pid_found:
            self.pid_group.setVisible(True)
            self.pid_group.setTitle(f"Associated PID Loop {self.pid_loop_index} Config")
            self.spin_pid_sp.setValue(pid_found.setpoint)
            self.spin_pid_kp.setValue(pid_found.kp)
            self.spin_pid_ki.setValue(pid_found.ki)
            self.spin_pid_kd.setValue(pid_found.kd)
            self._baseline_pid_setpoint = self.spin_pid_sp.value()
        else:
            self.pid_group.setVisible(False)

    def _on_force_toggled(self, checked: bool) -> None:
        self.spin_force_val.setEnabled(checked)

    def _apply_changes(self) -> None:
        if self.selected_tag_id < 0:
            QMessageBox.warning(
                self, "No Selection", "Please select a mimic element first."
            )
            return

        # 1. Check if we need to write Manual Force Override
        if self.chk_force_active.isChecked():
            val = self.spin_force_val.value()
            # Force overrides are allowed for Operators by design (see set_role),
            # but a raw tag write to the live plant must be confirmed first.
            if (
                QMessageBox.question(
                    self,
                    "Confirm PLC write",
                    f"Force Tag {self.selected_tag_id} to {val}? "
                    "This overrides live control of the tag on the plant.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                == QMessageBox.StandardButton.Yes
            ):
                worker = HttpWorker(
                    "POST",
                    f"{self.backend_url}/api/tags/{self.selected_tag_id}",
                    json={"value": val},
                    timeout=1.0,
                    parent=self,
                )
                worker.success.connect(lambda data: self._on_force_success(val, data))
                worker.error.connect(self._on_force_error)
                start_http_request(
                    self,
                    "force_worker",
                    worker,
                    busy_button=self.btn_apply,
                    busy_text="Applying...",
                )

        # 2. Update config parameters (limits & PID)
        if self.user_role == "Admin" and self.routing_config:
            tag_id = self.selected_tag_id

            # Update safety limits (range end == "disabled" == None)
            if tag_id < len(self.routing_config.interlocks):
                low_value = self.spin_low_limit.value()
                high_value = self.spin_high_limit.value()
                self.routing_config.interlocks[tag_id].low_limit = (
                    None if low_value <= self.spin_low_limit.minimum() else low_value
                )
                self.routing_config.interlocks[tag_id].high_limit = (
                    None if high_value >= self.spin_high_limit.maximum() else high_value
                )

            # Update PID loop configs
            if self.pid_group.isVisible() and self.pid_loop_index >= 0:
                pid = self.routing_config.pids[self.pid_loop_index]
                pid.setpoint = self.spin_pid_sp.value()
                pid.kp = self.spin_pid_kp.value()
                pid.ki = self.spin_pid_ki.value()
                pid.kd = self.spin_pid_kd.value()

            # Send updated config back to PLC
            worker = HttpWorker(
                "POST",
                f"{self.backend_url}/api/routing",
                json=self.routing_config.dict(),
                timeout=2.0,
                parent=self,
            )
            worker.success.connect(self._on_routing_success)
            worker.error.connect(self._on_routing_error)
            start_http_request(
                self,
                "routing_worker",
                worker,
                busy_button=self.btn_apply,
                busy_text="Applying...",
            )

        elif self.user_role != "Admin" and (
            self.spin_low_limit.value() != self._baseline_low_limit
            or self.spin_pid_sp.value() != self._baseline_pid_setpoint
        ):
            # QDoubleSpinBox has no isModified() (that is a QLineEdit method);
            # calling it raised AttributeError inside this slot and, since
            # PyQt 5.5, an unhandled exception in a slot aborts the whole HMI
            # via qFatal. Compare against the last-loaded baseline instead
            # (issue #3320).
            QMessageBox.critical(
                self,
                "Access Denied",
                "Safety limits and PID gains are read-only for Operators.",
            )

    def _on_force_success(self, val, data):
        logger.info(f"Forced tag {self.selected_tag_id} to manual value: {val}")

    def _on_force_error(self, err_msg):
        QMessageBox.critical(
            self,
            "Override Failed",
            f"Failed to force tag value: {err_msg}",
        )

    def _on_routing_success(self, data):
        logger.info("DCS settings updated successfully in backend.")
        QMessageBox.information(
            self,
            "Saved",
            "Inspector parameters deployed successfully to PLC.",
        )
        self.configUpdated.emit()  # Ask main window to reload configuration

    def _on_routing_error(self, err_msg):
        QMessageBox.critical(
            self,
            "Save Failed",
            f"Failed to write routing updates: {err_msg}",
        )
