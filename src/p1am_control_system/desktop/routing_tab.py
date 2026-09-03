# mypy: ignore-errors
# ruff: noqa: E501
import logging
import math
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .workers import HttpWorker, start_http_request

logger = logging.getLogger("p1am_control.desktop.routing")


def _parse_limit_cell(text: str) -> float | None:
    """Parse one interlock-limit table cell.

    A blank cell means the side is disabled (``None``). Anything else must be
    a finite float: NaN/Inf would be refused by the backend (#3974), so reject
    it here with the same ``ValueError`` the caller already handles.

    Raises:
        ValueError: If ``text`` is neither blank nor a finite float.
    """
    stripped = text.strip()
    if not stripped:
        return None
    value = float(stripped)
    if not math.isfinite(value):
        raise ValueError(f"limit must be finite, got {stripped!r}")
    return value


class RoutingTab(QWidget):
    """DCS Routing Matrix and Interlocks Configuration Tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("routing_tab")

        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        self.routing_config = None
        self.user_role = "Operator"

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Top section: Input/Output Routing Matrix
        top_layout = QHBoxLayout()

        # Input Routing Box (6 physical inputs AI 0-5)
        self.input_group = QGroupBox("Physical Input Routing (AI 0 - AI 5)", self)
        input_grid = QGridLayout(self.input_group)
        self.input_spins = []
        for i in range(6):
            input_grid.addWidget(QLabel(f"AI {i} mapped to Tag:", self), i, 0)
            spin = QSpinBox(self)
            spin.setRange(0, 31)
            self.input_spins.append(spin)
            input_grid.addWidget(spin, i, 1)
        top_layout.addWidget(self.input_group)

        # Output Routing Box (2 physical outputs AO 0-1)
        self.output_group = QGroupBox("Physical Output Routing (AO 0 - AO 1)", self)
        output_grid = QGridLayout(self.output_group)
        self.output_spins = []
        for i in range(2):
            output_grid.addWidget(QLabel(f"AO {i} mapped to Tag:", self), i, 0)
            spin = QSpinBox(self)
            spin.setRange(0, 31)
            self.output_spins.append(spin)
            output_grid.addWidget(spin, i, 1)
        top_layout.addWidget(self.output_group)

        layout.addLayout(top_layout)

        # Bottom section: Tag Interlock Configuration Grid
        self.interlock_group = QGroupBox("Tag Interlocks (Safety Limits)", self)
        interlock_layout = QVBoxLayout(self.interlock_group)

        # Scrollable table for all 32 tags
        self.table = QTableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Tag ID", "Low Limit", "High Limit"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        interlock_layout.addWidget(self.table)

        layout.addWidget(self.interlock_group)

        # Deploy Config Button
        self.btn_deploy = QPushButton("Deploy Configuration to PLC", self)
        self.btn_deploy.setStyleSheet("font-weight: bold; height: 35px;")
        self.btn_deploy.clicked.connect(self._deploy_config)
        layout.addWidget(self.btn_deploy)

        # Apply default read-only restrictions
        self.set_role(self.user_role)

    def set_role(self, role: str) -> None:
        """Enables/disables routing edits based on role. Limits can only be configured by Admin."""
        self.user_role = role
        is_admin = role == "Admin"

        # Enable or disable inputs
        for spin in self.input_spins:
            spin.setEnabled(is_admin)
        for spin in self.output_spins:
            spin.setEnabled(is_admin)

        # Enable or disable table editing
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.AllEditTriggers
            if is_admin
            else QTableWidget.EditTrigger.NoEditTriggers
        )

        self.btn_deploy.setEnabled(is_admin)
        if is_admin:
            self.btn_deploy.setToolTip("Deploy changed configuration parameters to PLC")
        else:
            self.btn_deploy.setToolTip(
                "Requires Admin privileges to modify PLC configuration"
            )

    def set_routing_config(self, config) -> None:
        """Loads routing and interlock data from model config into view controls."""
        self.routing_config = config

        # Load input routing
        for i, val in enumerate(config.input_routing):
            if i < len(self.input_spins):
                self.input_spins[i].setValue(val)

        # Load output routing
        for i, val in enumerate(config.output_routing):
            if i < len(self.output_spins):
                self.output_spins[i].setValue(val)

        # Load interlocks
        self.table.setRowCount(32)
        for i in range(32):
            # Tag ID
            id_item = QTableWidgetItem(f"Tag {i}")
            id_item.setFlags(
                id_item.flags() ^ Qt.ItemFlag.ItemIsEditable
            )  # Tag ID is always read-only
            self.table.setItem(i, 0, id_item)

            # Low/High Limits. ``None`` (side disabled) is shown as a blank
            # cell; a blank cell deploys back as ``None``.
            if i < len(config.interlocks):
                low_val = config.interlocks[i].low_limit
                high_val = config.interlocks[i].high_limit
            else:
                low_val = None
                high_val = None

            low_item = QTableWidgetItem("" if low_val is None else f"{low_val:.2f}")
            high_item = QTableWidgetItem("" if high_val is None else f"{high_val:.2f}")

            self.table.setItem(i, 1, low_item)
            self.table.setItem(i, 2, high_item)

    def _deploy_config(self) -> None:
        if not self.routing_config:
            QMessageBox.warning(self, "Failed", "No routing configuration loaded.")
            return

        if self.user_role != "Admin":
            QMessageBox.critical(
                self,
                "Access Denied",
                "Only Admin users can deploy configurations to the PLC.",
            )
            return

        if (
            QMessageBox.question(
                self,
                "Confirm PLC write",
                "Deploy the full DCS routing and interlock matrix to the PLC? "
                "This persists to PLC NVRAM and takes effect on the live plant.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        try:
            # 1. Collect inputs
            inputs = [spin.value() for spin in self.input_spins]
            outputs = [spin.value() for spin in self.output_spins]

            # 2. Collect interlocks from table
            interlocks = []
            for i in range(32):
                try:
                    low_val = _parse_limit_cell(self.table.item(i, 1).text())
                    high_val = _parse_limit_cell(self.table.item(i, 2).text())
                except ValueError:
                    QMessageBox.critical(
                        self,
                        "Invalid Inputs",
                        f"Interlock limit values for Tag {i} must be finite "
                        "floats or blank (disabled).",
                    )
                    return
                interlocks.append({"low_limit": low_val, "high_limit": high_val})

            # Update our routing config model structure
            self.routing_config.input_routing = inputs
            self.routing_config.output_routing = outputs
            for idx, item in enumerate(interlocks):
                if idx < len(self.routing_config.interlocks):
                    self.routing_config.interlocks[idx].low_limit = item["low_limit"]
                    self.routing_config.interlocks[idx].high_limit = item["high_limit"]

            # 3. Post back to backend
            worker = HttpWorker(
                "POST",
                f"{self.backend_url}/api/routing",
                json=self.routing_config.dict(),
                timeout=3.0,
                parent=self,
            )
            worker.success.connect(self._on_deploy_success)
            worker.error.connect(self._on_deploy_error)
            start_http_request(
                self,
                "deploy_worker",
                worker,
                busy_button=self.btn_deploy,
                busy_text="Deploying...",
                restore_button=lambda was: was and self.user_role == "Admin",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare deploy: {e}")

    def _on_deploy_success(self, data):
        logger.info("Deployed new DCS routing configuration to PLC.")
        QMessageBox.information(
            self,
            "Success",
            "Configuration successfully deployed and saved to PLC NVRAM.",
        )

    def _on_deploy_error(self, err_msg):
        QMessageBox.critical(
            self, "Deployment Failed", f"DCS routing write failed: {err_msg}"
        )
