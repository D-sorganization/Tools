# mypy: ignore-errors
# ruff: noqa: E501
import logging
import os

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .workers import HttpWorker

logger = logging.getLogger("p1am_control.desktop.control")


class ControlTab(QWidget):
    """PID Loop control and MPC Groundwork tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("control_tab")

        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

        # Operator/Admin role gate. Live-loop tuning and gain writes require
        # Admin; updated via :meth:`set_role` from the main window. Defaults to
        # the least-privileged role so a tab created before login cannot write.
        self.user_role = "Operator"

        # Current active PID configurations fetched from PLC/Backend
        self.routing_config = None

        # Real-time PV/SP history for the active tracking plot
        self.tracking_time = []
        self.tracking_pv = []
        self.tracking_sp = []
        self.max_tracking_points = 200  # ~20 seconds of data at 10Hz

        self._init_ui()

        # Active tuning state
        self.tuning_active = False

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Tab Widget for PID vs MPC
        self.sub_tabs = QTabWidget(self)

        # Tab 1: PID Tuning
        pid_tab = QWidget()
        self._setup_pid_tab(pid_tab)
        self.sub_tabs.addTab(pid_tab, "PID Loop Tuning")

        # Tab 2: MPC Groundwork & Comparison
        mpc_tab = QWidget()
        self._setup_mpc_tab(mpc_tab)
        self.sub_tabs.addTab(mpc_tab, "MPC Groundwork")

        layout.addWidget(self.sub_tabs)

    def _setup_pid_tab(self, widget: QWidget) -> None:
        layout = QHBoxLayout(widget)

        # Left side: Loop Selector and Real-Time Tracking Plot
        left_layout = QVBoxLayout()

        loop_sel_layout = QHBoxLayout()
        loop_sel_layout.addWidget(QLabel("Select Loop:", widget))
        self.loop_combo = QComboBox(widget)
        self.loop_combo.addItems(
            [
                "Loop 0: Drying Hopper Level (Tag 1 -> 2)",
                "Loop 1: Pyrolysis Temp (Tag 3 -> 4)",
                "Loop 2: Combustion Temp (Tag 5 -> 6)",
                "Loop 3: Reduction Temp (Tag 7 -> 8)",
            ]
        )
        self.loop_combo.currentIndexChanged.connect(self._on_loop_changed)
        loop_sel_layout.addWidget(self.loop_combo)
        loop_sel_layout.addStretch()
        left_layout.addLayout(loop_sel_layout)

        # Plot Widget for PV/SP tracking
        self.tracking_plot = pg.PlotWidget(widget)
        self.tracking_plot.setBackground(self.palette().color(QPalette.ColorRole.Base))
        self.tracking_plot.showGrid(x=True, y=True, alpha=0.3)
        self.tracking_plot.setLabel("left", "Process Value")
        self.tracking_plot.setLabel("bottom", "Time (samples)")
        self.tracking_plot.addLegend()

        self.curve_pv = self.tracking_plot.plot(
            pen=pg.mkPen(
                color=self.palette().color(QPalette.ColorRole.Highlight), width=2
            ),
            name="PV",
        )
        self.curve_sp = self.tracking_plot.plot(
            pen=pg.mkPen(
                color=self.palette().color(QPalette.ColorRole.WindowText),
                width=2,
                style=Qt.PenStyle.DashLine,
            ),
            name="Setpoint",
        )

        left_layout.addWidget(self.tracking_plot)
        layout.addLayout(left_layout, 2)

        # Right side: Tuning Panel and Cohen-Coon recommendations
        right_layout = QVBoxLayout()

        # Tuning Panel Box
        tuning_group = QGroupBox("Loop Tuning Controls", widget)
        tuning_grid = QGridLayout(tuning_group)

        self.btn_start_tuning = QPushButton("Start Tuning Mode", widget)
        self.btn_start_tuning.clicked.connect(self._start_tuning)
        tuning_grid.addWidget(self.btn_start_tuning, 0, 0, 1, 2)

        tuning_grid.addWidget(QLabel("Step CV Value (%):", widget), 1, 0)
        self.spin_step_val = QDoubleSpinBox(widget)
        self.spin_step_val.setRange(0.0, 100.0)
        self.spin_step_val.setValue(50.0)
        tuning_grid.addWidget(self.spin_step_val, 1, 1)

        self.btn_apply_step = QPushButton("Apply Step Change", widget)
        self.btn_apply_step.clicked.connect(self._apply_step)
        self.btn_apply_step.setEnabled(False)
        tuning_grid.addWidget(self.btn_apply_step, 2, 0, 1, 2)

        self.btn_stop_tuning = QPushButton("Stop Tuning", widget)
        self.btn_stop_tuning.clicked.connect(self._stop_tuning)
        self.btn_stop_tuning.setEnabled(False)
        tuning_grid.addWidget(self.btn_stop_tuning, 3, 0, 1, 2)

        right_layout.addWidget(tuning_group)

        # Cohen-Coon Panel Box
        param_group = QGroupBox("Identified Parameters & Cohen-Coon", widget)
        param_grid = QGridLayout(param_group)

        param_grid.addWidget(QLabel("Gain (Kp):", widget), 0, 0)
        self.lbl_ident_kp = QLabel("--", widget)
        param_grid.addWidget(self.lbl_ident_kp, 0, 1)

        param_grid.addWidget(QLabel("Tau (τ):", widget), 1, 0)
        self.lbl_ident_tau = QLabel("--", widget)
        param_grid.addWidget(self.lbl_ident_tau, 1, 1)

        param_grid.addWidget(QLabel("Delay (θ):", widget), 2, 0)
        self.lbl_ident_theta = QLabel("--", widget)
        param_grid.addWidget(self.lbl_ident_theta, 2, 1)

        # Recommended gains
        param_grid.addWidget(QLabel("Recommended Gains:", widget), 3, 0, 1, 2)

        param_grid.addWidget(QLabel("Proportional (Kp):", widget), 4, 0)
        self.lbl_recom_kp = QLabel("--", widget)
        param_grid.addWidget(self.lbl_recom_kp, 4, 1)

        param_grid.addWidget(QLabel("Integral (Ki):", widget), 5, 0)
        self.lbl_recom_ki = QLabel("--", widget)
        param_grid.addWidget(self.lbl_recom_ki, 5, 1)

        param_grid.addWidget(QLabel("Derivative (Kd):", widget), 6, 0)
        self.lbl_recom_kd = QLabel("--", widget)
        param_grid.addWidget(self.lbl_recom_kd, 6, 1)

        self.btn_apply_gains = QPushButton("Apply Recommended Gains", widget)
        self.btn_apply_gains.clicked.connect(self._apply_recommended_gains)
        self.btn_apply_gains.setEnabled(False)
        param_grid.addWidget(self.btn_apply_gains, 7, 0, 1, 2)

        right_layout.addWidget(param_group)
        right_layout.addStretch()

        layout.addLayout(right_layout, 1)

    def _setup_mpc_tab(self, widget: QWidget) -> None:
        layout = QHBoxLayout(widget)

        # Left side: MPC configuration inputs
        left_layout = QVBoxLayout()
        mpc_group = QGroupBox("MPC Parameters", widget)
        grid = QGridLayout(mpc_group)

        grid.addWidget(QLabel("Prediction Horizon (P):", widget), 0, 0)
        self.spin_p_horiz = QSpinBox(widget)
        self.spin_p_horiz.setRange(2, 30)
        self.spin_p_horiz.setValue(10)
        grid.addWidget(self.spin_p_horiz, 0, 1)

        grid.addWidget(QLabel("Control Horizon (M):", widget), 1, 0)
        self.spin_c_horiz = QSpinBox(widget)
        self.spin_c_horiz.setRange(1, 10)
        self.spin_c_horiz.setValue(3)
        grid.addWidget(self.spin_c_horiz, 1, 1)

        grid.addWidget(QLabel("Setpoint:", widget), 2, 0)
        self.spin_mpc_sp = QDoubleSpinBox(widget)
        self.spin_mpc_sp.setRange(0.0, 100.0)
        self.spin_mpc_sp.setValue(50.0)
        grid.addWidget(self.spin_mpc_sp, 2, 1)

        grid.addWidget(QLabel("Penalty Factor (rho):", widget), 3, 0)
        self.spin_mpc_rho = QDoubleSpinBox(widget)
        self.spin_mpc_rho.setRange(0.0, 10.0)
        self.spin_mpc_rho.setSingleStep(0.05)
        self.spin_mpc_rho.setValue(0.1)
        grid.addWidget(self.spin_mpc_rho, 3, 1)

        grid.addWidget(QLabel("Plant Gain (Kp):", widget), 4, 0)
        self.spin_plant_gain = QDoubleSpinBox(widget)
        self.spin_plant_gain.setRange(0.1, 5.0)
        self.spin_plant_gain.setValue(1.2)
        grid.addWidget(self.spin_plant_gain, 4, 1)

        grid.addWidget(QLabel("Plant Tau (τ):", widget), 5, 0)
        self.spin_plant_tau = QDoubleSpinBox(widget)
        self.spin_plant_tau.setRange(0.5, 20.0)
        self.spin_plant_tau.setValue(5.0)
        grid.addWidget(self.spin_plant_tau, 5, 1)

        grid.addWidget(QLabel("Plant Delay (θ):", widget), 6, 0)
        self.spin_plant_delay = QDoubleSpinBox(widget)
        self.spin_plant_delay.setRange(0.0, 5.0)
        self.spin_plant_delay.setValue(1.0)
        grid.addWidget(self.spin_plant_delay, 6, 1)

        self.btn_simulate_mpc = QPushButton("Simulate PID vs MPC", widget)
        self.btn_simulate_mpc.clicked.connect(self._simulate_mpc)
        grid.addWidget(self.btn_simulate_mpc, 7, 0, 1, 2)

        left_layout.addWidget(mpc_group)
        left_layout.addStretch()
        layout.addLayout(left_layout, 1)

        # Right side: plots showing step responses of PID vs MPC
        right_layout = QVBoxLayout()

        # PV comparison plot
        self.mpc_pv_plot = pg.PlotWidget(widget)
        self.mpc_pv_plot.setBackground(self.palette().color(QPalette.ColorRole.Base))
        self.mpc_pv_plot.showGrid(x=True, y=True, alpha=0.3)
        self.mpc_pv_plot.setLabel("left", "Process Value (PV)")
        self.mpc_pv_plot.setLabel("bottom", "Time (seconds)")
        self.mpc_pv_plot.addLegend()
        self.mpc_pv_plot.setTitle("PV Tracking Comparison")

        self.curve_pid_pv = self.mpc_pv_plot.plot(
            pen=pg.mkPen(color=Qt.GlobalColor.red, width=2), name="PID PV"
        )
        self.curve_mpc_pv = self.mpc_pv_plot.plot(
            pen=pg.mkPen(color=Qt.GlobalColor.darkGreen, width=2), name="MPC PV"
        )
        right_layout.addWidget(self.mpc_pv_plot)

        # CV comparison plot
        self.mpc_cv_plot = pg.PlotWidget(widget)
        self.mpc_cv_plot.setBackground(self.palette().color(QPalette.ColorRole.Base))
        self.mpc_cv_plot.showGrid(x=True, y=True, alpha=0.3)
        self.mpc_cv_plot.setLabel("left", "Control Value (CV)")
        self.mpc_cv_plot.setLabel("bottom", "Time (seconds)")
        self.mpc_cv_plot.addLegend()
        self.mpc_cv_plot.setTitle("CV Output Effort Comparison")

        self.curve_pid_cv = self.mpc_cv_plot.plot(
            pen=pg.mkPen(color=Qt.GlobalColor.red, width=2), name="PID CV"
        )
        self.curve_mpc_cv = self.mpc_cv_plot.plot(
            pen=pg.mkPen(color=Qt.GlobalColor.darkGreen, width=2), name="MPC CV"
        )
        right_layout.addWidget(self.mpc_cv_plot)

        layout.addLayout(right_layout, 2)

    def _on_loop_changed(self, idx: int) -> None:
        self.tracking_time.clear()
        self.tracking_pv.clear()
        self.tracking_sp.clear()
        logger.info(f"Active PID loop changed to: Loop {idx}")

    def update_telemetry(self, tags: list[float]) -> None:
        """Called by main thread to feed live setpoints and PV data."""
        if not self.routing_config or self.tuning_active:
            return

        idx = self.loop_combo.currentIndex()
        if idx >= len(self.routing_config.pids):
            return

        pid_cfg = self.routing_config.pids[idx]
        pv_tag = pid_cfg.pv_tag_id
        sp = pid_cfg.setpoint

        if pv_tag < len(tags):
            pv = tags[pv_tag]

            self.tracking_pv.append(pv)
            self.tracking_sp.append(sp)

            if len(self.tracking_pv) > self.max_tracking_points:
                self.tracking_pv.pop(0)
                self.tracking_sp.pop(0)

            self.curve_pv.setData(self.tracking_pv)
            self.curve_sp.setData(self.tracking_sp)

    def set_routing_config(self, config) -> None:
        """Saves current routing/PID configs fetched from backend."""
        self.routing_config = config

    def set_role(self, role: str) -> None:
        """Enable/disable live-loop tuning controls based on user role.

        Tuning a live loop (open-loop step tests) and writing PID gains to the
        PLC are Admin-only actions, mirroring ``RoutingTab.set_role``. The
        buttons are disabled for Operators for UX feedback; the slots also
        re-check the role at runtime because tuning flows re-enable buttons.
        """
        self.user_role = role
        is_admin = role == "Admin"

        self.btn_start_tuning.setEnabled(is_admin)
        # Step/Stop/Apply-gains are only meaningful mid-tuning; gate the entry
        # point (Start Tuning) and let the existing tuning state machine manage
        # the rest. Apply-gains is additionally disabled until a tune completes.
        if not is_admin:
            self.btn_apply_step.setEnabled(False)
            self.btn_stop_tuning.setEnabled(False)
            self.btn_apply_gains.setEnabled(False)

        tip = (
            "Tune live PID loops (Admin)"
            if is_admin
            else "Requires Admin privileges to tune live PID loops"
        )
        self.btn_start_tuning.setToolTip(tip)

    def _require_admin(self, action: str) -> bool:
        """Return ``True`` if the current role may perform *action*.

        Shows an Access Denied dialog and returns ``False`` for non-Admins.
        """
        if self.user_role != "Admin":
            QMessageBox.critical(
                self,
                "Access Denied",
                f"Only Admin users can {action}.",
            )
            return False
        return True

    def _on_connection_error(self, err_msg: str) -> None:
        QMessageBox.critical(
            self, "Connection Error", f"Could not reach backend: {err_msg}"
        )

    def _start_tuning(self) -> None:
        if not self._require_admin("start live-loop tuning"):
            return
        idx = self.loop_combo.currentIndex()
        if (
            QMessageBox.question(
                self,
                "Confirm PLC write",
                f"Start open-loop tuning on live Loop {idx}? "
                "This takes the loop out of automatic control.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.start_worker = HttpWorker(
            "POST", f"{self.backend_url}/api/pid/{idx}/tuning/start", timeout=1.0
        )
        self.start_worker.success.connect(
            lambda data: self._on_start_tuning_success(idx, data)
        )
        self.start_worker.error.connect(self._on_connection_error)
        self.start_worker.start()

    def _on_start_tuning_success(self, idx, data):
        self.tuning_active = True
        self.btn_start_tuning.setEnabled(False)
        self.btn_apply_step.setEnabled(True)
        self.btn_stop_tuning.setEnabled(True)
        self.btn_apply_gains.setEnabled(False)
        logger.info(f"Tuning mode started for PID loop {idx}")

    def _apply_step(self) -> None:
        if not self._require_admin("apply tuning steps to a live loop"):
            return
        idx = self.loop_combo.currentIndex()
        step_val = self.spin_step_val.value()
        if (
            QMessageBox.question(
                self,
                "Confirm PLC write",
                f"Apply a {step_val}% control-output step to live Loop {idx}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.step_worker = HttpWorker(
            "POST",
            f"{self.backend_url}/api/pid/{idx}/tuning/step",
            json={"step_value": step_val},
            timeout=1.0,
        )
        self.step_worker.success.connect(
            lambda data: self._on_apply_step_success(idx, step_val, data)
        )
        self.step_worker.error.connect(self._on_connection_error)
        self.step_worker.start()

    def _on_apply_step_success(self, idx, step_val, data):
        logger.info(f"Tuning step applied to loop {idx}: {step_val}%")
        QMessageBox.information(
            self, "Step Applied", f"Tuning step CV set to {step_val}%"
        )

    def _stop_tuning(self) -> None:
        idx = self.loop_combo.currentIndex()
        self.stop_worker = HttpWorker(
            "POST", f"{self.backend_url}/api/pid/{idx}/tuning/stop", timeout=1.5
        )
        self.stop_worker.success.connect(
            lambda data: self._on_stop_tuning_success(idx, data)
        )
        self.stop_worker.error.connect(self._on_connection_error)
        self.stop_worker.start()

    def _on_stop_tuning_success(self, idx, data):
        self.tuning_active = False
        self.btn_start_tuning.setEnabled(True)
        self.btn_apply_step.setEnabled(False)
        self.btn_stop_tuning.setEnabled(False)

        if data.get("status") == "success":
            params = data["parameters"]
            recom = data["recommended_pid"]

            self.lbl_ident_kp.setText(str(params["kp"]))
            self.lbl_ident_tau.setText(str(params["tau"]))
            self.lbl_ident_theta.setText(str(params["theta"]))

            self.lbl_recom_kp.setText(str(recom["kp"]))
            self.lbl_recom_ki.setText(str(recom["ki"]))
            self.lbl_recom_kd.setText(str(recom["kd"]))

            self.btn_apply_gains.setEnabled(True)
            logger.info(f"Tuning stopped for loop {idx}. Parameters: {params}")
        else:
            QMessageBox.warning(
                self,
                "Tuning Stopped",
                data.get("message", "No parameters identified."),
            )

    def _apply_recommended_gains(self) -> None:
        if not self.routing_config:
            return

        if not self._require_admin("apply PID gains to a live loop"):
            return

        idx = self.loop_combo.currentIndex()
        try:
            kp = float(self.lbl_recom_kp.text())
            ki = float(self.lbl_recom_ki.text())
            kd = float(self.lbl_recom_kd.text())

            if (
                QMessageBox.question(
                    self,
                    "Confirm PLC write",
                    f"Apply PID gains kp={kp}, ki={ki}, kd={kd} to live Loop "
                    f"{idx}? This retunes a loop currently controlling the plant.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                != QMessageBox.StandardButton.Yes
            ):
                return

            # Mutate local copy of routing config
            self.routing_config.pids[idx].kp = kp
            self.routing_config.pids[idx].ki = ki
            self.routing_config.pids[idx].kd = kd

            self.gains_worker = HttpWorker(
                "POST",
                f"{self.backend_url}/api/routing",
                json=self.routing_config.dict(),
                timeout=2.0,
            )
            self.gains_worker.success.connect(
                lambda data: self._on_apply_gains_success(idx, data)
            )
            self.gains_worker.error.connect(self._on_connection_error)
            self.gains_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply gains: {e}")

    def _on_apply_gains_success(self, idx, data):
        QMessageBox.information(
            self,
            "Gains Applied",
            f"Successfully applied gains to PLC Loop {idx}.",
        )
        self.btn_apply_gains.setEnabled(False)
        logger.info(f"Applied recommended PID gains to Loop {idx}")

    def _simulate_mpc(self) -> None:
        payload = {
            "prediction_horizon": self.spin_p_horiz.value(),
            "control_horizon": self.spin_c_horiz.value(),
            "setpoint": self.spin_mpc_sp.value(),
            "rho": self.spin_mpc_rho.value(),
            "process_gain": self.spin_plant_gain.value(),
            "process_tau": self.spin_plant_tau.value(),
            "process_delay": self.spin_plant_delay.value(),
        }

        self.mpc_worker = HttpWorker(
            "POST", f"{self.backend_url}/api/mpc/simulate", json=payload, timeout=2.0
        )
        self.mpc_worker.success.connect(self._on_simulate_mpc_success)
        self.mpc_worker.error.connect(self._on_connection_error)
        self.mpc_worker.start()

    def _on_simulate_mpc_success(self, data):
        times = data["time"]

        # Plot responses
        self.curve_pid_pv.setData(times, data["pid"]["pv"])
        self.curve_mpc_pv.setData(times, data["mpc"]["pv"])

        self.curve_pid_cv.setData(times, data["pid"]["cv"])
        self.curve_mpc_cv.setData(times, data["mpc"]["cv"])

        logger.info("MPC vs PID simulation completed successfully.")
