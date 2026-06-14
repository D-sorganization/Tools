# mypy: ignore-errors
# ruff: noqa: E501
"""MPC setup and request handling for the P1AM desktop control tab."""

import logging

from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .plot_compat import pg
from .workers import HttpWorker, start_http_request

logger = logging.getLogger("p1am_control.desktop.control")


class ControlTabMpcMixin:
    """MPC groundwork tab behavior for ``ControlTab``."""

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

        right_layout = QVBoxLayout()
        self.mpc_pv_plot = pg.PlotWidget(widget)
        self.mpc_pv_plot.setBackground(self.palette().color(QPalette.ColorRole.Base))
        self.mpc_pv_plot.showGrid(x=True, y=True, alpha=0.3)
        self.mpc_pv_plot.setLabel("left", "Process Value (PV)")
        self.mpc_pv_plot.setLabel("bottom", "Time (seconds)")
        self.mpc_pv_plot.addLegend()
        self.mpc_pv_plot.setTitle("PV Tracking Comparison")
        self.curve_pid_pv = self.mpc_pv_plot.plot(
            pen=pg.mkPen(color="r", width=2), name="PID PV"
        )
        self.curve_mpc_pv = self.mpc_pv_plot.plot(
            pen=pg.mkPen(color=(0, 100, 0), width=2), name="MPC PV"
        )
        right_layout.addWidget(self.mpc_pv_plot)

        self.mpc_cv_plot = pg.PlotWidget(widget)
        self.mpc_cv_plot.setBackground(self.palette().color(QPalette.ColorRole.Base))
        self.mpc_cv_plot.showGrid(x=True, y=True, alpha=0.3)
        self.mpc_cv_plot.setLabel("left", "Control Value (CV)")
        self.mpc_cv_plot.setLabel("bottom", "Time (seconds)")
        self.mpc_cv_plot.addLegend()
        self.mpc_cv_plot.setTitle("CV Output Effort Comparison")
        self.curve_pid_cv = self.mpc_cv_plot.plot(
            pen=pg.mkPen(color="r", width=2), name="PID CV"
        )
        self.curve_mpc_cv = self.mpc_cv_plot.plot(
            pen=pg.mkPen(color=(0, 100, 0), width=2), name="MPC CV"
        )
        right_layout.addWidget(self.mpc_cv_plot)

        layout.addLayout(right_layout, 2)

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

        worker = HttpWorker(
            "POST",
            f"{self.backend_url}/api/mpc/simulate",
            json=payload,
            timeout=2.0,
            parent=self,
        )
        worker.success.connect(self._on_simulate_mpc_success)
        worker.error.connect(self._on_connection_error)
        start_http_request(
            self,
            "mpc_worker",
            worker,
            busy_button=self.btn_simulate_mpc,
            busy_text="Simulating...",
        )

    def _on_simulate_mpc_success(self, data):
        times = data["time"]

        self.curve_pid_pv.setData(times, data["pid"]["pv"])
        self.curve_mpc_pv.setData(times, data["mpc"]["pv"])
        self.curve_pid_cv.setData(times, data["pid"]["cv"])
        self.curve_mpc_cv.setData(times, data["mpc"]["cv"])

        logger.info("MPC vs PID simulation completed successfully.")
