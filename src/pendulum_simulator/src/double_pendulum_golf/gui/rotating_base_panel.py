"""Qualified rotating-base study controls and reviewer diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from shared.python.swing_sim.rotating_base import (
    MATCHING_RULES,
    MODEL_TIER,
    TORSO_PROFILES,
    REGISTERED_TORSO_RATES_RAD_S,
    RotatingBaseRunRequest,
    RotatingBaseRunResult,
    load_embedded_qualified_study,
    registered_run_json,
    run_registered_case,
)
from .no_scroll_widgets import NoScrollComboBox

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


_PROFILE_LABELS = {
    "accelerate": "Accelerate (+55 N·m)",
    "constant_rate": "Zero Torso Command (0 N·m)",
    "decelerate": "Decelerate (−55 N·m)",
}
_MATCHING_LABELS = {
    "relative_club_rate": "Relative Club Rate",
    "absolute_club_rate": "Absolute Club Rate",
}
_METRICS = (
    ("impact_speed_m_s", "Delivery Speed", "m/s"),
    ("contact_work_on_club_j", "Contact Work on Club", "J"),
    ("braking_grip_work_j", "Braking Grip Work", "J"),
    ("force_couple_work_j", "Force-Couple Work", "J"),
    ("negative_along_path_impulse_ns", "Negative Along-Path Impulse", "N·s"),
    ("bilateral_wrist_work_j", "Bilateral Wrist Work", "J"),
    ("total_control_work_j", "Total Control Work", "J"),
    ("distal_energy_gain_j", "Distal Energy Gain", "J"),
    ("peak_grip_force_n", "Peak Grip Force", "N"),
    ("maximum_constraint_residual_m", "Position Closure", "m"),
    ("maximum_velocity_constraint_residual_m_s", "Velocity Closure", "m/s"),
    ("maximum_contact_power_identity_residual_w", "Power Identity Residual", "W"),
    ("work_energy_closure_j", "Work–Energy Closure", "J"),
)


class _RotatingBaseWorker(QObject):
    """Execute one registered case off the GUI thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, request: RotatingBaseRunRequest) -> None:
        super().__init__()
        if not isinstance(request, RotatingBaseRunRequest):
            raise TypeError("request must be a RotatingBaseRunRequest")
        self._request = request

    def run(self) -> None:
        """Emit one qualified result or a bounded diagnostic."""
        try:
            self.finished.emit(run_registered_case(self._request))
        except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
            logger.exception("Registered rotating-base execution failed")
            self.failed.emit(str(exc))


class RotatingBasePanel(QScrollArea):
    """Expose the qualified 18-case model without anatomical relabeling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: RotatingBaseRunResult | None = None
        self._thread: QThread | None = None
        self._worker: _RotatingBaseWorker | None = None
        self.setWidgetResizable(True)
        self.setAccessibleName("Qualified Rotating-Base Study")
        content = QWidget(self)
        self.setWidget(content)
        layout = QVBoxLayout(content)
        layout.addWidget(self._build_boundaries())
        layout.addWidget(self._build_design_controls())
        layout.addWidget(self._build_metrics())
        layout.addWidget(self._build_plot())
        layout.addStretch()

    def _build_boundaries(self) -> QGroupBox:
        group = QGroupBox("Scientific Scope")
        layout = QVBoxLayout(group)
        self._boundary = QLabel(
            f"Qualified tier: {MODEL_TIER}. The torso coordinate is a "
            "nonanatomical finite-inertia model coordinate. This is model evidence "
            "with no governed human validation and no coaching recommendation."
        )
        self._boundary.setWordWrap(True)
        layout.addWidget(self._boundary)
        authority = load_embedded_qualified_study().study
        killswitch = authority.same_state_killswitch
        channel_text = "; ".join(
            f"{name.replace('_', ' ')} Δ speed "
            f"{channel.delivery_speed_difference_m_s:.3g} m/s, Δ contact work "
            f"{channel.post_branch_contact_work_difference_j:.3g} J"
            for name, channel in killswitch.channels
        )
        self._killswitches = QLabel(
            f"Registered exact same-state killswitches at {killswitch.branch_time_s:g} s: "
            f"{channel_text}. All {authority.attempted_case_count} rows are retained "
            f"({authority.valid_case_count} valid)."
        )
        self._killswitches.setWordWrap(True)
        layout.addWidget(self._killswitches)
        return group

    def _build_design_controls(self) -> QGroupBox:
        group = QGroupBox("Registered Case")
        layout = QGridLayout(group)
        self._profile_combo = NoScrollComboBox()
        self._matching_combo = NoScrollComboBox()
        self._rate_combo = NoScrollComboBox()
        for profile in TORSO_PROFILES:
            self._profile_combo.addItem(_PROFILE_LABELS[profile], profile)
        for rule in MATCHING_RULES:
            self._matching_combo.addItem(_MATCHING_LABELS[rule], rule)
        for rate in REGISTERED_TORSO_RATES_RAD_S:
            self._rate_combo.addItem(f"{rate:g} rad/s", rate)
        self._profile_combo.setAccessibleName("Torso Program")
        self._matching_combo.setAccessibleName("Club-Rate Matching Rule")
        self._rate_combo.setAccessibleName("Initial Torso Rate")
        profile_label = QLabel("Torso Program:")
        profile_label.setBuddy(self._profile_combo)
        matching_label = QLabel("Matching Rule:")
        matching_label.setBuddy(self._matching_combo)
        rate_label = QLabel("Initial Torso Rate:")
        rate_label.setBuddy(self._rate_combo)
        layout.addWidget(profile_label, 0, 0)
        layout.addWidget(self._profile_combo, 0, 1)
        layout.addWidget(matching_label, 0, 2)
        layout.addWidget(self._matching_combo, 0, 3)
        layout.addWidget(rate_label, 1, 0)
        layout.addWidget(self._rate_combo, 1, 1)
        actions = QHBoxLayout()
        self._run_button = QPushButton("Run Registered Case")
        self._run_button.setAccessibleName("Run Registered Rotating-Base Case")
        self._run_button.clicked.connect(self._start_run)
        self._export_button = QPushButton("Export Governed JSON")
        self._export_button.setAccessibleName("Export Governed Rotating-Base JSON")
        self._export_button.setEnabled(False)
        self._export_button.clicked.connect(self._export_result)
        actions.addWidget(self._run_button)
        actions.addWidget(self._export_button)
        layout.addLayout(actions, 1, 2, 1, 2)
        self._status = QLabel("Select one registered case and run the qualified model.")
        self._status.setAccessibleName("Rotating-Base Execution Status")
        self._status.setWordWrap(True)
        layout.addWidget(self._status, 2, 0, 1, 4)
        return group

    def _build_metrics(self) -> QGroupBox:
        group = QGroupBox("Transfer and Numerical Diagnostics")
        layout = QGridLayout(group)
        self._metric_labels: dict[str, QLabel] = {}
        for index, (key, title, unit) in enumerate(_METRICS):
            row = index // 2
            column = 2 * (index % 2)
            layout.addWidget(QLabel(f"{title}:"), row, column)
            value = QLabel(f"-- {unit}")
            value.setAccessibleName(title)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(value, row, column + 1)
            self._metric_labels[key] = value
        return group

    def _build_plot(self) -> QWidget:
        if not _HAS_MPL:
            self._figure = None
            return QLabel("Install matplotlib to view registered-case traces.")
        self._figure = Figure(figsize=(8, 9), dpi=100)
        self._axis_power = self._figure.add_subplot(511)
        self._axis_couple = self._figure.add_subplot(512, sharex=self._axis_power)
        self._axis_rates = self._figure.add_subplot(513, sharex=self._axis_power)
        self._axis_energy = self._figure.add_subplot(514, sharex=self._axis_power)
        self._axis_force = self._figure.add_subplot(515, sharex=self._axis_power)
        self._canvas = FigureCanvasQTAgg(self._figure)
        return cast(QWidget, self._canvas)

    def _request(self) -> RotatingBaseRunRequest:
        return RotatingBaseRunRequest(
            torso_profile=str(self._profile_combo.currentData()),
            matching_rule=str(self._matching_combo.currentData()),
            initial_torso_rate_rad_s=float(self._rate_combo.currentData()),
        )

    def _start_run(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._result = None
        self._export_button.setEnabled(False)
        self._run_button.setEnabled(False)
        self._status.setText("Running the full-resolution registered case…")
        thread = QThread(self)
        worker = _RotatingBaseWorker(self._request())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._accept_result)
        worker.finished.connect(thread.quit)
        worker.failed.connect(self._accept_failure)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _accept_result(self, result: object) -> None:
        if not isinstance(result, RotatingBaseRunResult):
            raise TypeError("result must be a RotatingBaseRunResult")
        self._result = result
        self._profile_combo.setCurrentIndex(
            self._profile_combo.findData(result.request.torso_profile)
        )
        self._matching_combo.setCurrentIndex(
            self._matching_combo.findData(result.request.matching_rule)
        )
        self._rate_combo.setCurrentIndex(
            self._rate_combo.findData(result.request.initial_torso_rate_rad_s)
        )
        self._run_button.setEnabled(True)
        self._export_button.setEnabled(True)
        case = result.case
        if case.valid:
            state = "Valid registered row"
        else:
            reasons = ", ".join(case.exclusion_reasons)
            state = f"Invalid/adverse retained — {reasons}"
        self._status.setText(
            f"Case {case.case_index}: {state}. Source {result.source_revision[:12]}."
        )
        for key, _title, unit in _METRICS:
            value = getattr(case.metrics, key)
            self._metric_labels[key].setText(f"{value:.6g} {unit}")
        self._draw_result(result)

    def _accept_failure(self, message: str) -> None:
        self._run_button.setEnabled(True)
        self._status.setText(f"Registered execution failed: {message}")

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _draw_result(self, result: RotatingBaseRunResult) -> None:
        if self._figure is None:
            return
        trace = result.trace
        for axis in (
            self._axis_power,
            self._axis_couple,
            self._axis_rates,
            self._axis_energy,
            self._axis_force,
        ):
            axis.clear()
        self._axis_power.plot(trace.time_s, trace.contact_power_on_club_w)
        self._axis_power.axhline(0.0, color="black", linewidth=0.7)
        self._axis_power.set_ylabel("Contact Power (W)")
        self._axis_couple.plot(trace.time_s, trace.force_generated_couple_nm)
        self._axis_couple.axhline(0.0, color="black", linewidth=0.7)
        self._axis_couple.set_ylabel("Force Couple (N·m)")
        self._axis_rates.plot(trace.time_s, trace.torso_rate_rad_s, label="Torso")
        self._axis_rates.plot(trace.time_s, trace.club_rate_rad_s, label="Club")
        self._axis_rates.set_ylabel("Rate (rad/s)")
        self._axis_rates.legend(loc="best", fontsize=8)
        self._axis_energy.plot(
            trace.time_s,
            trace.distal_segment_kinetic_energy_j,
        )
        self._axis_energy.set_ylabel("Distal Energy (J)")
        force = np.linalg.norm(trace.force_on_club_n, axis=2)
        self._axis_force.plot(trace.time_s, force[:, 0], label="Lead Grip")
        self._axis_force.plot(trace.time_s, force[:, 1], label="Trail Grip")
        self._axis_force.set_xlabel("Time (s)")
        self._axis_force.set_ylabel("Grip Force (N)")
        self._axis_force.legend(loc="best", fontsize=8)
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def _export_result(self) -> None:
        if self._result is None:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Governed Rotating-Base Run",
            f"rotating_base_case_{self._result.case.case_index}.json",
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            Path(path).write_text(registered_run_json(self._result), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        self._status.setText(f"Exported governed case evidence to {path}")


__all__ = ["RotatingBasePanel"]
