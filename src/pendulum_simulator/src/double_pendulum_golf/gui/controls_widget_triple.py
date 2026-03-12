"""
Control panel widget for triple pendulum inputs.

Refactored: parse_float / parse_coeffs now imported from controls_utils (DRY).
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .controls_utils import (
    STYLE_CHECK,
    STYLE_GROUP,
    parse_coeffs,
    parse_coeffs_lenient,
    parse_float,
    require_non_negative,
    require_positive,
)
from .controls_widget import LabeledInput
from .torque_preview_widget import TorquePreviewWidget


class ControlsWidgetTriple(QWidget):
    """Parameter input panel for the triple pendulum."""

    run_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    frame_changed = pyqtSignal(int)
    export_data_requested = pyqtSignal()
    export_video_requested = pyqtSignal()
    gravity_changed = pyqtSignal(bool)
    forces_changed = pyqtSignal(bool)

    PRESETS = {
        "Triple Swing": (
            120.0,
            -60.0,
            -30.0,
            0.0,
            0.0,
            0.0,
            "-25, 10",
            "0",
            "0",
            2.0,
            5.0,
            0.5,
            0.4,
            0.20,  # L1: Hub (sternum → shoulder)
            0.65,  # L2: Arm
            1.10,  # L3: Club
        ),
        "Free Triple Pendulum": (
            90.0,
            60.0,
            -45.0,
            0.0,
            0.0,
            0.0,
            "0",
            "0",
            "0",
            5.0,
            1.0,
            0.5,
            0.2,
            0.50,  # L1
            0.50,  # L2
            0.50,  # L3
        ),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.setMaximumWidth(360)
        self._is_playing = False
        self._build_ui()
        self._apply_preset("Triple Swing")

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        style_group = STYLE_GROUP  # use canonical shared style

        preset_group = QGroupBox("Preset")
        preset_group.setStyleSheet(style_group)
        pl = QVBoxLayout(preset_group)
        self.preset_combo = QComboBox()
        self.preset_combo.setStyleSheet(
            "background: #2a2a38; color: #e0e0f0; border: 1px solid #505068;"
            "border-radius: 3px; padding: 4px;"
        )
        for name in self.PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        pl.addWidget(self.preset_combo)
        main_layout.addWidget(preset_group)

        phys_group = QGroupBox("Physical Parameters")
        phys_group.setStyleSheet(style_group)
        pl2 = QVBoxLayout(phys_group)
        self.inp_m1 = LabeledInput("m1 (kg)", "5.0", "Mass of segment 1")
        self.inp_m2 = LabeledInput("m2 (kg)", "0.5", "Mass of segment 2")
        self.inp_m3 = LabeledInput("m3 (kg)", "0.4", "Mass of segment 3")
        self.inp_L1 = LabeledInput(
            "L1 (m) — Hub",
            "0.20",
            "Length of segment 1: Hub (sternum → shoulder)",
        )
        self.inp_L2 = LabeledInput(
            "L2 (m) — Arm",
            "0.65",
            "Length of segment 2: Arm",
        )
        self.inp_L3 = LabeledInput(
            "L3 (m) — Club",
            "1.10",
            "Length of segment 3: Club",
        )
        self.inp_scapula = LabeledInput(
            "Scapula °",
            "0",
            "Scapula protraction/retraction offset angle (#1152).\n"
            "0° = neutral, positive = protracted (forward).",
        )
        for w in [
            self.inp_m1,
            self.inp_m2,
            self.inp_m3,
            self.inp_L1,
            self.inp_L2,
            self.inp_L3,
            self.inp_scapula,
        ]:
            pl2.addWidget(w)
        main_layout.addWidget(phys_group)

        ic_group = QGroupBox("Initial Conditions")
        ic_group.setStyleSheet(style_group)
        pl3 = QVBoxLayout(ic_group)
        self.inp_theta1 = LabeledInput("theta1 (deg)", "120", "Segment 1 angle")
        self.inp_phi1 = LabeledInput("phi1 (deg)", "-60", "Segment 2 relative angle")
        self.inp_phi2 = LabeledInput("phi2 (deg)", "-30", "Segment 3 relative angle")
        self.inp_dtheta1 = LabeledInput("dtheta1 (rad/s)", "0", "Segment 1 velocity")
        self.inp_dphi1 = LabeledInput("dphi1 (rad/s)", "0", "Segment 2 velocity")
        self.inp_dphi2 = LabeledInput("dphi2 (rad/s)", "0", "Segment 3 velocity")
        for w in [
            self.inp_theta1,
            self.inp_phi1,
            self.inp_phi2,
            self.inp_dtheta1,
            self.inp_dphi1,
            self.inp_dphi2,
        ]:
            pl3.addWidget(w)
        main_layout.addWidget(ic_group)

        torque_group = QGroupBox("Torque Polynomials (c0, c1, c2, ...)")
        torque_group.setStyleSheet(style_group)
        pl4 = QVBoxLayout(torque_group)
        self.inp_tau_shoulder = LabeledInput(
            "Shoulder",
            "-25, 10",
            "Polynomial coefficients: tau(t) = c0 + c1*t + c2*t^2 + ...",
        )
        self.inp_tau_elbow = LabeledInput(
            "Elbow", "0", "Polynomial coefficients: tau(t) = c0 + c1*t + c2*t^2 + ..."
        )
        self.inp_tau_wrist = LabeledInput(
            "Wrist", "0", "Polynomial coefficients: tau(t) = c0 + c1*t + c2*t^2 + ..."
        )
        pl4.addWidget(self.inp_tau_shoulder)
        pl4.addWidget(self.inp_tau_elbow)
        pl4.addWidget(self.inp_tau_wrist)

        self.btn_funcgen = QPushButton("📈 Signal Toolkit…")
        self.btn_funcgen.setToolTip(
            "Design a waveform and import as torque coefficients"
        )
        self.btn_funcgen.setStyleSheet(
            "QPushButton{background:#282848;color:#b0b0e0;border:1px solid #404068;"
            "border-radius:4px;padding:4px 8px;font-size:10px;}"
            "QPushButton:hover{background:#32326a;}"
        )
        self.btn_funcgen.clicked.connect(self._open_function_generator)
        pl4.addWidget(self.btn_funcgen)

        main_layout.addWidget(torque_group)

        preview_group = QGroupBox("Torque Preview")
        preview_group.setStyleSheet(style_group)
        preview_layout = QVBoxLayout(preview_group)
        self.torque_preview = TorquePreviewWidget()
        preview_layout.addWidget(self.torque_preview)
        main_layout.addWidget(preview_group)

        time_group = QGroupBox("Simulation")
        time_group.setStyleSheet(style_group)
        pl5 = QVBoxLayout(time_group)
        self.inp_tend = LabeledInput("Duration (s)", "2.0", "Total simulation time")
        pl5.addWidget(self.inp_tend)
        main_layout.addWidget(time_group)

        # ── Dissipation parameters ────────────────────────────────
        diss_group = QGroupBox("Dissipation")
        diss_group.setStyleSheet(style_group)
        diss_layout = QVBoxLayout(diss_group)
        self.inp_b1 = LabeledInput("b1", "0.0", "Viscous damping shoulder (N·m·s)")
        self.inp_b2 = LabeledInput("b2", "0.0", "Viscous damping elbow (N·m·s)")
        self.inp_b3 = LabeledInput("b3", "0.0", "Viscous damping wrist (N·m·s)")
        self.inp_mu1 = LabeledInput("μ1", "0.0", "Coulomb friction shoulder (N·m)")
        self.inp_mu2 = LabeledInput("μ2", "0.0", "Coulomb friction elbow (N·m)")
        self.inp_mu3 = LabeledInput("μ3", "0.0", "Coulomb friction wrist (N·m)")
        for w in [
            self.inp_b1,
            self.inp_b2,
            self.inp_b3,
            self.inp_mu1,
            self.inp_mu2,
            self.inp_mu3,
        ]:
            diss_layout.addWidget(w)
        main_layout.addWidget(diss_group)

        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setStyleSheet(
            "QPushButton { background: #2d6b3f; color: white; border: none;"
            "border-radius: 5px; padding: 10px; font-size: 13px; font-weight: bold; }"
            "QPushButton:hover { background: #3a8a52; }"
            "QPushButton:pressed { background: #1f5030; }"
        )
        self.btn_run.clicked.connect(self.run_requested.emit)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setStyleSheet(
            "QPushButton { background: #5a3030; color: white; border: none;"
            "border-radius: 5px; padding: 10px; font-size: 13px; }"
            "QPushButton:hover { background: #7a4040; }"
        )
        self.btn_reset.clicked.connect(self.reset_requested.emit)

        btn_layout.addWidget(self.btn_run, stretch=2)
        btn_layout.addWidget(self.btn_reset, stretch=1)
        main_layout.addLayout(btn_layout)

        # Playback controls are in the toolstrip — create hidden compat widgets
        self.btn_play = QPushButton()
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 5.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.valueChanged.connect(lambda v: self.speed_changed.emit(v))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.frame_changed.emit)

        export_group = QGroupBox("Export")
        export_group.setStyleSheet(style_group)
        export_layout = QHBoxLayout(export_group)

        self.btn_export_data = QPushButton("Export Data")
        self.btn_export_data.setStyleSheet(
            "QPushButton { background: #303050; color: #c0c0e0; border: 1px solid #505068;"
            "border-radius: 4px; padding: 6px 10px; }"
            "QPushButton:hover { background: #3a3a60; }"
        )
        self.btn_export_data.clicked.connect(self.export_data_requested.emit)

        self.btn_export_video = QPushButton("Export Video")
        self.btn_export_video.setStyleSheet(
            "QPushButton { background: #303050; color: #c0c0e0; border: 1px solid #505068;"
            "border-radius: 4px; padding: 6px 10px; }"
            "QPushButton:hover { background: #3a3a60; }"
        )
        self.btn_export_video.clicked.connect(self.export_video_requested.emit)

        export_layout.addWidget(self.btn_export_data)
        export_layout.addWidget(self.btn_export_video)
        main_layout.addWidget(export_group)

        # ── Physics & Display toggles ─────────────────────────────
        vis_group = QGroupBox("Physics & Display")
        vis_group.setStyleSheet(
            "QGroupBox { color: #c0c0d8; border: 1px solid #404058;"
            "border-radius: 5px; margin-top: 8px; padding-top: 14px;"
            "font-weight: bold; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        vl = QVBoxLayout(vis_group)
        vl.setSpacing(4)

        self.chk_gravity = QCheckBox("🌍  Gravity enabled")
        self.chk_gravity.setChecked(True)
        self.chk_gravity.setStyleSheet(STYLE_CHECK)
        self.chk_gravity.toggled.connect(self.gravity_changed.emit)

        self.chk_forces = QCheckBox("↗  Show force vectors")
        self.chk_forces.setChecked(False)
        self.chk_forces.setStyleSheet(STYLE_CHECK)
        self.chk_forces.toggled.connect(self.forces_changed.emit)
        self.chk_forces.setVisible(False)  # #1143: force toggle lives in toolstrip

        vl.addWidget(self.chk_gravity)
        # chk_forces hidden — toolstrip is the single source of truth (#1143)
        main_layout.addWidget(vis_group)

        main_layout.addStretch()

        self.inp_tau_shoulder.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tau_elbow.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tau_wrist.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tend.edit.textChanged.connect(self._update_torque_preview)

    def _apply_preset(self, name: str) -> None:
        if name not in self.PRESETS:
            return
        (
            theta1,
            phi1,
            phi2,
            dth,
            dph1,
            dph2,
            tau_sh,
            tau_el,
            tau_wr,
            tend,
            m1,
            m2,
            m3,
            L1,
            L2,
            L3,
        ) = self.PRESETS[name]

        self.inp_theta1.set_value(str(theta1))
        self.inp_phi1.set_value(str(phi1))
        self.inp_phi2.set_value(str(phi2))
        self.inp_dtheta1.set_value(str(dth))
        self.inp_dphi1.set_value(str(dph1))
        self.inp_dphi2.set_value(str(dph2))
        self.inp_tau_shoulder.set_value(tau_sh)
        self.inp_tau_elbow.set_value(tau_el)
        self.inp_tau_wrist.set_value(tau_wr)
        self.inp_tend.set_value(str(tend))
        self.inp_m1.set_value(str(m1))
        self.inp_m2.set_value(str(m2))
        self.inp_m3.set_value(str(m3))
        self.inp_L1.set_value(str(L1))
        self.inp_L2.set_value(str(L2))
        self.inp_L3.set_value(str(L3))
        self._update_torque_preview()

    def get_params(self) -> dict:
        """Parse all inputs and return a simulation parameter dict.

        Raises
        ------
        ValueError if any field cannot be parsed or violates input contracts.
        """
        m1 = parse_float(self.inp_m1, "m1")
        m2 = parse_float(self.inp_m2, "m2")
        m3 = parse_float(self.inp_m3, "m3")
        L1 = parse_float(self.inp_L1, "L1")
        L2 = parse_float(self.inp_L2, "L2")
        L3 = parse_float(self.inp_L3, "L3")
        b1 = require_non_negative(parse_float(self.inp_b1, "b1"), "b1")
        b2 = require_non_negative(parse_float(self.inp_b2, "b2"), "b2")
        b3 = require_non_negative(parse_float(self.inp_b3, "b3"), "b3")
        mu1 = require_non_negative(parse_float(self.inp_mu1, "μ1"), "μ1")
        mu2 = require_non_negative(parse_float(self.inp_mu2, "μ2"), "μ2")
        mu3 = require_non_negative(parse_float(self.inp_mu3, "μ3"), "μ3")
        require_positive(m1, "m1")
        require_positive(m2, "m2")
        require_positive(m3, "m3")
        require_positive(L1, "L1")
        require_positive(L2, "L2")
        require_positive(L3, "L3")

        return {
            "m1": m1,
            "m2": m2,
            "m3": m3,
            "L1": L1,
            "L2": L2,
            "L3": L3,
            "theta1_rad": np.radians(parse_float(self.inp_theta1, "theta1")),
            "phi1_rad": np.radians(parse_float(self.inp_phi1, "phi1")),
            "phi2_rad": np.radians(parse_float(self.inp_phi2, "phi2")),
            "dtheta1": parse_float(self.inp_dtheta1, "dtheta1"),
            "dphi1": parse_float(self.inp_dphi1, "dphi1"),
            "dphi2": parse_float(self.inp_dphi2, "dphi2"),
            "shoulder_coeffs": parse_coeffs(self.inp_tau_shoulder, "Shoulder torque"),
            "elbow_coeffs": parse_coeffs(self.inp_tau_elbow, "Elbow torque"),
            "wrist_coeffs": parse_coeffs(self.inp_tau_wrist, "Wrist torque"),
            "t_end": parse_float(self.inp_tend, "Duration"),
            "gravity_on": self.chk_gravity.isChecked(),
            "b1": b1,
            "b2": b2,
            "b3": b3,
            "mu1": mu1,
            "mu2": mu2,
            "mu3": mu3,
            "scapula_deg": parse_float(self.inp_scapula, "Scapula"),
        }

    def _update_torque_preview(self) -> None:
        try:
            t_end = float(self.inp_tend.value)
        except ValueError:
            t_end = 2.0

        self.torque_preview.set_duration(t_end)
        self.torque_preview.set_profiles(
            [
                (
                    "Shoulder",
                    parse_coeffs_lenient(self.inp_tau_shoulder),
                    QColor(230, 120, 50),
                ),
                (
                    "Elbow",
                    parse_coeffs_lenient(self.inp_tau_elbow),
                    QColor(120, 200, 140),
                ),
                (
                    "Wrist",
                    parse_coeffs_lenient(self.inp_tau_wrist),
                    QColor(120, 180, 230),
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Function generator integration
    # ------------------------------------------------------------------

    def _open_function_generator(self) -> None:
        """Open Signal Toolkit as a dialog for torque design."""
        from .function_generator_dialog import FunctionGeneratorDialog

        dlg = FunctionGeneratorDialog(self, joint_names=["Shoulder", "Elbow", "Wrist"])
        dlg.torque_imported.connect(self._on_torque_imported)
        dlg.exec()

    def _on_torque_imported(self, joint: str, coeffs: list[float]) -> None:
        """Receive torque profile imported from Function Generator."""
        coeffs_str = ", ".join(f"{c:.4g}" for c in coeffs)
        joint_lower = joint.lower()
        if joint_lower == "shoulder":
            self.inp_tau_shoulder.set_value(coeffs_str)
        elif joint_lower == "elbow":
            self.inp_tau_elbow.set_value(coeffs_str)
        else:
            self.inp_tau_wrist.set_value(coeffs_str)
        self._update_torque_preview()

    def _on_play_toggled(self, checked: bool) -> None:
        self._is_playing = checked
        self.btn_play.setText("Pause" if checked else "Play")
        self.play_toggled.emit(checked)

    def set_slider_range(self, max_val: int) -> None:
        """Pre: max_val >= 0"""
        assert max_val >= 0, f"Slider max must be non-negative, got {max_val}"
        self.slider.setRange(0, max_val)

    def set_slider_value(self, val: int) -> None:
        """Pre: 0 <= val <= slider.maximum()"""
        assert (
            0 <= val <= self.slider.maximum()
        ), f"Slider value {val} out of range [0, {self.slider.maximum()}]"
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)

    def stop_playback(self) -> None:
        self.btn_play.setChecked(False)
