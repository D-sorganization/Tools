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
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .controls_utils import (
    STYLE_GROUP,
    parse_coeffs,
    parse_coeffs_lenient,
    parse_float,
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
            0.6,
            0.6,
            0.3,
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
            1.0,
            1.0,
            0.5,
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
        self.inp_L1 = LabeledInput("L1 (m)", "0.6", "Length of segment 1")
        self.inp_L2 = LabeledInput("L2 (m)", "0.6", "Length of segment 2")
        self.inp_L3 = LabeledInput("L3 (m)", "0.6", "Length of segment 3")
        for w in [
            self.inp_m1,
            self.inp_m2,
            self.inp_m3,
            self.inp_L1,
            self.inp_L2,
            self.inp_L3,
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

        play_group = QGroupBox("Playback")
        play_group.setStyleSheet(style_group)
        pl6 = QVBoxLayout(play_group)

        ctrl_row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setStyleSheet(
            "QPushButton { background: #303050; color: #c0c0e0; border: 1px solid #505068;"
            "border-radius: 4px; padding: 6px 12px; }"
            "QPushButton:checked { background: #504030; color: #f0d080; }"
        )
        self.btn_play.toggled.connect(self._on_play_toggled)
        ctrl_row.addWidget(self.btn_play)

        ctrl_row.addWidget(QLabel("Speed:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 5.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setStyleSheet(
            "background: #2a2a38; color: #e0e0f0; border: 1px solid #505068;"
        )
        self.speed_spin.valueChanged.connect(lambda v: self.speed_changed.emit(v))
        ctrl_row.addWidget(self.speed_spin)
        pl6.addLayout(ctrl_row)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal { background: #303048; height: 6px;"
            "border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #7070a0; width: 14px;"
            "margin: -5px 0; border-radius: 7px; }"
        )
        self.slider.valueChanged.connect(self.frame_changed.emit)
        pl6.addWidget(self.slider)
        main_layout.addWidget(play_group)

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

        _STYLE_CHECK = (
            "QCheckBox { color: #c0c0d8; font-size: 11px; spacing: 5px; }"
            "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #505068;"
            "border-radius: 3px; background: #2a2a38; }"
            "QCheckBox::indicator:checked { background: #5060a0; border-color: #7080c0; }"
        )
        self.chk_gravity = QCheckBox("🌍  Gravity enabled")
        self.chk_gravity.setChecked(True)
        self.chk_gravity.setStyleSheet(_STYLE_CHECK)
        self.chk_gravity.toggled.connect(self.gravity_changed.emit)

        self.chk_forces = QCheckBox("↗  Show force vectors")
        self.chk_forces.setChecked(False)
        self.chk_forces.setStyleSheet(_STYLE_CHECK)
        self.chk_forces.toggled.connect(self.forces_changed.emit)

        vl.addWidget(self.chk_gravity)
        vl.addWidget(self.chk_forces)
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
        ValueError  if any field cannot be parsed.
        AssertionError  if any mass or length is non-positive.
        """
        m1 = parse_float(self.inp_m1, "m1")
        m2 = parse_float(self.inp_m2, "m2")
        m3 = parse_float(self.inp_m3, "m3")
        L1 = parse_float(self.inp_L1, "L1")
        L2 = parse_float(self.inp_L2, "L2")
        L3 = parse_float(self.inp_L3, "L3")
        assert m1 > 0 and m2 > 0 and m3 > 0, "All masses must be positive"
        assert L1 > 0 and L2 > 0 and L3 > 0, "All lengths must be positive"

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

    def _on_play_toggled(self, checked: bool) -> None:
        self._is_playing = checked
        self.btn_play.setText("Pause" if checked else "Play")
        self.play_toggled.emit(checked)

    def set_slider_range(self, max_val: int) -> None:
        self.slider.setRange(0, max_val)

    def set_slider_value(self, val: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)

    def stop_playback(self) -> None:
        self.btn_play.setChecked(False)
