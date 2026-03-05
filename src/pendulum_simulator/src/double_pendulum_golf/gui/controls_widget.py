"""
Control panel widget with parameter inputs, initial conditions,
torque polynomial editors, and playback controls.

Provides golf-swing presets alongside fully customizable inputs.
"""

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QGroupBox,
    QComboBox,
    QDoubleSpinBox,
)

from .torque_preview_widget import TorquePreviewWidget


class LabeledInput(QWidget):
    """A label + line-edit pair used throughout the control panel.

    DRY: This avoids repeating the label-edit pattern dozens of times.
    """

    def __init__(self, label: str, default: str, tooltip: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setFixedWidth(90)
        lbl.setStyleSheet("color: #b0b0c0; font-size: 11px;")
        layout.addWidget(lbl)

        self.edit = QLineEdit(default)
        self.edit.setStyleSheet(
            "background: #2a2a38; color: #e0e0f0; border: 1px solid #505068;"
            "border-radius: 3px; padding: 3px 6px; font-family: monospace;"
        )
        if tooltip:
            self.edit.setToolTip(tooltip)
        layout.addWidget(self.edit)

    @property
    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, text: str) -> None:
        self.edit.setText(text)


class ControlsWidget(QWidget):
    """Parameter input panel with presets and playback controls.

    Signals
    -------
    run_requested : emitted when user clicks "Run Simulation"
    reset_requested : emitted when user clicks "Reset"
    play_toggled(bool) : emitted when play/pause toggled (True = play)
    speed_changed(float) : emitted when playback speed changes
    frame_changed(int) : emitted when user drags the timeline slider
    """

    run_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    frame_changed = pyqtSignal(int)
    export_data_requested = pyqtSignal()
    export_video_requested = pyqtSignal()

    # Presets: (name, theta1_deg, phi_deg, dtheta1, dphi,
    #           shoulder_coeffs, wrist_coeffs, t_end,
    #           m1, m2, L1, L2)
    PRESETS = {
        "Golf Swing (passive wrist)": (
            120.0,
            -90.0,
            0.0,
            0.0,
            "-25, 10",
            "0",
            2.0,
            5.0,
            0.5,
            0.6,
            1.0,
        ),
        "Golf Swing (active wrist)": (
            120.0,
            -90.0,
            0.0,
            0.0,
            "-25, 10",
            "-2, 3",
            2.0,
            5.0,
            0.5,
            0.6,
            1.0,
        ),
        "Free Double Pendulum": (
            90.0,
            90.0,
            0.0,
            0.0,
            "0",
            "0",
            5.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        "Straight Drop": (
            0.1,
            0.1,
            0.0,
            0.0,
            "0",
            "0",
            3.0,
            1.0,
            1.0,
            0.8,
            0.8,
        ),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.setMaximumWidth(360)
        self._is_playing = False
        self._build_ui()
        self._apply_preset("Golf Swing (passive wrist)")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        style_group = (
            "QGroupBox { color: #c0c0d8; border: 1px solid #404058;"
            "border-radius: 5px; margin-top: 8px; padding-top: 14px;"
            "font-weight: bold; }"
            "QGroupBox::title { subcontrol-origin: margin;"
            "left: 10px; padding: 0 4px; }"
        )

        # Preset selector
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

        # Physical parameters
        phys_group = QGroupBox("Physical Parameters")
        phys_group.setStyleSheet(style_group)
        pl2 = QVBoxLayout(phys_group)
        self.inp_m1 = LabeledInput("m1 (kg)", "5.0", "Mass of arm segment")
        self.inp_m2 = LabeledInput("m2 (kg)", "0.5", "Mass of club segment")
        self.inp_L1 = LabeledInput("L1 (m)", "0.6", "Length of arm segment")
        self.inp_L2 = LabeledInput("L2 (m)", "1.0", "Length of club segment")
        for w in [self.inp_m1, self.inp_m2, self.inp_L1, self.inp_L2]:
            pl2.addWidget(w)
        main_layout.addWidget(phys_group)

        # Initial conditions
        ic_group = QGroupBox("Initial Conditions")
        ic_group.setStyleSheet(style_group)
        pl3 = QVBoxLayout(ic_group)
        self.inp_theta1 = LabeledInput(
            "\u03b81 (deg)", "120", "Arm angle from vertical"
        )
        self.inp_phi = LabeledInput("\u03c6 (deg)", "-90", "Club angle relative to arm")
        self.inp_dtheta1 = LabeledInput("d\u03b81 (rad/s)", "0", "Arm angular velocity")
        self.inp_dphi = LabeledInput("d\u03c6 (rad/s)", "0", "Club angular velocity")
        for w in [self.inp_theta1, self.inp_phi, self.inp_dtheta1, self.inp_dphi]:
            pl3.addWidget(w)
        main_layout.addWidget(ic_group)

        # Torque polynomials
        torque_group = QGroupBox("Torque Polynomials (c0, c1, c2, ...)")
        torque_group.setStyleSheet(style_group)
        pl4 = QVBoxLayout(torque_group)
        self.inp_tau_shoulder = LabeledInput(
            "Shoulder",
            "-25, 10",
            "Polynomial coefficients: \u03c4(t) = c0 + c1*t + c2*t\u00b2 + ...",
        )
        self.inp_tau_wrist = LabeledInput(
            "Wrist",
            "0",
            "Polynomial coefficients: \u03c4(t) = c0 + c1*t + c2*t\u00b2 + ...",
        )
        pl4.addWidget(self.inp_tau_shoulder)
        pl4.addWidget(self.inp_tau_wrist)
        main_layout.addWidget(torque_group)

        preview_group = QGroupBox("Torque Preview")
        preview_group.setStyleSheet(style_group)
        preview_layout = QVBoxLayout(preview_group)
        self.torque_preview = TorquePreviewWidget()
        preview_layout.addWidget(self.torque_preview)
        main_layout.addWidget(preview_group)

        # Simulation duration
        time_group = QGroupBox("Simulation")
        time_group.setStyleSheet(style_group)
        pl5 = QVBoxLayout(time_group)
        self.inp_tend = LabeledInput("Duration (s)", "2.0", "Total simulation time")
        pl5.addWidget(self.inp_tend)
        main_layout.addWidget(time_group)

        # Dissipation parameters
        diss_group = QGroupBox("Dissipation (optional)")
        diss_group.setStyleSheet(style_group)
        pl_diss = QVBoxLayout(diss_group)
        self.inp_b1 = LabeledInput(
            "b1 (N·m·s)",
            "0.0",
            "Viscous damping at joint 1 (shoulder) — proportional to angular velocity",
        )
        self.inp_b2 = LabeledInput(
            "b2 (N·m·s)",
            "0.0",
            "Viscous damping at joint 2 (wrist) — proportional to angular velocity",
        )
        self.inp_mu1 = LabeledInput(
            "\u03bc1 (N·m)",
            "0.0",
            "Coulomb friction at joint 1 (shoulder) — constant magnitude opposing motion",
        )
        self.inp_mu2 = LabeledInput(
            "\u03bc2 (N·m)",
            "0.0",
            "Coulomb friction at joint 2 (wrist) — constant magnitude opposing motion",
        )
        for w in [self.inp_b1, self.inp_b2, self.inp_mu1, self.inp_mu2]:
            pl_diss.addWidget(w)
        main_layout.addWidget(diss_group)

        # Run / Reset buttons
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Simulation")
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

        # Playback controls
        play_group = QGroupBox("Playback")
        play_group.setStyleSheet(style_group)
        pl6 = QVBoxLayout(play_group)

        # Play/Pause + Speed
        ctrl_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
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

        # Timeline slider
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

        # Export controls
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

        main_layout.addStretch()

        self.inp_tau_shoulder.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tau_wrist.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tend.edit.textChanged.connect(self._update_torque_preview)

    # ------------------------------------------------------------------
    # Preset application
    # ------------------------------------------------------------------

    def _apply_preset(self, name: str) -> None:
        """Load a named preset into all input fields."""
        if name not in self.PRESETS:
            return
        theta1, phi, dth, dph, tau_sh, tau_wr, tend, m1, m2, L1, L2 = self.PRESETS[name]

        self.inp_theta1.set_value(str(theta1))
        self.inp_phi.set_value(str(phi))
        self.inp_dtheta1.set_value(str(dth))
        self.inp_dphi.set_value(str(dph))
        self.inp_tau_shoulder.set_value(tau_sh)
        self.inp_tau_wrist.set_value(tau_wr)
        self.inp_tend.set_value(str(tend))
        self.inp_m1.set_value(str(m1))
        self.inp_m2.set_value(str(m2))
        self.inp_L1.set_value(str(L1))
        self.inp_L2.set_value(str(L2))
        self._update_torque_preview()

    # ------------------------------------------------------------------
    # Value parsing (with validation)
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Parse all input fields and return as a dict.

        Returns
        -------
        dict with keys: m1, m2, L1, L2, theta1_rad, phi_rad,
            dtheta1, dphi, shoulder_coeffs, wrist_coeffs, t_end

        Raises
        ------
        ValueError if any input cannot be parsed.
        """

        def parse_float(widget: LabeledInput, name: str) -> float:
            try:
                return float(widget.value)
            except ValueError:
                raise ValueError(f"Cannot parse '{name}': '{widget.value}'")

        def parse_coeffs(widget: LabeledInput, name: str) -> list:
            try:
                parts = widget.value.split(",")
                return [float(p.strip()) for p in parts if p.strip()]
            except ValueError:
                raise ValueError(
                    f"Cannot parse '{name}' coefficients: '{widget.value}'"
                )

        return {
            "m1": parse_float(self.inp_m1, "m1"),
            "m2": parse_float(self.inp_m2, "m2"),
            "L1": parse_float(self.inp_L1, "L1"),
            "L2": parse_float(self.inp_L2, "L2"),
            "theta1_rad": np.radians(parse_float(self.inp_theta1, "θ1")),
            "phi_rad": np.radians(parse_float(self.inp_phi, "φ")),
            "dtheta1": parse_float(self.inp_dtheta1, "dθ1"),
            "dphi": parse_float(self.inp_dphi, "dφ"),
            "shoulder_coeffs": parse_coeffs(self.inp_tau_shoulder, "Shoulder torque"),
            "wrist_coeffs": parse_coeffs(self.inp_tau_wrist, "Wrist torque"),
            "t_end": parse_float(self.inp_tend, "Duration"),
            "b1": parse_float(self.inp_b1, "b1"),
            "b2": parse_float(self.inp_b2, "b2"),
            "mu1": parse_float(self.inp_mu1, "μ1"),
            "mu2": parse_float(self.inp_mu2, "μ2"),
        }

    def _update_torque_preview(self) -> None:
        try:
            t_end = float(self.inp_tend.value)
        except ValueError:
            t_end = 2.0

        def parse_coeffs(widget: LabeledInput) -> list:
            parts = widget.value.split(",")
            coeffs = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                try:
                    coeffs.append(float(part))
                except ValueError:
                    return []
            return coeffs or [0.0]

        shoulder = parse_coeffs(self.inp_tau_shoulder)
        wrist = parse_coeffs(self.inp_tau_wrist)

        self.torque_preview.set_duration(t_end)
        self.torque_preview.set_profiles(
            [
                ("Shoulder", shoulder, QColor(230, 120, 50)),
                ("Wrist", wrist, QColor(120, 180, 230)),
            ]
        )

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _on_play_toggled(self, checked: bool) -> None:
        self._is_playing = checked
        self.btn_play.setText("⏸ Pause" if checked else "▶ Play")
        self.play_toggled.emit(checked)

    def set_slider_range(self, max_val: int) -> None:
        """Update the timeline slider range after simulation runs."""
        self.slider.setRange(0, max_val)

    def set_slider_value(self, val: int) -> None:
        """Update slider position without emitting signal (for animation)."""
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)

    def stop_playback(self) -> None:
        """Force stop playback state."""
        self.btn_play.setChecked(False)
