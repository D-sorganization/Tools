"""
Control panel widget with parameter inputs, initial conditions,
torque polynomial editors, and playback controls.

Provides golf-swing presets alongside fully customizable inputs.
New in UI/UX upgrade:
- Gravity on/off toggle
- Show/hide force vectors toggle
- Gravity & force signals passed to simulation panel
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
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .torque_preview_widget import TorquePreviewWidget

_STYLE_BTN_BASE = (
    "QPushButton {{ background: {bg}; color: {fg}; border: {border};"
    "border-radius: 4px; padding: {pad}; font-size: {fs}; {extra} }}"
    "QPushButton:hover {{ background: {hover}; }}"
    "QPushButton:pressed {{ background: {press}; }}"
)

_STYLE_GROUP = (
    "QGroupBox { color: #c8c8e0; border: 1px solid #404060;"
    "border-radius: 6px; margin-top: 10px; padding-top: 16px;"
    "font-weight: bold; font-size: 11px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
)

_STYLE_SLIDER = (
    "QSlider::groove:horizontal { background: #252540; height: 6px; border-radius: 3px; }"
    "QSlider::sub-page:horizontal { background: #5060a0; border-radius: 3px; }"
    "QSlider::handle:horizontal { background: #8090d0; width: 14px;"
    "margin: -5px 0; border-radius: 7px; border: 1px solid #6070b0; }"
    "QSlider::handle:horizontal:hover { background: #a0b0f0; }"
)

_STYLE_CHECK = (
    "QCheckBox { color: #c0c0d8; font-size: 11px; spacing: 5px; }"
    "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #505068;"
    "border-radius: 3px; background: #2a2a38; }"
    "QCheckBox::indicator:checked { background: #5060a0; border-color: #7080c0; }"
)


class LabeledInput(QWidget):
    """A label + line-edit pair used throughout the control panel.

    DRY: This avoids repeating the label-edit pattern dozens of times.
    """

    def __init__(
        self, label: str, default: str, tooltip: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setFixedWidth(92)
        lbl.setStyleSheet("color: #a8a8c0; font-size: 11px;")
        layout.addWidget(lbl)

        self.edit = QLineEdit(default)
        self.edit.setStyleSheet(
            "background: #22223a; color: #e4e4f4; border: 1px solid #484868;"
            "border-radius: 3px; padding: 3px 6px; font-family: monospace; font-size: 11px;"
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
    run_requested        : user clicks Run Simulation
    reset_requested      : user clicks Reset
    play_toggled(bool)   : play/pause toggled
    speed_changed(float) : playback speed changed
    frame_changed(int)   : user drags the timeline slider
    gravity_changed(bool): gravity on/off toggled (True = on)
    forces_changed(bool) : show/hide force vectors toggled (True = show)
    """

    run_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    frame_changed = pyqtSignal(int)
    export_data_requested = pyqtSignal()
    export_video_requested = pyqtSignal()
    gravity_changed = pyqtSignal(bool)
    forces_changed = pyqtSignal(bool)

    # Presets: (theta1_deg, phi_deg, dtheta1, dphi,
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)
        self._is_playing = False
        self._build_ui()
        self._apply_preset("Golf Swing (passive wrist)")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(5)

        # ── Preset selector ──────────────────────────────────────────
        preset_group = QGroupBox("Preset")
        preset_group.setStyleSheet(_STYLE_GROUP)
        pl = QVBoxLayout(preset_group)
        pl.setSpacing(4)
        self.preset_combo = QComboBox()
        self.preset_combo.setStyleSheet(
            "background: #22223a; color: #e4e4f4; border: 1px solid #484868;"
            "border-radius: 3px; padding: 4px; font-size: 11px;"
        )
        for name in self.PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        pl.addWidget(self.preset_combo)
        main_layout.addWidget(preset_group)

        # ── Physical parameters ───────────────────────────────────────
        phys_group = QGroupBox("Physical Parameters")
        phys_group.setStyleSheet(_STYLE_GROUP)
        pl2 = QVBoxLayout(phys_group)
        pl2.setSpacing(3)
        self.inp_m1 = LabeledInput("m1 (kg)", "5.0", "Mass of arm segment")
        self.inp_m2 = LabeledInput("m2 (kg)", "0.5", "Mass of club segment")
        self.inp_L1 = LabeledInput("L1 (m)", "0.6", "Length of arm segment")
        self.inp_L2 = LabeledInput("L2 (m)", "1.0", "Length of club segment")
        for w in [self.inp_m1, self.inp_m2, self.inp_L1, self.inp_L2]:
            pl2.addWidget(w)
        main_layout.addWidget(phys_group)

        # ── Initial conditions ────────────────────────────────────────
        ic_group = QGroupBox("Initial Conditions")
        ic_group.setStyleSheet(_STYLE_GROUP)
        pl3 = QVBoxLayout(ic_group)
        pl3.setSpacing(3)
        self.inp_theta1 = LabeledInput("θ1 (deg)", "120", "Arm angle from vertical")
        self.inp_phi = LabeledInput("φ (deg)", "-90", "Club angle relative to arm")
        self.inp_dtheta1 = LabeledInput("dθ1 (rad/s)", "0", "Arm angular velocity")
        self.inp_dphi = LabeledInput("dφ (rad/s)", "0", "Club angular velocity")
        for w in [self.inp_theta1, self.inp_phi, self.inp_dtheta1, self.inp_dphi]:
            pl3.addWidget(w)
        main_layout.addWidget(ic_group)

        # ── Torque polynomials ────────────────────────────────────────
        torque_group = QGroupBox("Torque Polynomials (c0, c1, c2, …)")
        torque_group.setStyleSheet(_STYLE_GROUP)
        pl4 = QVBoxLayout(torque_group)
        pl4.setSpacing(3)
        self.inp_tau_shoulder = LabeledInput(
            "Shoulder", "-25, 10", "τ(t) = c0 + c1·t + c2·t² + …"
        )
        self.inp_tau_wrist = LabeledInput("Wrist", "0", "τ(t) = c0 + c1·t + c2·t² + …")
        pl4.addWidget(self.inp_tau_shoulder)
        pl4.addWidget(self.inp_tau_wrist)
        main_layout.addWidget(torque_group)

        preview_group = QGroupBox("Torque Preview")
        preview_group.setStyleSheet(_STYLE_GROUP)
        preview_layout = QVBoxLayout(preview_group)
        self.torque_preview = TorquePreviewWidget()
        preview_layout.addWidget(self.torque_preview)
        main_layout.addWidget(preview_group)

        # ── Simulation duration ───────────────────────────────────────
        time_group = QGroupBox("Simulation")
        time_group.setStyleSheet(_STYLE_GROUP)
        pl5 = QVBoxLayout(time_group)
        pl5.setSpacing(3)
        self.inp_tend = LabeledInput("Duration (s)", "2.0", "Total simulation time")
        pl5.addWidget(self.inp_tend)
        main_layout.addWidget(time_group)

        # ── Dissipation ───────────────────────────────────────────────
        diss_group = QGroupBox("Dissipation (optional)")
        diss_group.setStyleSheet(_STYLE_GROUP)
        pl_diss = QVBoxLayout(diss_group)
        pl_diss.setSpacing(3)
        self.inp_b1 = LabeledInput("b1 (N·m·s)", "0.0", "Viscous damping at shoulder")
        self.inp_b2 = LabeledInput("b2 (N·m·s)", "0.0", "Viscous damping at wrist")
        self.inp_mu1 = LabeledInput("μ1 (N·m)", "0.0", "Coulomb friction at shoulder")
        self.inp_mu2 = LabeledInput("μ2 (N·m)", "0.0", "Coulomb friction at wrist")
        for w in [self.inp_b1, self.inp_b2, self.inp_mu1, self.inp_mu2]:
            pl_diss.addWidget(w)
        main_layout.addWidget(diss_group)

        # ── Physics toggles ───────────────────────────────────────────
        vis_group = QGroupBox("Physics & Display")
        vis_group.setStyleSheet(_STYLE_GROUP)
        vl = QVBoxLayout(vis_group)
        vl.setSpacing(4)

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

        # ── Run / Reset ───────────────────────────────────────────────
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

        # ── Playback ──────────────────────────────────────────────────
        play_group = QGroupBox("Playback")
        play_group.setStyleSheet(_STYLE_GROUP)
        pl6 = QVBoxLayout(play_group)
        pl6.setSpacing(4)

        ctrl_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setStyleSheet(
            "QPushButton { background: #2a2a48; color: #c0c0e8; border: 1px solid #484870;"
            "border-radius: 4px; padding: 6px 12px; font-size: 12px; }"
            "QPushButton:checked { background: #50402a; color: #f0d080; border-color: #807050; }"
            "QPushButton:hover { background: #383860; }"
        )
        self.btn_play.toggled.connect(self._on_play_toggled)
        ctrl_row.addWidget(self.btn_play)

        spd_lbl = QLabel("Speed:")
        spd_lbl.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        ctrl_row.addWidget(spd_lbl)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 10.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setFixedWidth(60)
        self.speed_spin.setStyleSheet(
            "background: #22223a; color: #e0e0f0; border: 1px solid #484868;"
            "border-radius: 3px; font-size: 11px;"
        )
        self.speed_spin.valueChanged.connect(lambda v: self.speed_changed.emit(v))
        ctrl_row.addWidget(self.speed_spin)
        pl6.addLayout(ctrl_row)

        # Timeline slider with step label
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setStyleSheet(_STYLE_SLIDER)
        self.slider.valueChanged.connect(self.frame_changed.emit)
        pl6.addWidget(self.slider)

        self.lbl_frame = QLabel("Frame: 0 / 0")
        self.lbl_frame.setStyleSheet("color: #7878a0; font-size: 10px;")
        self.lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl6.addWidget(self.lbl_frame)
        main_layout.addWidget(play_group)

        # ── Export ────────────────────────────────────────────────────
        export_group = QGroupBox("Export")
        export_group.setStyleSheet(_STYLE_GROUP)
        export_layout = QHBoxLayout(export_group)
        export_layout.setSpacing(4)

        for label, signal in [
            ("📊 Data", self.export_data_requested),
            ("🎬 Video", self.export_video_requested),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(
                "QPushButton { background: #282840; color: #b0b0d0; border: 1px solid #484868;"
                "border-radius: 4px; padding: 5px 8px; font-size: 11px; }"
                "QPushButton:hover { background: #343458; }"
            )
            btn.clicked.connect(signal.emit)
            export_layout.addWidget(btn)
        self.btn_export_data = export_group.findChildren(QPushButton)[0]
        self.btn_export_video = export_group.findChildren(QPushButton)[1]
        main_layout.addWidget(export_group)

        main_layout.addStretch()

        # Wire torque preview updates
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
            dtheta1, dphi, shoulder_coeffs, wrist_coeffs, t_end,
            b1, b2, mu1, mu2, gravity_on

        Raises
        ------
        ValueError if any input cannot be parsed.
        """

        def parse_float(widget: LabeledInput, name: str) -> float:
            try:
                return float(widget.value)
            except ValueError:
                raise ValueError(f"Cannot parse '{name}': '{widget.value}'") from None

        def parse_coeffs(widget: LabeledInput, name: str) -> list:
            try:
                parts = widget.value.split(",")
                return [float(p.strip()) for p in parts if p.strip()]
            except ValueError:
                raise ValueError(
                    f"Cannot parse '{name}' coefficients: '{widget.value}'"
                ) from None

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
            "gravity_on": self.chk_gravity.isChecked(),
        }

    def _update_torque_preview(self) -> None:
        try:
            t_end = float(self.inp_tend.value)
        except ValueError:
            t_end = 2.0

        def parse_coeffs(widget: LabeledInput) -> list:
            parts = widget.value.split(",")
            coeffs: list[float] = []
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
        self.lbl_frame.setText(f"Frame: 0 / {max_val}")

    def set_slider_value(self, val: int) -> None:
        """Update slider position without emitting signal (for animation)."""
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)
        self.lbl_frame.setText(f"Frame: {val} / {self.slider.maximum()}")

    def stop_playback(self) -> None:
        """Force stop playback state."""
        self.btn_play.setChecked(False)

    # ------------------------------------------------------------------
    # Accessors for toggle states
    # ------------------------------------------------------------------

    def gravity_on(self) -> bool:
        return self.chk_gravity.isChecked()

    def show_forces(self) -> bool:
        return self.chk_forces.isChecked()
