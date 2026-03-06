"""
Control panel widget with parameter inputs, initial conditions,
torque polynomial editors, and playback controls.

UI/UX upgrade:
- Compact two-column layout (m1/m2, L1/L2, etc. side-by-side)
- Adjustable sampling time (dt) for precision vs speed tradeoff
- Gravity on/off toggle & force vector toggle
- Force vector scale slider
- gravity_changed, forces_changed, force_scale_changed signals
- Frame label showing current / total
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

# ── Style constants ──────────────────────────────────────────────────────────
_STYLE_GROUP = (
    "QGroupBox { color: #c8c8e0; border: 1px solid #404060;"
    "border-radius: 6px; margin-top: 10px; padding-top: 14px;"
    "font-weight: bold; font-size: 10px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
)
_STYLE_SLIDER = (
    "QSlider::groove:horizontal { background: #252540; height: 6px; border-radius: 3px; }"
    "QSlider::sub-page:horizontal { background: #5060a0; border-radius: 3px; }"
    "QSlider::handle:horizontal { background: #8090d0; width: 14px;"
    "margin: -5px 0; border-radius: 7px; border: 1px solid #6070b0; }"
    "QSlider::handle:horizontal:hover { background: #a0b0f0; }"
)
_STYLE_CHECK = (
    "QCheckBox { color: #b8b8d0; font-size: 11px; spacing: 4px; }"
    "QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #484868;"
    "border-radius: 3px; background: #22223a; }"
    "QCheckBox::indicator:checked { background: #5060a0; border-color: #7080c0; }"
)
_STYLE_EDIT = (
    "background: #1e1e30; color: #e0e0f0; border: 1px solid #404060;"
    "border-radius: 3px; padding: 2px 5px; font-family: monospace; font-size: 10px;"
)
_STYLE_LABEL = "color: #9090b0; font-size: 10px;"
_STYLE_SPIN = (
    "background: #1e1e30; color: #e0e0f0; border: 1px solid #404060;"
    "border-radius: 3px; padding: 1px 4px; font-size: 10px;"
)


class LabeledInput(QWidget):
    """A label + line-edit pair used throughout the control panel."""

    def __init__(
        self,
        label: str,
        default: str,
        tooltip: str = "",
        label_width: int = 80,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        lbl = QLabel(label)
        lbl.setFixedWidth(label_width)
        lbl.setStyleSheet(_STYLE_LABEL)
        layout.addWidget(lbl)

        self.edit = QLineEdit(default)
        self.edit.setStyleSheet(_STYLE_EDIT)
        self.edit.setMinimumHeight(22)
        if tooltip:
            self.edit.setToolTip(tooltip)
        layout.addWidget(self.edit)

    @property
    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, text: str) -> None:
        self.edit.setText(text)


def _row(*widgets: QWidget) -> QHBoxLayout:
    """Helper: pack widgets into a horizontal row with no margin."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    for w in widgets:
        row.addWidget(w, stretch=1)
    return row


class ControlsWidget(QWidget):
    """Parameter input panel with presets and playback controls.

    Signals
    -------
    run_requested        : user clicks Run
    reset_requested      : user clicks Reset
    play_toggled(bool)   : play/pause toggled
    speed_changed(float) : playback speed changed
    frame_changed(int)   : timeline slider moved
    gravity_changed(bool): gravity on/off
    forces_changed(bool) : force vectors show/hide
    force_scale_changed(float): force vector display scale
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
    force_scale_changed = pyqtSignal(float)

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
        self.setMinimumWidth(270)
        self.setMaximumWidth(370)
        self._is_playing = False
        self._build_ui()
        self._apply_preset("Golf Swing (passive wrist)")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # ── Preset ───────────────────────────────────────────────────
        preset_group = QGroupBox("Preset")
        preset_group.setStyleSheet(_STYLE_GROUP)
        pl = QVBoxLayout(preset_group)
        pl.setContentsMargins(4, 10, 4, 4)
        pl.setSpacing(3)
        self.preset_combo = QComboBox()
        self.preset_combo.setStyleSheet(
            "background: #1e1e30; color: #e0e0f0; border: 1px solid #404060;"
            "border-radius: 3px; padding: 3px; font-size: 10px;"
        )
        for name in self.PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        pl.addWidget(self.preset_combo)
        main_layout.addWidget(preset_group)

        # ── Physical parameters (2-col) ───────────────────────────────
        phys_group = QGroupBox("Physical Parameters")
        phys_group.setStyleSheet(_STYLE_GROUP)
        pl2 = QVBoxLayout(phys_group)
        pl2.setContentsMargins(4, 12, 4, 4)
        pl2.setSpacing(3)
        lw = 56  # label width for paired inputs
        self.inp_m1 = LabeledInput("m1 (kg)", "5.0", "Mass of arm segment", lw)
        self.inp_m2 = LabeledInput("m2 (kg)", "0.5", "Mass of club segment", lw)
        self.inp_L1 = LabeledInput("L1 (m)", "0.6", "Length of arm segment", lw)
        self.inp_L2 = LabeledInput("L2 (m)", "1.0", "Length of club segment", lw)
        pl2.addLayout(_row(self.inp_m1, self.inp_m2))
        pl2.addLayout(_row(self.inp_L1, self.inp_L2))
        main_layout.addWidget(phys_group)

        # ── Initial conditions (2-col) ────────────────────────────────
        ic_group = QGroupBox("Initial Conditions")
        ic_group.setStyleSheet(_STYLE_GROUP)
        pl3 = QVBoxLayout(ic_group)
        pl3.setContentsMargins(4, 12, 4, 4)
        pl3.setSpacing(3)
        self.inp_theta1 = LabeledInput("θ1°", "120", "Arm angle from vertical", lw)
        self.inp_phi = LabeledInput("φ°", "-90", "Club angle relative to arm", lw)
        self.inp_dtheta1 = LabeledInput("dθ1", "0", "Arm angular velocity rad/s", lw)
        self.inp_dphi = LabeledInput("dφ", "0", "Club angular velocity rad/s", lw)
        pl3.addLayout(_row(self.inp_theta1, self.inp_phi))
        pl3.addLayout(_row(self.inp_dtheta1, self.inp_dphi))
        main_layout.addWidget(ic_group)

        # ── Torque polynomials ────────────────────────────────────────
        torque_group = QGroupBox("Torque Polynomials  (c0, c1, c2 …)")
        torque_group.setStyleSheet(_STYLE_GROUP)
        pl4 = QVBoxLayout(torque_group)
        pl4.setContentsMargins(4, 12, 4, 4)
        pl4.setSpacing(3)
        self.inp_tau_shoulder = LabeledInput(
            "Shoulder", "-25, 10", "τ(t)=c0+c1·t+…", 56
        )
        self.inp_tau_wrist = LabeledInput("Wrist", "0", "τ(t)=c0+c1·t+…", 56)
        pl4.addWidget(self.inp_tau_shoulder)
        pl4.addWidget(self.inp_tau_wrist)

        # Function generator button
        self.btn_funcgen = QPushButton("📈 Function Generator…")
        self.btn_funcgen.setToolTip(
            "Open Function Generator to design a waveform and import it as torque coefficients"
        )
        self.btn_funcgen.setStyleSheet(
            "QPushButton { background: #282848; color: #b0b0e0; border: 1px solid #404068;"
            "border-radius: 4px; padding: 4px 8px; font-size: 10px; }"
            "QPushButton:hover { background: #32326a; }"
        )
        self.btn_funcgen.clicked.connect(self._open_function_generator)
        pl4.addWidget(self.btn_funcgen)
        main_layout.addWidget(torque_group)

        preview_group = QGroupBox("Torque Preview")
        preview_group.setStyleSheet(_STYLE_GROUP)
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(4, 12, 4, 4)
        self.torque_preview = TorquePreviewWidget()
        preview_layout.addWidget(self.torque_preview)
        main_layout.addWidget(preview_group)

        # ── Simulation + dt (2-col) ───────────────────────────────────
        sim_group = QGroupBox("Simulation")
        sim_group.setStyleSheet(_STYLE_GROUP)
        pl5 = QVBoxLayout(sim_group)
        pl5.setContentsMargins(4, 12, 4, 4)
        pl5.setSpacing(3)
        self.inp_tend = LabeledInput("Duration s", "2.0", "Total simulation time", lw)
        self.inp_dt = LabeledInput(
            "dt (s)",
            "0.005",
            "Integration step size.\n"
            "Smaller = higher accuracy but slower.\n"
            "Larger = faster but less precise.\n"
            "Typical: 0.001 (fine) → 0.02 (fast)",
            lw,
        )
        pl5.addLayout(_row(self.inp_tend, self.inp_dt))
        main_layout.addWidget(sim_group)

        # ── Dissipation (2-col) ───────────────────────────────────────
        diss_group = QGroupBox("Dissipation")
        diss_group.setStyleSheet(_STYLE_GROUP)
        pl_diss = QVBoxLayout(diss_group)
        pl_diss.setContentsMargins(4, 12, 4, 4)
        pl_diss.setSpacing(3)
        self.inp_b1 = LabeledInput("b1", "0.0", "Viscous damping shoulder (N·m·s)", lw)
        self.inp_b2 = LabeledInput("b2", "0.0", "Viscous damping wrist (N·m·s)", lw)
        self.inp_mu1 = LabeledInput("μ1", "0.0", "Coulomb friction shoulder (N·m)", lw)
        self.inp_mu2 = LabeledInput("μ2", "0.0", "Coulomb friction wrist (N·m)", lw)
        pl_diss.addLayout(_row(self.inp_b1, self.inp_b2))
        pl_diss.addLayout(_row(self.inp_mu1, self.inp_mu2))
        main_layout.addWidget(diss_group)

        # ── Physics & Display ─────────────────────────────────────────
        # Gravity toggle (force vectors / scales are in the toolstrip)
        vis_group = QGroupBox("Physics")
        vis_group.setStyleSheet(_STYLE_GROUP)
        vl = QVBoxLayout(vis_group)
        vl.setContentsMargins(4, 12, 4, 4)
        vl.setSpacing(4)

        self.chk_gravity = QCheckBox("🌍 Gravity on")
        self.chk_gravity.setChecked(True)
        self.chk_gravity.setStyleSheet(_STYLE_CHECK)
        self.chk_gravity.toggled.connect(self.gravity_changed.emit)
        vl.addWidget(self.chk_gravity)
        main_layout.addWidget(vis_group)

        # Hidden but kept so existing signal wiring still works:
        # btn_run, btn_reset, btn_play, speed_spin, slider, lbl_frame
        # (all signals are emitted by toolstrip now)
        self.btn_run = QPushButton()  # hidden
        self.btn_run.clicked.connect(self.run_requested.emit)
        self.btn_reset = QPushButton()  # hidden
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        self.btn_play = QPushButton()  # hidden
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.speed_spin = QDoubleSpinBox()  # hidden
        self.speed_spin.setRange(0.05, 20.0)
        self.speed_spin.setValue(1.0)
        self.speed_spin.valueChanged.connect(lambda v: self.speed_changed.emit(v))
        self.slider = QSlider(Qt.Orientation.Horizontal)  # hidden
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.frame_changed.emit)
        self.lbl_frame = QLabel("Frame: 0 / 0")  # hidden
        self.chk_forces = QCheckBox()  # hidden
        self.chk_forces.toggled.connect(self.forces_changed.emit)

        # ── Export ────────────────────────────────────────────────────
        export_group = QGroupBox("Export")
        export_group.setStyleSheet(_STYLE_GROUP)
        export_layout = QHBoxLayout(export_group)
        export_layout.setContentsMargins(4, 10, 4, 4)
        self.btn_export_data = QPushButton("📊 Data")
        self.btn_export_video = QPushButton("🎬 Video")
        for btn in [self.btn_export_data, self.btn_export_video]:
            btn.setStyleSheet(
                "QPushButton { background: #1e1e2e; color: #9090b0; border: 1px solid #3a3a58;"
                "border-radius: 4px; padding: 4px 6px; font-size: 10px; }"
                "QPushButton:hover { background: #282848; }"
            )
        self.btn_export_data.clicked.connect(self.export_data_requested.emit)
        self.btn_export_video.clicked.connect(self.export_video_requested.emit)
        export_layout.addWidget(self.btn_export_data)
        export_layout.addWidget(self.btn_export_video)
        main_layout.addWidget(export_group)
        main_layout.addStretch()

        # Wire torque preview
        self.inp_tau_shoulder.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tau_wrist.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tend.edit.textChanged.connect(self._update_torque_preview)

    # ------------------------------------------------------------------
    # Function generator integration
    # ------------------------------------------------------------------

    def _open_function_generator(self) -> None:
        """Open Function Generator as a dialog for torque design."""
        from .function_generator_dialog import FunctionGeneratorDialog

        dlg = FunctionGeneratorDialog(self)
        dlg.torque_imported.connect(self._on_torque_imported)
        dlg.exec()

    def _on_torque_imported(self, joint: str, coeffs: list[float]) -> None:
        """Receive torque profile imported from Function Generator."""
        coeffs_str = ", ".join(f"{c:.4g}" for c in coeffs)
        if joint.lower() == "shoulder":
            self.inp_tau_shoulder.set_value(coeffs_str)
        else:
            self.inp_tau_wrist.set_value(coeffs_str)
        self._update_torque_preview()

    # ------------------------------------------------------------------
    # Preset application
    # ------------------------------------------------------------------

    def _apply_preset(self, name: str) -> None:
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
    # Value parsing
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Parse all input fields and return as a dict.

        Returns
        -------
        dict with keys: m1, m2, L1, L2, theta1_rad, phi_rad,
            dtheta1, dphi, shoulder_coeffs, wrist_coeffs, t_end,
            dt, b1, b2, mu1, mu2, gravity_on

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

        dt_raw = parse_float(self.inp_dt, "dt")
        dt = max(1e-5, min(0.1, dt_raw))  # clamp to sensible range

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
            "dt": dt,
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

        self.torque_preview.set_duration(t_end)
        self.torque_preview.set_profiles(
            [
                ("Shoulder", parse_coeffs(self.inp_tau_shoulder), QColor(230, 120, 50)),
                ("Wrist", parse_coeffs(self.inp_tau_wrist), QColor(120, 180, 230)),
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
        self.slider.setRange(0, max_val)
        self.lbl_frame.setText(f"Frame: 0 / {max_val}")

    def set_slider_value(self, val: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)
        self.lbl_frame.setText(f"Frame: {val} / {self.slider.maximum()}")

    def stop_playback(self) -> None:
        self.btn_play.setChecked(False)

    def gravity_on(self) -> bool:
        return self.chk_gravity.isChecked()

    def show_forces(self) -> bool:
        return self.chk_forces.isChecked()
