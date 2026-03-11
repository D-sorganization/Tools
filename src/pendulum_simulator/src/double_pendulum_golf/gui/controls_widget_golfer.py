"""
Control panel widget for the golfer upper-body model.

Provides inputs for all 8-segment parameters, initial conditions,
7 joint torque polynomials, dissipation, and simulation settings.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .controls_utils import (
    STYLE_CHECK,
    STYLE_GROUP,
    parse_coeffs,
    parse_float,
    require_non_negative,
    require_positive,
)
from .controls_widget import LabeledInput


class ControlsWidgetGolfer(QWidget):
    """Parameter input panel for the golfer upper-body model."""

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
        "Address Position": {
            "m_hub": 2.0,
            "m_r_upper": 3.5,
            "m_r_fore": 2.0,
            "m_l_upper": 3.5,
            "m_l_fore": 2.0,
            "m_club": 0.5,
            "L_hub": 0.15,
            "L_r_upper": 0.35,
            "L_r_fore": 0.30,
            "L_l_upper": 0.35,
            "L_l_fore": 0.30,
            "L_club": 1.1,
            "d_rs": 0.20,
            "d_ls": 0.20,
            "grip_right": 0.05,
            "grip_left": 0.25,
            "m_clubhead": 0.2,
            "theta_hub": 0.0,
            "alpha_rs": 0.0,
            "alpha_re": 0.0,
            "alpha_rh": 0.0,
            "alpha_ls": 0.0,
            "alpha_le": 0.0,
            "alpha_lh": 0.0,
            "tau_hub": "0",
            "tau_rs": "0",
            "tau_re": "0",
            "tau_rh": "0",
            "tau_ls": "0",
            "tau_le": "0",
            "tau_lh": "0",
            "t_end": 2.0,
            "L_rscap": 0.12,
            "L_lscap": 0.12,
            "m_rscap": 0.5,
            "m_lscap": 0.5,
        },
        "Backswing Start": {
            "m_hub": 2.0,
            "m_r_upper": 3.5,
            "m_r_fore": 2.0,
            "m_l_upper": 3.5,
            "m_l_fore": 2.0,
            "m_club": 0.5,
            "L_hub": 0.15,
            "L_r_upper": 0.35,
            "L_r_fore": 0.30,
            "L_l_upper": 0.35,
            "L_l_fore": 0.30,
            "L_club": 1.1,
            "d_rs": 0.20,
            "d_ls": 0.20,
            "grip_right": 0.05,
            "grip_left": 0.25,
            "m_clubhead": 0.2,
            "theta_hub": 15.0,
            "alpha_rs": -10.0,
            "alpha_re": 20.0,
            "alpha_rh": -5.0,
            "alpha_ls": 10.0,
            "alpha_le": -15.0,
            "alpha_lh": 5.0,
            "tau_hub": "-5, 2",
            "tau_rs": "-10, 5",
            "tau_re": "0",
            "tau_rh": "0",
            "tau_ls": "10, -5",
            "tau_le": "0",
            "tau_lh": "0",
            "t_end": 2.0,
            "L_rscap": 0.12,
            "L_lscap": 0.12,
            "m_rscap": 0.5,
            "m_lscap": 0.5,
        },
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.setMaximumWidth(400)
        self._is_playing = False
        self._build_ui()
        self._apply_preset("Address Position")

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        main = QVBoxLayout(container)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)
        style_group = STYLE_GROUP

        # Preset selector
        preset_grp = QGroupBox("Preset")
        preset_grp.setStyleSheet(style_group)
        pl = QVBoxLayout(preset_grp)
        self.preset_combo = QComboBox()
        self.preset_combo.setStyleSheet(
            "background:#2a2a38;color:#e0e0f0;border:1px solid #505068;"
            "border-radius:3px;padding:4px;"
        )
        for name in self.PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        pl.addWidget(self.preset_combo)
        main.addWidget(preset_grp)

        # Segment masses
        mass_grp = QGroupBox("Segment Masses (kg)")
        mass_grp.setStyleSheet(style_group)
        ml = QVBoxLayout(mass_grp)
        self.inp_m_hub = LabeledInput("Hub", "2.0", "Hub standoff mass")
        self.inp_m_r_upper = LabeledInput("R Upper", "3.5", "Right upper arm")
        self.inp_m_r_fore = LabeledInput("R Fore", "2.0", "Right forearm")
        self.inp_m_l_upper = LabeledInput("L Upper", "3.5", "Left upper arm")
        self.inp_m_l_fore = LabeledInput("L Fore", "2.0", "Left forearm")
        self.inp_m_club = LabeledInput("Club", "0.5", "Club shaft mass")
        self.inp_m_clubhead = LabeledInput("Clubhead", "0.2", "Clubhead point mass")
        for w in [
            self.inp_m_hub,
            self.inp_m_r_upper,
            self.inp_m_r_fore,
            self.inp_m_l_upper,
            self.inp_m_l_fore,
            self.inp_m_club,
            self.inp_m_clubhead,
        ]:
            ml.addWidget(w)
        main.addWidget(mass_grp)

        # Segment lengths
        len_grp = QGroupBox("Segment Lengths (m)")
        len_grp.setStyleSheet(style_group)
        ll = QVBoxLayout(len_grp)
        self.inp_L_hub = LabeledInput("Hub", "0.15", "Hub standoff length")
        self.inp_L_r_upper = LabeledInput("R Upper", "0.35", "Right upper arm")
        self.inp_L_r_fore = LabeledInput("R Fore", "0.30", "Right forearm")
        self.inp_L_l_upper = LabeledInput("L Upper", "0.35", "Left upper arm")
        self.inp_L_l_fore = LabeledInput("L Fore", "0.30", "Left forearm")
        self.inp_L_club = LabeledInput("Club", "1.1", "Club total length")
        for w in [
            self.inp_L_hub,
            self.inp_L_r_upper,
            self.inp_L_r_fore,
            self.inp_L_l_upper,
            self.inp_L_l_fore,
            self.inp_L_club,
        ]:
            ll.addWidget(w)
        main.addWidget(len_grp)

        # Shoulder offsets and grip positions
        geom_grp = QGroupBox("Geometry")
        geom_grp.setStyleSheet(style_group)
        gl = QVBoxLayout(geom_grp)
        self.inp_d_rs = LabeledInput("d_RS (m)", "0.20", "Hub to right shoulder offset")
        self.inp_d_ls = LabeledInput("d_LS (m)", "0.20", "Hub to left shoulder offset")
        self.inp_grip_right = LabeledInput(
            "Grip R (m)", "0.05", "Right hand grip from club base"
        )
        self.inp_grip_left = LabeledInput(
            "Grip L (m)", "0.25", "Left hand grip from club base"
        )
        self.inp_L_rscap = LabeledInput(
            "R Scap (m)",
            "0.12",
            "Right scapula link length.\n"
            "Connects the hub bar endpoint to the right shoulder.\n"
            "Set to 0 to disable scapula.",
        )
        self.inp_L_lscap = LabeledInput(
            "L Scap (m)",
            "0.12",
            "Left scapula link length.\n"
            "Connects the hub bar endpoint to the left shoulder.\n"
            "Set to 0 to disable scapula.",
        )
        self.inp_m_rscap = LabeledInput("R Scap m", "0.5", "Right scapula mass (kg)")
        self.inp_m_lscap = LabeledInput("L Scap m", "0.5", "Left scapula mass (kg)")
        for w in [
            self.inp_d_rs,
            self.inp_d_ls,
            self.inp_grip_right,
            self.inp_grip_left,
            self.inp_L_rscap,
            self.inp_L_lscap,
            self.inp_m_rscap,
            self.inp_m_lscap,
        ]:
            gl.addWidget(w)
        main.addWidget(geom_grp)

        # Initial conditions
        ic_grp = QGroupBox("Initial Conditions (deg)")
        ic_grp.setStyleSheet(style_group)
        il = QVBoxLayout(ic_grp)
        self.inp_th_hub = LabeledInput("Hub", "0", "Hub angle")
        self.inp_a_rs = LabeledInput("R Shoulder", "0", "Relative angle")
        self.inp_a_re = LabeledInput("R Elbow", "0", "Relative angle")
        self.inp_a_rh = LabeledInput("R Wrist", "0", "Relative angle")
        self.inp_a_ls = LabeledInput("L Shoulder", "0", "Relative angle")
        self.inp_a_le = LabeledInput("L Elbow", "0", "Relative angle")
        self.inp_a_lh = LabeledInput("L Wrist", "0", "Relative angle")
        for w in [
            self.inp_th_hub,
            self.inp_a_rs,
            self.inp_a_re,
            self.inp_a_rh,
            self.inp_a_ls,
            self.inp_a_le,
            self.inp_a_lh,
        ]:
            il.addWidget(w)
        main.addWidget(ic_grp)

        # Torque polynomials
        torque_grp = QGroupBox("Torque Polynomials (c0, c1, ...)")
        torque_grp.setStyleSheet(style_group)
        tl = QVBoxLayout(torque_grp)
        self.inp_tau_hub = LabeledInput("Hub", "0", "Hub torque")
        self.inp_tau_rs = LabeledInput("R Shoulder", "0", "RS torque")
        self.inp_tau_re = LabeledInput("R Elbow", "0", "RE torque")
        self.inp_tau_rh = LabeledInput("R Wrist", "0", "RH torque")
        self.inp_tau_ls = LabeledInput("L Shoulder", "0", "LS torque")
        self.inp_tau_le = LabeledInput("L Elbow", "0", "LE torque")
        self.inp_tau_lh = LabeledInput("L Wrist", "0", "LH torque")
        for w in [
            self.inp_tau_hub,
            self.inp_tau_rs,
            self.inp_tau_re,
            self.inp_tau_rh,
            self.inp_tau_ls,
            self.inp_tau_le,
            self.inp_tau_lh,
        ]:
            tl.addWidget(w)
        main.addWidget(torque_grp)

        # Simulation time
        sim_grp = QGroupBox("Simulation")
        sim_grp.setStyleSheet(style_group)
        sl = QVBoxLayout(sim_grp)
        self.inp_tend = LabeledInput("Duration (s)", "2.0", "Total time")
        sl.addWidget(self.inp_tend)
        main.addWidget(sim_grp)

        # Dissipation
        diss_grp = QGroupBox("Dissipation")
        diss_grp.setStyleSheet(style_group)
        dl = QVBoxLayout(diss_grp)
        self.inp_b_hub = LabeledInput("b hub", "0.0", "Viscous (N·m·s)")
        self.inp_b_rs = LabeledInput("b RS", "0.0", "Viscous (N·m·s)")
        self.inp_b_re = LabeledInput("b RE", "0.0", "Viscous (N·m·s)")
        self.inp_b_rh = LabeledInput("b RH", "0.0", "Viscous (N·m·s)")
        self.inp_b_ls = LabeledInput("b LS", "0.0", "Viscous (N·m·s)")
        self.inp_b_le = LabeledInput("b LE", "0.0", "Viscous (N·m·s)")
        self.inp_b_lh = LabeledInput("b LH", "0.0", "Viscous (N·m·s)")
        for w in [
            self.inp_b_hub,
            self.inp_b_rs,
            self.inp_b_re,
            self.inp_b_rh,
            self.inp_b_ls,
            self.inp_b_le,
            self.inp_b_lh,
        ]:
            dl.addWidget(w)
        main.addWidget(diss_grp)

        # Run / Reset buttons
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setStyleSheet(
            "QPushButton{background:#2d6b3f;color:white;border:none;"
            "border-radius:5px;padding:10px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#3a8a52;}"
            "QPushButton:pressed{background:#1f5030;}"
        )
        self.btn_run.clicked.connect(self.run_requested.emit)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setStyleSheet(
            "QPushButton{background:#5a3030;color:white;border:none;"
            "border-radius:5px;padding:10px;font-size:13px;}"
            "QPushButton:hover{background:#7a4040;}"
        )
        self.btn_reset.clicked.connect(self.reset_requested.emit)

        btn_layout.addWidget(self.btn_run, stretch=2)
        btn_layout.addWidget(self.btn_reset, stretch=1)
        main.addLayout(btn_layout)

        # Playback
        play_grp = QGroupBox("Playback")
        play_grp.setStyleSheet(style_group)
        plb = QVBoxLayout(play_grp)

        ctrl_row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setStyleSheet(
            "QPushButton{background:#303050;color:#c0c0e0;"
            "border:1px solid #505068;border-radius:4px;padding:6px 12px;}"
            "QPushButton:checked{background:#504030;color:#f0d080;}"
        )
        self.btn_play.toggled.connect(self._on_play_toggled)
        ctrl_row.addWidget(self.btn_play)

        ctrl_row.addWidget(QLabel("Speed:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 5.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setStyleSheet(
            "background:#2a2a38;color:#e0e0f0;border:1px solid #505068;"
        )
        self.speed_spin.valueChanged.connect(lambda v: self.speed_changed.emit(v))
        ctrl_row.addWidget(self.speed_spin)
        plb.addLayout(ctrl_row)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal{background:#303048;height:6px;"
            "border-radius:3px;}"
            "QSlider::handle:horizontal{background:#7070a0;width:14px;"
            "margin:-5px 0;border-radius:7px;}"
        )
        self.slider.valueChanged.connect(self.frame_changed.emit)
        plb.addWidget(self.slider)
        main.addWidget(play_grp)

        # Export
        export_grp = QGroupBox("Export")
        export_grp.setStyleSheet(style_group)
        el = QHBoxLayout(export_grp)
        self.btn_export_data = QPushButton("Export Data")
        self.btn_export_data.setStyleSheet(
            "QPushButton{background:#303050;color:#c0c0e0;"
            "border:1px solid #505068;border-radius:4px;padding:6px 10px;}"
            "QPushButton:hover{background:#3a3a60;}"
        )
        self.btn_export_data.clicked.connect(self.export_data_requested.emit)
        self.btn_export_video = QPushButton("Export Video")
        self.btn_export_video.setStyleSheet(
            "QPushButton{background:#303050;color:#c0c0e0;"
            "border:1px solid #505068;border-radius:4px;padding:6px 10px;}"
            "QPushButton:hover{background:#3a3a60;}"
        )
        self.btn_export_video.clicked.connect(self.export_video_requested.emit)
        el.addWidget(self.btn_export_data)
        el.addWidget(self.btn_export_video)
        main.addWidget(export_grp)

        # Physics & Display toggles
        vis_grp = QGroupBox("Physics & Display")
        vis_grp.setStyleSheet(style_group)
        vl = QVBoxLayout(vis_grp)
        self.chk_gravity = QCheckBox("Gravity enabled")
        self.chk_gravity.setChecked(True)
        self.chk_gravity.setStyleSheet(STYLE_CHECK)
        self.chk_gravity.toggled.connect(self.gravity_changed.emit)
        self.chk_forces = QCheckBox("Show force vectors")
        self.chk_forces.setChecked(False)
        self.chk_forces.setStyleSheet(STYLE_CHECK)
        self.chk_forces.toggled.connect(self.forces_changed.emit)
        self.chk_forces.setVisible(False)  # #1143: force toggle lives in toolstrip
        vl.addWidget(self.chk_gravity)
        # chk_forces hidden — toolstrip is the single source of truth (#1143)
        main.addWidget(vis_grp)

        main.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)

    def _apply_preset(self, name: str) -> None:
        if name not in self.PRESETS:
            return
        p = self.PRESETS[name]
        self.inp_m_hub.set_value(str(p["m_hub"]))
        self.inp_m_r_upper.set_value(str(p["m_r_upper"]))
        self.inp_m_r_fore.set_value(str(p["m_r_fore"]))
        self.inp_m_l_upper.set_value(str(p["m_l_upper"]))
        self.inp_m_l_fore.set_value(str(p["m_l_fore"]))
        self.inp_m_club.set_value(str(p["m_club"]))
        self.inp_m_clubhead.set_value(str(p["m_clubhead"]))
        self.inp_L_hub.set_value(str(p["L_hub"]))
        self.inp_L_r_upper.set_value(str(p["L_r_upper"]))
        self.inp_L_r_fore.set_value(str(p["L_r_fore"]))
        self.inp_L_l_upper.set_value(str(p["L_l_upper"]))
        self.inp_L_l_fore.set_value(str(p["L_l_fore"]))
        self.inp_L_club.set_value(str(p["L_club"]))
        self.inp_d_rs.set_value(str(p["d_rs"]))
        self.inp_d_ls.set_value(str(p["d_ls"]))
        self.inp_grip_right.set_value(str(p["grip_right"]))
        self.inp_grip_left.set_value(str(p["grip_left"]))
        self.inp_th_hub.set_value(str(p["theta_hub"]))
        self.inp_a_rs.set_value(str(p["alpha_rs"]))
        self.inp_a_re.set_value(str(p["alpha_re"]))
        self.inp_a_rh.set_value(str(p["alpha_rh"]))
        self.inp_a_ls.set_value(str(p["alpha_ls"]))
        self.inp_a_le.set_value(str(p["alpha_le"]))
        self.inp_a_lh.set_value(str(p["alpha_lh"]))
        self.inp_tau_hub.set_value(str(p["tau_hub"]))
        self.inp_tau_rs.set_value(str(p["tau_rs"]))
        self.inp_tau_re.set_value(str(p["tau_re"]))
        self.inp_tau_rh.set_value(str(p["tau_rh"]))
        self.inp_tau_ls.set_value(str(p["tau_ls"]))
        self.inp_tau_le.set_value(str(p["tau_le"]))
        self.inp_tau_lh.set_value(str(p["tau_lh"]))
        self.inp_tend.set_value(str(p["t_end"]))
        self.inp_L_rscap.set_value(str(p.get("L_rscap", 0.12)))
        self.inp_L_lscap.set_value(str(p.get("L_lscap", 0.12)))
        self.inp_m_rscap.set_value(str(p.get("m_rscap", 0.5)))
        self.inp_m_lscap.set_value(str(p.get("m_lscap", 0.5)))

    def get_params(self) -> dict:
        """Parse all inputs into a simulation parameter dict.

        Raises ValueError on invalid input.
        """
        params = {
            "m_hub": parse_float(self.inp_m_hub, "m_hub"),
            "m_r_upper": parse_float(self.inp_m_r_upper, "m_r_upper"),
            "m_r_fore": parse_float(self.inp_m_r_fore, "m_r_fore"),
            "m_l_upper": parse_float(self.inp_m_l_upper, "m_l_upper"),
            "m_l_fore": parse_float(self.inp_m_l_fore, "m_l_fore"),
            "m_club": parse_float(self.inp_m_club, "m_club"),
            "m_clubhead": parse_float(self.inp_m_clubhead, "m_clubhead"),
            "L_hub": parse_float(self.inp_L_hub, "L_hub"),
            "L_r_upper": parse_float(self.inp_L_r_upper, "L_r_upper"),
            "L_r_fore": parse_float(self.inp_L_r_fore, "L_r_fore"),
            "L_l_upper": parse_float(self.inp_L_l_upper, "L_l_upper"),
            "L_l_fore": parse_float(self.inp_L_l_fore, "L_l_fore"),
            "L_club": parse_float(self.inp_L_club, "L_club"),
            "d_rs": parse_float(self.inp_d_rs, "d_rs"),
            "d_ls": parse_float(self.inp_d_ls, "d_ls"),
            "grip_right": parse_float(self.inp_grip_right, "grip_right"),
            "grip_left": parse_float(self.inp_grip_left, "grip_left"),
            "theta_hub_rad": np.radians(parse_float(self.inp_th_hub, "theta_hub")),
            "alpha_rs_rad": np.radians(parse_float(self.inp_a_rs, "alpha_rs")),
            "alpha_re_rad": np.radians(parse_float(self.inp_a_re, "alpha_re")),
            "alpha_rh_rad": np.radians(parse_float(self.inp_a_rh, "alpha_rh")),
            "alpha_ls_rad": np.radians(parse_float(self.inp_a_ls, "alpha_ls")),
            "alpha_le_rad": np.radians(parse_float(self.inp_a_le, "alpha_le")),
            "alpha_lh_rad": np.radians(parse_float(self.inp_a_lh, "alpha_lh")),
            "hub_coeffs": parse_coeffs(self.inp_tau_hub, "Hub torque"),
            "rs_coeffs": parse_coeffs(self.inp_tau_rs, "RS torque"),
            "re_coeffs": parse_coeffs(self.inp_tau_re, "RE torque"),
            "rh_coeffs": parse_coeffs(self.inp_tau_rh, "RH torque"),
            "ls_coeffs": parse_coeffs(self.inp_tau_ls, "LS torque"),
            "le_coeffs": parse_coeffs(self.inp_tau_le, "LE torque"),
            "lh_coeffs": parse_coeffs(self.inp_tau_lh, "LH torque"),
            "t_end": parse_float(self.inp_tend, "Duration"),
            "gravity_on": self.chk_gravity.isChecked(),
            "b_hub": parse_float(self.inp_b_hub, "b_hub"),
            "b_rs": parse_float(self.inp_b_rs, "b_rs"),
            "b_re": parse_float(self.inp_b_re, "b_re"),
            "b_rh": parse_float(self.inp_b_rh, "b_rh"),
            "b_ls": parse_float(self.inp_b_ls, "b_ls"),
            "b_le": parse_float(self.inp_b_le, "b_le"),
            "b_lh": parse_float(self.inp_b_lh, "b_lh"),
            "L_rscap": parse_float(self.inp_L_rscap, "L_rscap"),
            "L_lscap": parse_float(self.inp_L_lscap, "L_lscap"),
            "m_rscap": parse_float(self.inp_m_rscap, "m_rscap"),
            "m_lscap": parse_float(self.inp_m_lscap, "m_lscap"),
        }
        for name in (
            "m_hub",
            "m_r_upper",
            "m_r_fore",
            "m_l_upper",
            "m_l_fore",
            "m_club",
            "L_hub",
            "L_r_upper",
            "L_r_fore",
            "L_l_upper",
            "L_l_fore",
            "L_club",
        ):
            require_positive(params[name], name)
        for name in (
            "m_clubhead",
            "d_rs",
            "d_ls",
            "grip_right",
            "grip_left",
            "b_hub",
            "b_rs",
            "b_re",
            "b_rh",
            "b_ls",
            "b_le",
            "b_lh",
        ):
            require_non_negative(params[name], name)
        if params["grip_right"] > params["L_club"]:
            raise ValueError("grip_right must be ≤ L_club")
        if params["grip_left"] > params["L_club"]:
            raise ValueError("grip_left must be ≤ L_club")
        return params

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
