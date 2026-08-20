"""Club Tester and Heavy Hit parameter input controls (C6, H4)."""

from __future__ import annotations

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.club_tester_models import (
    CLUB_PRESETS,
    GOLFER_PRESETS,
    ClubTesterState,
    import_golfer_model,
)

logger = logging.getLogger(__name__)

__all__ = ["ClubTesterControlsPanel"]


def _spin(
    low: float,
    high: float,
    default: float,
    decimals: int,
    suffix: str,
    name: str,
    tip: str,
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setValue(default)
    box.setDecimals(decimals)
    if suffix:
        box.setSuffix(suffix)
    box.setAccessibleName(name)
    box.setToolTip(tip)
    return box


class ClubTesterControlsPanel(QWidget):
    """Controls for selecting baseline club, tweaking counterfactuals, and coupling."""

    runRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    sweepRequested = pyqtSignal()
    stateChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_baseline_box())
        layout.addWidget(self._build_counterfactual_box())
        layout.addWidget(self._build_delivery_box())
        layout.addWidget(self._build_coupling_box())
        layout.addWidget(self._build_action_box())
        layout.addStretch(1)

        self._connect_signals()

    def _build_baseline_box(self) -> QGroupBox:
        box = QGroupBox("Baseline Club")
        form = QFormLayout(box)
        self._club_combo = QComboBox()
        self._club_combo.setAccessibleName("Baseline Club Preset")
        self._club_combo.setToolTip(
            "Select the reference club model from the standard library."
        )
        self._club_combo.addItems(list(CLUB_PRESETS))
        form.addRow("Club Preset", self._club_combo)
        return box

    def _build_counterfactual_box(self) -> QGroupBox:
        box = QGroupBox("Counterfactual Tweaks")
        form = QFormLayout(box)

        self._mass_scale_spin = _spin(
            0.5,
            1.5,
            1.0,
            2,
            "x",
            "Head Mass Scale",
            "Scale factor on clubhead mass (0.5 to 1.5).",
        )
        form.addRow("Head Mass Scale", self._mass_scale_spin)

        self._cg_back_spin = _spin(
            -20.0,
            20.0,
            0.0,
            1,
            " mm",
            "CG Back Delta",
            "Center of gravity shift backward (+ = deeper CG).",
        )
        form.addRow("CG Back Offset", self._cg_back_spin)

        self._cg_toe_spin = _spin(
            -20.0,
            20.0,
            0.0,
            1,
            " mm",
            "CG Toe Delta",
            "Center of gravity shift toward toe (+ = toe side).",
        )
        form.addRow("CG Toe Offset", self._cg_toe_spin)

        self._loft_delta_spin = _spin(
            -4.0,
            4.0,
            0.0,
            1,
            "°",
            "Loft Delta",
            "Static face loft delta in degrees (-4° to +4°).",
        )
        form.addRow("Loft Delta", self._loft_delta_spin)

        self._ei_scale_spin = _spin(
            0.5,
            2.0,
            1.0,
            2,
            "x",
            "Shaft EI Scale",
            "Scale factor on shaft bending stiffness EI (0.5 to 2.0).",
        )
        form.addRow("Shaft EI Scale", self._ei_scale_spin)

        self._gj_scale_spin = _spin(
            0.5,
            2.0,
            1.0,
            2,
            "x",
            "Shaft GJ Scale",
            "Scale factor on shaft torsional stiffness GJ (0.5 to 2.0).",
        )
        form.addRow("Shaft GJ Scale", self._gj_scale_spin)

        return box

    def _build_delivery_box(self) -> QGroupBox:
        box = QGroupBox("Swing Delivery Kinematics")
        form = QFormLayout(box)

        self._omega_spin = _spin(
            10.0,
            70.0,
            39.0,
            1,
            " rad/s",
            "Grip Angular Velocity",
            "Grip rotation rate approaching impact.",
        )
        form.addRow("Grip Omega", self._omega_spin)

        self._alpha_spin = _spin(
            -300.0,
            100.0,
            -80.0,
            1,
            " rad/s²",
            "Grip Angular Acceleration",
            "Grip deceleration rate during release.",
        )
        form.addRow("Grip Alpha", self._alpha_spin)

        self._radius_spin = _spin(
            0.5,
            2.0,
            1.15,
            2,
            " m",
            "Effective Swing Radius",
            "Distance from swing center to grip.",
        )
        form.addRow("Swing Radius", self._radius_spin)

        self._downswing_spin = _spin(
            0.1,
            1.0,
            0.30,
            2,
            " s",
            "Downswing Duration",
            "Total time from transition to impact.",
        )
        form.addRow("Downswing Time", self._downswing_spin)

        self._recovery_spin = _spin(
            0.0,
            1.0,
            0.50,
            2,
            "",
            "Release Recovery Fraction",
            "Fraction of shaft bend recovered at impact.",
        )
        form.addRow("Release Recovery", self._recovery_spin)

        return box

    def _build_coupling_box(self) -> QGroupBox:
        box = QGroupBox("Heavy Hit (Hand/Body Coupling)")
        form = QFormLayout(box)

        self._coupling_check = QCheckBox("Enable Hand/Body Impact Coupling")
        self._coupling_check.setChecked(True)
        self._coupling_check.setAccessibleName("Enable Heavy Hit Coupling")
        self._coupling_check.setToolTip(
            "Model transient impulse transmission through shaft to hands during impact."
        )
        form.addRow(self._coupling_check)

        self._golfer_combo = QComboBox()
        self._golfer_combo.setAccessibleName("Golfer Boundary Preset")
        self._golfer_combo.setToolTip("Preset or imported golfer boundary parameters.")
        self._golfer_combo.addItems(list(GOLFER_PRESETS.keys()))
        form.addRow("Golfer Preset", self._golfer_combo)

        self._hand_mass_spin = _spin(
            0.5,
            20.0,
            2.5,
            2,
            " kg",
            "Effective Hand Mass",
            "Mass of hands and forearms coupled to grip.",
        )
        form.addRow("Effective Hand Mass", self._hand_mass_spin)

        self._grip_stiffness_spin = _spin(
            0.0,
            5000000.0,
            50000.0,
            0,
            " N/m",
            "Grip Stiffness",
            "Restoring stiffness of hands and body at grip.",
        )
        form.addRow("Grip Stiffness", self._grip_stiffness_spin)

        self._grip_damping_spin = _spin(
            0.0,
            1000.0,
            50.0,
            1,
            " N·s/m",
            "Grip Damping",
            "Viscous damping between grip and body.",
        )
        form.addRow("Grip Damping", self._grip_damping_spin)

        self._shaft_stiffness_spin = _spin(
            500.0,
            500000.0,
            10000.0,
            0,
            " N/m",
            "Shaft Longitudinal Stiffness",
            "Longitudinal/flexural shaft stiffness along hit direction.",
        )
        form.addRow("Shaft Stiffness", self._shaft_stiffness_spin)

        self._import_model_button = QPushButton(
            "Import Golfer Model (MJCF/URDF/.osim)…"
        )
        self._import_model_button.setAccessibleName("Import Golfer Model")
        self._import_model_button.setToolTip(
            "Load a biomechanical model exported from MuJoCo, Drake, Pinocchio, "
            "or OpenSim."
        )
        self._import_model_button.clicked.connect(self._on_import_model)
        form.addRow(self._import_model_button)

        return box

    def _build_action_box(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._run_button = QPushButton("Run Fitting Evaluation")
        self._run_button.setAccessibleName("Run Club Tester Evaluation")
        self._run_button.setToolTip(
            "Evaluate baseline club and counterfactual deltas through the full "
            "simulation pipeline."
        )
        self._run_button.setStyleSheet("font-weight: bold; padding: 6px;")
        self._run_button.clicked.connect(self.runRequested.emit)
        layout.addWidget(self._run_button)

        row = QHBoxLayout()
        self._sweep_button = QPushButton("Run Heavy Hit Sweep")
        self._sweep_button.setAccessibleName("Run Heavy Hit Sweep")
        self._sweep_button.setToolTip(
            "Run multi-axis decoupling sweep across hand stiffness, mass, and "
            "shaft rigidity."
        )
        self._sweep_button.clicked.connect(self.sweepRequested.emit)
        row.addWidget(self._sweep_button)

        self._export_button = QPushButton("Export Report JSON…")
        self._export_button.setAccessibleName("Export Fitting Report JSON")
        self._export_button.setToolTip(
            "Export deterministic fitting and coupling results as versioned JSON."
        )
        self._export_button.clicked.connect(self.exportRequested.emit)
        row.addWidget(self._export_button)

        layout.addLayout(row)
        return container

    def _connect_signals(self) -> None:
        self._club_combo.currentIndexChanged.connect(self.stateChanged.emit)
        for spin in (
            self._mass_scale_spin,
            self._cg_back_spin,
            self._cg_toe_spin,
            self._loft_delta_spin,
            self._ei_scale_spin,
            self._gj_scale_spin,
            self._omega_spin,
            self._alpha_spin,
            self._radius_spin,
            self._downswing_spin,
            self._recovery_spin,
            self._hand_mass_spin,
            self._grip_stiffness_spin,
            self._grip_damping_spin,
            self._shaft_stiffness_spin,
        ):
            spin.valueChanged.connect(self.stateChanged.emit)
        self._coupling_check.toggled.connect(self.stateChanged.emit)
        self._golfer_combo.currentIndexChanged.connect(self._on_golfer_preset_changed)

    def _on_golfer_preset_changed(self, index: int) -> None:
        preset_name = self._golfer_combo.currentText()
        if preset_name in GOLFER_PRESETS:
            preset = GOLFER_PRESETS[preset_name]
            self._hand_mass_spin.setValue(preset.effective_mass_kg)
            self._grip_stiffness_spin.setValue(preset.stiffness_n_m)
            self._grip_damping_spin.setValue(preset.damping_n_s_m)
            self.stateChanged.emit()

    def _on_import_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Golfer Model",
            "",
            "Model Files (*.xml *.urdf *.osim *.json);;All Files (*)",
        )
        if not file_path:
            return
        try:
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            chain = import_golfer_model(text, file_path)
            # Default hand reduction on first leaf body
            last_body = chain.bodies[-1]
            self._hand_mass_spin.setValue(last_body.mass_kg)
            if last_body.joint:
                self._grip_stiffness_spin.setValue(last_body.joint.stiffness)
                self._grip_damping_spin.setValue(last_body.joint.damping)
            self.stateChanged.emit()
            logger.info(
                "Imported golfer model: %s with %d bodies",
                chain.source_id,
                len(chain.bodies),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to import golfer model: %s", exc)

    def state(self) -> ClubTesterState:
        """Extract current control state as immutable dataclass."""
        preset_name = self._golfer_combo.currentText()
        provenance = (
            GOLFER_PRESETS[preset_name].provenance
            if preset_name in GOLFER_PRESETS
            else "custom_import"
        )
        return ClubTesterState(
            preset_club=self._club_combo.currentText(),
            head_mass_scale=self._mass_scale_spin.value(),
            cg_back_delta_m=self._cg_back_spin.value() / 1000.0,
            cg_toe_delta_m=self._cg_toe_spin.value() / 1000.0,
            loft_delta_deg=self._loft_delta_spin.value(),
            ei_scale=self._ei_scale_spin.value(),
            gj_scale=self._gj_scale_spin.value(),
            omega_rad_s=self._omega_spin.value(),
            alpha_rad_s2=self._alpha_spin.value(),
            swing_radius_m=self._radius_spin.value(),
            downswing_duration_s=self._downswing_spin.value(),
            release_recovery=self._recovery_spin.value(),
            enable_heavy_hit=self._coupling_check.isChecked(),
            grip_mass_kg=self._hand_mass_spin.value(),
            grip_stiffness_n_m=self._grip_stiffness_spin.value(),
            grip_damping_n_s_m=self._grip_damping_spin.value(),
            shaft_stiffness_n_m=self._shaft_stiffness_spin.value(),
            grip_provenance=provenance,
        )
