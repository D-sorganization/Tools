"""Editable canonical inputs for the PyQt capability workflow."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
)

from rate_of_closure.application.capability_workflow import (
    CAPABILITY_WORKFLOW_INTEGER_BOUNDS,
    CAPABILITY_WORKFLOW_NUMERIC_BOUNDS,
    CapabilityWorkflowInputs,
)
from shared.python.swing_sim.flight.capability_contract import CapabilityObjective


@dataclass(frozen=True)
class _NumericSpec:
    key: str
    label: str
    unit: str
    minimum: float
    maximum: float
    step: float


@dataclass(frozen=True)
class _IntegerSpec:
    key: str
    label: str
    minimum: int
    maximum: int


def _numeric_spec(key: str, label: str, unit: str, step: float) -> _NumericSpec:
    minimum, maximum = CAPABILITY_WORKFLOW_NUMERIC_BOUNDS[key]
    return _NumericSpec(key, label, unit, minimum, maximum, step)


def _integer_spec(key: str, label: str) -> _IntegerSpec:
    minimum, maximum = CAPABILITY_WORKFLOW_INTEGER_BOUNDS[key]
    return _IntegerSpec(key, label, minimum, maximum)


_NUMERIC_SPECS = (
    _numeric_spec("ball_speed_mps", "Ball speed center", "m/s", 0.1),
    _numeric_spec("ball_speed_std_mps", "Ball speed std. dev.", "m/s", 0.1),
    _numeric_spec("launch_angle_deg", "Launch angle center", "deg", 0.1),
    _numeric_spec("launch_angle_std_deg", "Launch angle std. dev.", "deg", 0.1),
    _numeric_spec("launch_direction_deg", "Launch direction (+ right)", "deg", 0.1),
    _numeric_spec("launch_direction_std_deg", "Direction std. dev.", "deg", 0.1),
    _numeric_spec("total_spin_rpm", "Fixed total spin", "rpm", 10),
    _numeric_spec("spin_axis_tilt_deg", "Spin tilt (+ fade/right)", "deg", 0.1),
    _numeric_spec("target_distance_m", "Target distance", "m", 1),
    _numeric_spec("target_lateral_m", "Target lateral (+ right)", "m", 1),
    _numeric_spec("target_radius_m", "Target radius", "m", 1),
    _numeric_spec("max_time_s", "Maximum flight time", "s", 0.1),
    _numeric_spec(
        "trajectory_sample_interval_s", "Trajectory sample interval", "s", 0.001
    ),
)
_INTEGER_SPECS = (
    _integer_spec("candidate_budget", "Candidate budget"),
    _integer_spec("ensemble_size", "Trials per candidate"),
    _integer_spec("alternatives_count", "Alternatives retained"),
    _integer_spec("seed", "Deterministic seed"),
)


class CapabilityControls(QGroupBox):
    """Form that exposes every factor consumed by the interactive workflow."""

    changed = pyqtSignal()  # noqa: N815

    def __init__(self) -> None:
        super().__init__("Player, Club, Target, and Search Basis")
        self._numeric: dict[str, QDoubleSpinBox] = {}
        self._integers: dict[str, QSpinBox] = {}
        self._build_ui()
        self.set_inputs(CapabilityWorkflowInputs())

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        self.profile_id = QLineEdit()
        self.club_id = QLineEdit()
        self.objective = QComboBox()
        self.profile_id.setToolTip(
            "Stable identifier saved with this user-authored capability profile."
        )
        self.club_id.setToolTip(
            "Club key optimized by this workflow; it must match the profile club."
        )
        self.objective.setToolTip(
            "Ranking criterion applied to the complete retained shot ensemble."
        )
        for objective in CapabilityObjective:
            self.objective.addItem(objective.value.replace("_", " ").title(), objective)
        layout.addRow("Profile ID", self.profile_id)
        layout.addRow("Club ID", self.club_id)
        layout.addRow("Objective", self.objective)
        for numeric_spec in _NUMERIC_SPECS:
            self._add_numeric(layout, numeric_spec)
        for integer_spec in _INTEGER_SPECS:
            self._add_integer(layout, integer_spec)
        self.profile_id.textChanged.connect(self.changed)
        self.club_id.textChanged.connect(self.changed)
        self.objective.currentIndexChanged.connect(self.changed)

    def _add_numeric(self, layout: QFormLayout, spec: _NumericSpec) -> None:
        spin = QDoubleSpinBox()
        spin.setRange(spec.minimum, spec.maximum)
        spin.setDecimals(3)
        spin.setSingleStep(spec.step)
        spin.setSuffix(f" {spec.unit}")
        spin.setAccessibleName(spec.label)
        spin.setToolTip(
            f"{spec.label} in {spec.unit}; allowed range {spec.minimum:g} to "
            f"{spec.maximum:g}. Positive lateral and direction values are right "
            "of target."
        )
        spin.valueChanged.connect(self.changed)
        layout.addRow(spec.label, spin)
        self._numeric[spec.key] = spin

    def _add_integer(self, layout: QFormLayout, spec: _IntegerSpec) -> None:
        spin = QSpinBox()
        spin.setRange(spec.minimum, spec.maximum)
        spin.setAccessibleName(spec.label)
        spin.setToolTip(
            f"{spec.label}; allowed range {spec.minimum} to {spec.maximum}. "
            "Candidate budget times trials is the total model-evaluation count."
        )
        spin.valueChanged.connect(self.changed)
        layout.addRow(spec.label, spin)
        self._integers[spec.key] = spin

    def inputs(self) -> CapabilityWorkflowInputs:
        """Return an immutable canonical snapshot of every visible control."""
        numeric = {key: float(spin.value()) for key, spin in self._numeric.items()}
        integer = {key: int(spin.value()) for key, spin in self._integers.items()}
        return CapabilityWorkflowInputs(
            profile_id=self.profile_id.text(),
            club_id=self.club_id.text(),
            objective=CapabilityObjective(self.objective.currentData()),
            ball_speed_mps=numeric["ball_speed_mps"],
            ball_speed_std_mps=numeric["ball_speed_std_mps"],
            launch_angle_deg=numeric["launch_angle_deg"],
            launch_angle_std_deg=numeric["launch_angle_std_deg"],
            launch_direction_deg=numeric["launch_direction_deg"],
            launch_direction_std_deg=numeric["launch_direction_std_deg"],
            total_spin_rpm=numeric["total_spin_rpm"],
            spin_axis_tilt_deg=numeric["spin_axis_tilt_deg"],
            target_distance_m=numeric["target_distance_m"],
            target_lateral_m=numeric["target_lateral_m"],
            target_radius_m=numeric["target_radius_m"],
            max_time_s=numeric["max_time_s"],
            trajectory_sample_interval_s=numeric["trajectory_sample_interval_s"],
            candidate_budget=integer["candidate_budget"],
            ensemble_size=integer["ensemble_size"],
            alternatives_count=integer["alternatives_count"],
            seed=integer["seed"],
        )

    def set_inputs(self, inputs: CapabilityWorkflowInputs) -> None:
        """Replace the complete form while emitting only one invalidation."""
        self.blockSignals(True)
        self.profile_id.setText(inputs.profile_id)
        self.club_id.setText(inputs.club_id)
        self.objective.setCurrentIndex(self.objective.findData(inputs.objective))
        for key, numeric_spin in self._numeric.items():
            numeric_spin.setValue(float(getattr(inputs, key)))
        for key, integer_spin in self._integers.items():
            integer_spin.setValue(int(getattr(inputs, key)))
        self.blockSignals(False)
        self.changed.emit()


__all__ = ["CapabilityControls"]
