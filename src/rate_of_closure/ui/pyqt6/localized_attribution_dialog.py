"""Explicit PyQt authoring for a separate localized paired study."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import cast

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.simulation.pipeline import configured_swing_sample_times
from rate_of_closure.variation._localized_attribution_contract import TARGET_REGISTRY
from rate_of_closure.variation.localized_attribution import AttributionTarget
from rate_of_closure.variation.localized_attribution_producer import (
    LocalizedAttributionDesign,
)
from rate_of_closure.variation.simulation_adapter import spatial_point_ids_for_source
from shared.python.contracts import require
from shared.python.swing_sim.variation import (
    LOCALIZED_TORQUE_VARIABLE_JOINTS,
    NoiseSpec,
    VariationPlan,
)

_DESIGN_ID = "pyqt.explicit-paired-localized.v1"
_STATE_NAMES = ("position_x_m", "position_y_m", "position_z_m")


@dataclass(frozen=True)
class StateTargetSelection:
    """Exact spatial point and sample-grid time selected for state responses."""

    point_id: str
    time_s: float

    def __post_init__(self) -> None:
        require(
            isinstance(self.point_id, str) and self.point_id.startswith("swing."),
            "state point must be a spatial swing.* ID",
        )
        require(
            isinstance(self.time_s, (int, float))
            and not isinstance(self.time_s, bool)
            and math.isfinite(self.time_s)
            and self.time_s >= 0.0,
            "state time must be finite and >= 0",
        )


def _target(
    name: str, target_id: str, selection: StateTargetSelection
) -> AttributionTarget:
    definition = TARGET_REGISTRY[name]
    is_state = definition.kind == "state"
    return AttributionTarget(
        target_id,
        definition.kind,
        name,
        definition.unit,
        definition.convention,
        selection.time_s if is_state else None,
        selection.point_id if is_state else None,
        definition.coordinate_frame,
    )


def _targets(selection: StateTargetSelection) -> tuple[AttributionTarget, ...]:
    state = tuple(_target(name, f"state.{name}", selection) for name in _STATE_NAMES)
    scalar = tuple(
        _target(name, f"{definition.kind}.{name}", selection)
        for name, definition in TARGET_REGISTRY.items()
        if definition.kind != "state"
    )
    return state + scalar


def _localized_specs(plan: VariationPlan) -> tuple[NoiseSpec, ...]:
    specs = tuple(
        spec
        for spec in plan.noise
        if spec.variable_key in LOCALIZED_TORQUE_VARIABLE_JOINTS
    )
    require(bool(specs), "paired study requires at least one localized torque source")
    return specs


def build_localized_attribution_design(
    plan: VariationPlan,
    config: SimulationConfig,
    deltas_nm: dict[str, float],
    selection: StateTargetSelection,
) -> LocalizedAttributionDesign:
    """Normalize current UI authority into one explicit paired producer design."""
    require(isinstance(plan, VariationPlan), "plan must be VariationPlan")
    require(isinstance(config, SimulationConfig), "config must be SimulationConfig")
    require(isinstance(selection, StateTargetSelection), "invalid state selection")
    specs = _localized_specs(plan)
    source_plan = replace(plan, noise=specs, n_runs=2 * len(specs), groups=())
    return LocalizedAttributionDesign(
        _DESIGN_ID,
        source_plan,
        config,
        _targets(selection),
        deltas_nm,
    )


class LocalizedAttributionRunDialog(QDialog):
    """Require confirmation of the exact separate planted-intervention design."""

    def __init__(
        self,
        plan: VariationPlan,
        config: SimulationConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure Separate Paired Study")
        self._plan = plan
        self._config = config
        self._specs = _localized_specs(plan)
        self._delta_editors: dict[str, QDoubleSpinBox] = {}
        self._explanation = QLabel(
            "Each source is evaluated as one explicit baseline and one run with "
            "only that source changed. Other global Monte Carlo factors remain "
            "fixed at their declared base values. Results are planted-intervention "
            "responses, not causal estimates."
        )
        self._explanation.setWordWrap(True)
        self._sources = self._build_source_table()
        self._point = QComboBox()
        self._point.setAccessibleName("Paired attribution spatial state point")
        self._time = QComboBox()
        self._time.setAccessibleName("Paired attribution exact state sample time")
        self._populate_state_selectors()
        self._summary = QLabel(
            f"{len(self._specs)} localized sources · "
            f"{2 * len(self._specs)} explicit trials · 17 fixed targets."
        )
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        accept_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if accept_button is None:
            raise RuntimeError("paired study dialog requires an accept button")
        accept_button.setText("Run Separate Paired Study")
        self._buttons.accepted.connect(self._accept_validated)
        self._buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        for widget in (
            self._explanation,
            self._sources,
            QLabel("State response point"),
            self._point,
            QLabel("Exact state response sample time"),
            self._time,
            self._summary,
            self._buttons,
        ):
            layout.addWidget(widget)

    def _build_source_table(self) -> QTableWidget:
        table = QTableWidget(len(self._specs), 5)
        table.setHorizontalHeaderLabels(
            ("Spec ID", "Variable", "Topological joint", "Window", "Planted Δ torque")
        )
        table.setAccessibleName("Localized paired attribution sources")
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for row, spec in enumerate(self._specs):
            assert spec.spec_id is not None and spec.time_window_s is not None
            joint = LOCALIZED_TORQUE_VARIABLE_JOINTS[spec.variable_key]
            window = f"[{spec.time_window_s[0]:g}, {spec.time_window_s[1]:g}) s"
            for column, value in enumerate(
                (spec.spec_id, spec.variable_key, joint, window)
            ):
                table.setItem(row, column, QTableWidgetItem(value))
            editor = QDoubleSpinBox()
            editor.setRange(-1_000_000.0, 1_000_000.0)
            editor.setDecimals(9)
            editor.setValue(spec.scale)
            editor.setSuffix(" N·m")
            editor.setAccessibleName(f"Planted delta torque for {spec.spec_id}")
            editor.setToolTip(
                "Finite nonzero planted torque change. The noise scale is only a "
                "starting suggestion and must be explicitly confirmed."
            )
            table.setCellWidget(row, 4, editor)
            self._delta_editors[spec.spec_id] = editor
        table.resizeColumnsToContents()
        return table

    def _populate_state_selectors(self) -> None:
        points = spatial_point_ids_for_source(self._config.source_kind)
        for point in points:
            self._point.addItem(point, point)
        clubhead = self._point.findData("swing.clubhead.reference")
        self._point.setCurrentIndex(max(clubhead, 0))
        times = tuple(
            float(value) for value in configured_swing_sample_times(self._config)
        )
        for value in times:
            self._time.addItem(f"{value:.9g} s", value)
        preferred = min(range(len(times)), key=lambda index: abs(times[index] - 0.02))
        self._time.setCurrentIndex(preferred)

    def build_design(self) -> LocalizedAttributionDesign:
        """Build the strict design or fail before the dialog is accepted."""
        deltas = {
            spec_id: editor.value() for spec_id, editor in self._delta_editors.items()
        }
        selection = StateTargetSelection(
            cast(str, self._point.currentData()), cast(float, self._time.currentData())
        )
        return build_localized_attribution_design(
            self._plan, self._config, deltas, selection
        )

    def _accept_validated(self) -> None:
        try:
            self.build_design()
        except Exception as error:  # noqa: BLE001 - keep invalid dialog open visibly
            self._summary.setText(f"Cannot run paired study: {error}")
            return
        self.accept()


__all__ = [
    "LocalizedAttributionRunDialog",
    "StateTargetSelection",
    "build_localized_attribution_design",
]
