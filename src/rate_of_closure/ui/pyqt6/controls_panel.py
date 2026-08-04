"""Input controls for the rate-of-closure impact explorer.

One widget owning every scenario input, emitting ``scenarioChanged`` with a
fresh :class:`~rate_of_closure.model.ImpactScenario` whenever any value
moves. The window never reaches into individual spin boxes (LoD) — it
consumes complete scenarios.
"""

from __future__ import annotations

import logging
from dataclasses import fields

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.model import ImpactScenario
from rate_of_closure.presets import PRESETS, preset_names

logger = logging.getLogger(__name__)

__all__ = ["ControlsPanel"]

#: field name -> (label, unit suffix, decimals, single step)
_FIELD_SPECS: dict[str, tuple[str, str, int, float]] = {
    "clubhead_speed_mph": ("Clubhead Speed", " mph", 1, 1.0),
    "omega_plane_dps": ("In-Plane Rotation (SPV)", " deg/s", 0, 50.0),
    "omega_shaft_dps": ("About-Shaft Rotation (HTV)", " deg/s", 0, 50.0),
    "lie_angle_deg": ("Shaft Lie at Impact", " deg", 1, 1.0),
    "com_to_face_mm": ("GC to Face Center", " mm", 1, 1.0),
    "impact_offset_toe_mm": ("Impact Toward Toe", " mm", 1, 1.0),
    "impact_offset_high_mm": ("Impact Above Center", " mm", 1, 1.0),
    "contact_duration_us": ("Contact Duration", " µs", 0, 10.0),
}

_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Delivery", ("clubhead_speed_mph", "lie_angle_deg")),
    ("Rotation Rates", ("omega_plane_dps", "omega_shaft_dps")),
    (
        "Geometry",
        ("com_to_face_mm", "impact_offset_toe_mm", "impact_offset_high_mm"),
    ),
    ("Contact", ("contact_duration_us",)),
)


class ControlsPanel(QWidget):
    """Scenario inputs grouped the way the model is parameterised."""

    scenarioChanged = pyqtSignal(object)  # noqa: N815 - Qt signal convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._updating = False

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_preset_box())
        for title, names in _GROUPS:
            layout.addWidget(self._build_group(title, names))
        layout.addStretch(1)
        self.apply_preset(preset_names()[0])

    # ── construction ────────────────────────────────────────────────
    def _build_preset_box(self) -> QGroupBox:
        box = QGroupBox("Preset")
        form = QFormLayout(box)
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(preset_names())
        self._preset_combo.currentTextChanged.connect(self.apply_preset)
        form.addRow("Scenario", self._preset_combo)
        return box

    def _build_group(self, title: str, names: tuple[str, ...]) -> QGroupBox:
        box = QGroupBox(title)
        form = QFormLayout(box)
        bounds = ImpactScenario.__dataclass_fields__  # noqa: F841 - doc aid
        from rate_of_closure.model import _BOUNDS  # localized: single source

        for name in names:
            label, suffix, decimals, step = _FIELD_SPECS[name]
            spin = QDoubleSpinBox()
            low, high = _BOUNDS[name]
            spin.setRange(low, high)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setSuffix(suffix)
            spin.valueChanged.connect(self._on_value_changed)
            self._spins[name] = spin
            form.addRow(label, spin)
        return box

    # ── behaviour ───────────────────────────────────────────────────
    def apply_preset(self, name: str) -> None:
        """Load a named preset into the controls and emit the scenario."""
        preset = PRESETS.get(name)
        if preset is None:
            logger.warning("unknown preset requested: %s", name)
            return
        self._updating = True
        try:
            for field in fields(ImpactScenario):
                self._spins[field.name].setValue(getattr(preset, field.name))
        finally:
            self._updating = False
        self._emit()

    def scenario(self) -> ImpactScenario:
        """The scenario currently described by the controls."""
        return ImpactScenario(
            **{name: spin.value() for name, spin in self._spins.items()}
        )

    def _on_value_changed(self) -> None:
        if not self._updating:
            self._emit()

    def _emit(self) -> None:
        self.scenarioChanged.emit(self.scenario())
