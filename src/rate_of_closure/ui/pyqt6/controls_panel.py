"""Input controls for the rate-of-closure impact explorer.

One widget owning every scenario input, emitting ``scenarioChanged`` with a
fresh :class:`~rate_of_closure.model.ImpactScenario` whenever any value
moves. The window never reaches into individual spin boxes (LoD) — it
consumes complete scenarios.

Values are typed, not spun: the entry boxes hide their step arrows, and
each carries hover guidance with a suggested golf-swing range and the
source of the suggestion. A Units group converts speed, rotation, and
length displays the way the UpstreamDrift apps do — the model itself
always stays canonical (mph, deg/s, mm, µs).
"""

from __future__ import annotations

import logging
from dataclasses import fields

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.model import _BOUNDS, ImpactScenario
from rate_of_closure.presets import PRESETS, preset_names
from rate_of_closure.units import (
    FIELD_GUIDANCE,
    QUANTITY_UNITS,
    convert_from_canonical,
    convert_to_canonical,
)

logger = logging.getLogger(__name__)

__all__ = ["ControlsPanel"]

#: field name -> (Title Case label, quantity or fixed suffix, decimals)
#: quantity names index QUANTITY_UNITS; fixed suffixes start with a space.
_FIELD_SPECS: dict[str, tuple[str, str, int]] = {
    "clubhead_speed_mph": ("Clubhead Speed", "speed", 1),
    "omega_plane_dps": ("In-Plane Rotation (SPV)", "rotation", 0),
    "omega_shaft_dps": ("About-Shaft Rotation (HTV)", "rotation", 0),
    "lie_angle_deg": ("Shaft Lie at Impact", " deg", 1),
    "com_to_face_mm": ("GC to Face Center", "length", 1),
    "impact_offset_toe_mm": ("Impact Toward Toe", "length", 1),
    "impact_offset_high_mm": ("Impact Above Center", "length", 1),
    "contact_duration_us": ("Contact Duration", " µs", 0),
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

_UNIT_LABELS: dict[str, str] = {
    "speed": "Speed",
    "rotation": "Rotation",
    "length": "Length",
}


class ControlsPanel(QWidget):
    """Scenario inputs grouped the way the model is parameterised."""

    scenarioChanged = pyqtSignal(object)  # noqa: N815 - Qt signal convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._unit_combos: dict[str, QComboBox] = {}
        #: Selected display unit per quantity; model stays canonical.
        self._units: dict[str, str] = {
            quantity: next(iter(table)) for quantity, table in QUANTITY_UNITS.items()
        }
        self._updating = False

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_preset_box())
        layout.addWidget(self._build_units_box())
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

    def _build_units_box(self) -> QGroupBox:
        box = QGroupBox("Units")
        form = QFormLayout(box)
        for quantity, table in QUANTITY_UNITS.items():
            combo = QComboBox()
            combo.addItems(list(table))
            combo.setToolTip(
                f"Display unit for {_UNIT_LABELS[quantity].lower()} inputs "
                "and results; the model always computes in canonical units."
            )
            combo.currentTextChanged.connect(
                lambda unit, q=quantity: self._on_unit_changed(q, unit)
            )
            self._unit_combos[quantity] = combo
            form.addRow(_UNIT_LABELS[quantity], combo)
        return box

    def _build_group(self, title: str, names: tuple[str, ...]) -> QGroupBox:
        box = QGroupBox(title)
        form = QFormLayout(box)
        for name in names:
            label, quantity_or_suffix, decimals = _FIELD_SPECS[name]
            spin = QDoubleSpinBox()
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setKeyboardTracking(False)
            spin.setDecimals(decimals)
            spin.setToolTip(FIELD_GUIDANCE[name])
            self._configure_spin_range(spin, name)
            spin.valueChanged.connect(self._on_value_changed)
            self._spins[name] = spin
            form.addRow(label, spin)
        return box

    # ── units ───────────────────────────────────────────────────────
    def unit_for(self, quantity: str) -> str:
        """The selected display unit for a quantity."""
        return self._units[quantity]

    def _quantity_of(self, name: str) -> str | None:
        quantity_or_suffix = _FIELD_SPECS[name][1]
        return None if quantity_or_suffix.startswith(" ") else quantity_or_suffix

    def _configure_spin_range(self, spin: QDoubleSpinBox, name: str) -> None:
        """Set range and suffix in the field's current display unit."""
        quantity = self._quantity_of(name)
        low, high = _BOUNDS[name]
        if quantity is None:
            spin.setRange(low, high)
            spin.setSuffix(_FIELD_SPECS[name][1])
            return
        unit = self._units[quantity]
        # Non-canonical units get extra decimals so switching units loses
        # no more than display resolution.
        base_decimals = _FIELD_SPECS[name][2]
        canonical_unit = next(iter(QUANTITY_UNITS[quantity]))
        spin.setDecimals(base_decimals if unit == canonical_unit else base_decimals + 3)
        spin.setRange(
            convert_from_canonical(quantity, unit, low),
            convert_from_canonical(quantity, unit, high),
        )
        spin.setSuffix(f" {unit}")

    def _on_unit_changed(self, quantity: str, unit: str) -> None:
        """Re-display every affected field in the new unit, same value."""
        previous = self._units[quantity]
        if unit == previous:
            return
        self._updating = True
        try:
            for name in _FIELD_SPECS:
                if self._quantity_of(name) != quantity:
                    continue
                spin = self._spins[name]
                canonical = convert_to_canonical(quantity, previous, spin.value())
                self._units[quantity] = unit
                self._configure_spin_range(spin, name)
                spin.setValue(convert_from_canonical(quantity, unit, canonical))
            self._units[quantity] = unit
        finally:
            self._updating = False
        self._emit()

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
                canonical = getattr(preset, field.name)
                quantity = self._quantity_of(field.name)
                displayed = (
                    canonical
                    if quantity is None
                    else convert_from_canonical(
                        quantity, self._units[quantity], canonical
                    )
                )
                self._spins[field.name].setValue(displayed)
        finally:
            self._updating = False
        self._emit()

    def scenario(self) -> ImpactScenario:
        """The scenario currently described by the controls (canonical)."""
        values: dict[str, float] = {}
        for name, spin in self._spins.items():
            quantity = self._quantity_of(name)
            values[name] = (
                spin.value()
                if quantity is None
                else convert_to_canonical(quantity, self._units[quantity], spin.value())
            )
        return ImpactScenario(**values)

    def _on_value_changed(self) -> None:
        if not self._updating:
            self._emit()

    def _emit(self) -> None:
        self.scenarioChanged.emit(self.scenario())
