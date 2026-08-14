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
from dataclasses import fields, replace

from PyQt6.QtCore import QSignalBlocker, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.club import (
    ClubAssemblyBinding,
    ClubSpec,
    club_names,
    get_club,
)
from rate_of_closure.model import _BOUNDS, ImpactScenario
from rate_of_closure.presets import PRESETS, preset_names
from rate_of_closure.ui.pyqt6.club_artifact_ui import (
    export_clubhead_engineering_sidecar,
    export_clubhead_stl,
    import_club_assembly_binding,
)
from rate_of_closure.ui.pyqt6.responsive_layout import HeightForWidthGroupBox
from rate_of_closure.units import (
    FIELD_GUIDANCE,
    QUANTITY_UNITS,
    convert_from_canonical,
    convert_to_canonical,
    set_display_distance_unit,
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
    "distance": "Distance",
}


class ControlsPanel(QWidget):
    """Scenario inputs grouped the way the model is parameterised."""

    scenarioChanged = pyqtSignal(object)  # noqa: N815 - Qt signal convention
    #: Emitted with a ClubSpec when the user asks for a parametric head.
    clubHeadRequested = pyqtSignal(object)  # noqa: N815 - Qt signal convention
    #: Emitted with the new unit when the Distance display unit changes
    #: (#4125 H6) so distance surfaces across the app re-render.
    distanceUnitChanged = pyqtSignal(str)  # noqa: N815 - Qt signal convention
    #: Publishes only an exact selected-spec binding, or None on invalidation.
    assemblyBindingChanged = pyqtSignal(object)  # noqa: N815 - Qt convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._unit_combos: dict[str, QComboBox] = {}
        #: Selected display unit per quantity; model stays canonical.
        self._units: dict[str, str] = {
            quantity: next(iter(table)) for quantity, table in QUANTITY_UNITS.items()
        }
        self._updating = False
        self._assembly_binding: ClubAssemblyBinding | None = None
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_preset_box())
        layout.addWidget(self._build_club_box())
        layout.addWidget(self._build_units_box())
        for title, names in _GROUPS:
            layout.addWidget(self._build_group(title, names))
        layout.addStretch(1)
        self.apply_preset(preset_names()[0])

    # ── construction ────────────────────────────────────────────────
    @staticmethod
    def _configure_form(form: QFormLayout) -> None:
        """Let labels wrap instead of imposing a wide control rail."""
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

    def _build_preset_box(self) -> QGroupBox:
        box: QGroupBox = HeightForWidthGroupBox("Preset")
        form = QFormLayout(box)
        self._configure_form(form)
        self._preset_combo = QComboBox()
        self._configure_combo(self._preset_combo)
        self._preset_combo.addItems(preset_names())
        self._preset_combo.setToolTip(
            "Load a sourced scenario preset (Cheetham 2014 tour data, "
            "the forum worked example, zero-rotation control, ...); "
            "every input stays editable afterwards."
        )
        self._preset_combo.currentTextChanged.connect(self.apply_preset)
        form.addRow("Scenario", self._preset_combo)
        return box

    def _build_club_box(self) -> QGroupBox:
        box: QGroupBox = HeightForWidthGroupBox("Club")
        form = QFormLayout(box)
        self._configure_form(form)

        self._club_combo = QComboBox()
        self._configure_combo(self._club_combo)
        self._club_combo.addItems(club_names())
        self._club_combo.setCurrentText("Driver 10.5°")
        self._club_combo.setToolTip(FIELD_GUIDANCE["club_selection"])
        self._club_combo.currentTextChanged.connect(self._on_club_changed)
        form.addRow("Club", self._club_combo)

        self._loft_spin = QDoubleSpinBox()
        self._loft_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._loft_spin.setKeyboardTracking(False)
        self._loft_spin.setDecimals(1)
        self._loft_spin.setRange(0.0, 70.0)
        self._loft_spin.setSuffix(" deg")
        self._loft_spin.setToolTip(FIELD_GUIDANCE["club_loft_deg"])
        self._loft_spin.setMinimumWidth(84)  # readable at small windows
        form.addRow("Loft", self._loft_spin)

        self._curvature_check = QCheckBox("Curved Face (Bulge && Roll)")
        self._curvature_check.setToolTip(FIELD_GUIDANCE["face_curvature_enabled"])
        self._curvature_check.toggled.connect(self._on_curvature_toggled)
        form.addRow(self._curvature_check)

        self._bulge_spin = QDoubleSpinBox()
        self._roll_spin = QDoubleSpinBox()
        for spin, key, label in (
            (self._bulge_spin, "face_bulge_radius_mm", "Bulge Radius"),
            (self._roll_spin, "face_roll_radius_mm", "Roll Radius"),
        ):
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setKeyboardTracking(False)
            spin.setDecimals(0)
            spin.setRange(100.0, 2000.0)
            spin.setSuffix(" mm")
            spin.setToolTip(FIELD_GUIDANCE[key])
            spin.setMinimumWidth(84)  # readable at small windows
            form.addRow(label, spin)

        self._generate_button = QPushButton("Generate Representative Head")
        self._generate_button.setToolTip(
            "Build a parametric head mesh from the selected club spec "
            "(loft, mass envelope, bulge && roll) and render it in the "
            "3D view in place of the wireframe."
        )
        self._generate_button.clicked.connect(self._on_generate_head)
        form.addRow(self._generate_button)

        self._export_head_button = QPushButton("Export Selected Head STL…")
        self._export_head_button.setToolTip(
            "Save the selected club's deterministic parametric head as a "
            "binary STL. The model computes in SI metres; exported unitless "
            "STL coordinates are millimetres in the canonical head frame "
            "(x target, y up, z toe)."
        )
        self._export_head_button.clicked.connect(self._on_export_head)
        form.addRow(self._export_head_button)
        self._import_assembly_button = QPushButton("Import Assembly Binding…")
        self._import_assembly_button.setToolTip(
            "Load a strict versioned binding for this exact selected club. "
            "Only qualified measured, manufacturer, CAD-integrated, or "
            "qualified-analysis sources can make complete CG and inertia "
            "properties available; mismatched identities fail closed."
        )
        self._import_assembly_button.clicked.connect(self._on_import_assembly)
        form.addRow(self._import_assembly_button)
        self._binding_status = QLabel(
            "No assembly binding loaded — complete CG and tensors unavailable."
        )
        self._binding_status.setWordWrap(True)
        form.addRow(self._binding_status)
        self._export_engineering_button = QPushButton("Export Engineering JSON…")
        self._export_engineering_button.setToolTip(
            "Save a versioned sidecar with the exact STL digest, frames, "
            "mass provenance, and explicit unavailable CG/tensor capabilities."
        )
        self._export_engineering_button.clicked.connect(self._on_export_engineering)
        form.addRow(self._export_engineering_button)
        self._export_status = QLabel("")
        self._export_status.setWordWrap(True)
        form.addRow(self._export_status)

        self._loft_spin.valueChanged.connect(self._clear_assembly_binding)
        self._curvature_check.toggled.connect(self._clear_assembly_binding)
        self._bulge_spin.valueChanged.connect(self._clear_assembly_binding)
        self._roll_spin.valueChanged.connect(self._clear_assembly_binding)

        self._on_club_changed(self._club_combo.currentText())
        return box

    def _build_units_box(self) -> QGroupBox:
        box: QGroupBox = HeightForWidthGroupBox("Units")
        form = QFormLayout(box)
        self._configure_form(form)
        for quantity, table in QUANTITY_UNITS.items():
            combo = QComboBox()
            self._configure_combo(combo)
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
        box: QGroupBox = HeightForWidthGroupBox(title)
        form = QFormLayout(box)
        self._configure_form(form)
        for name in names:
            label, quantity_or_suffix, decimals = _FIELD_SPECS[name]
            spin = QDoubleSpinBox()
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setKeyboardTracking(False)
            spin.setDecimals(decimals)
            spin.setToolTip(FIELD_GUIDANCE[name])
            spin.setMinimumWidth(84)  # readable at small windows
            self._configure_spin_range(spin, name)
            spin.valueChanged.connect(self._on_value_changed)
            self._spins[name] = spin
            form.addRow(label, spin)
        return box

    @staticmethod
    def _configure_combo(combo: QComboBox) -> None:
        """Keep compact rails responsive without truncating chosen values."""
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(10)

    # ── units ───────────────────────────────────────────────────────
    def unit_for(self, quantity: str) -> str:
        """The selected display unit for a quantity."""
        return self._units[quantity]

    def unit_selections(self) -> dict[str, str]:
        """Return a detached snapshot of every display-unit selection."""
        return dict(self._units)

    def apply_workspace_state(
        self,
        scenario: ImpactScenario,
        club: ClubSpec,
        units: dict[str, str],
    ) -> None:
        """Apply a validated explorer slice without intermediate emissions.

        The current native editor can vary a library club's loft and face
        curvature. Other static club fields must still match that library
        entry; unsupported documents fail before any control changes.
        """
        if not isinstance(scenario, ImpactScenario) or not isinstance(club, ClubSpec):
            raise TypeError("workspace scenario and club types are invalid")
        if set(units) != set(QUANTITY_UNITS) or any(
            value not in QUANTITY_UNITS[key] for key, value in units.items()
        ):
            raise ValueError("workspace units are invalid")
        base = get_club(club.name)
        variable_fields = {
            "loft_deg",
            "face_bulge_radius_m",
            "face_roll_radius_m",
        }
        if any(
            getattr(base, field.name) != getattr(club, field.name)
            for field in fields(ClubSpec)
            if field.name not in variable_fields
        ):
            raise ValueError("native editor cannot represent this custom club")
        self._updating = True
        try:
            for unit_quantity, unit in units.items():
                with QSignalBlocker(self._unit_combos[unit_quantity]):
                    self._unit_combos[unit_quantity].setCurrentText(unit)
                self._units[unit_quantity] = unit
            set_display_distance_unit(units["distance"])
            for name, spin in self._spins.items():
                self._configure_spin_range(spin, name)
            self._club_combo.setCurrentText(club.name)
            self._on_club_changed(club.name)
            self._loft_spin.setValue(club.loft_deg)
            curved = (
                club.face_bulge_radius_m is not None
                or club.face_roll_radius_m is not None
            )
            self._curvature_check.setChecked(curved)
            if club.face_bulge_radius_m is not None:
                self._bulge_spin.setValue(club.face_bulge_radius_m * 1000.0)
            if club.face_roll_radius_m is not None:
                self._roll_spin.setValue(club.face_roll_radius_m * 1000.0)
            for field in fields(ImpactScenario):
                value = getattr(scenario, field.name)
                field_quantity = self._quantity_of(field.name)
                displayed = (
                    value
                    if field_quantity is None
                    else convert_from_canonical(
                        field_quantity, self._units[field_quantity], value
                    )
                )
                self._spins[field.name].setValue(displayed)
        finally:
            self._updating = False
        self.distanceUnitChanged.emit(units["distance"])
        self._emit()

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
        if quantity == "distance":
            # Ball-flight distances (#4125 H6): a session-wide display
            # preference — every distance surface re-reads it on render.
            set_display_distance_unit(unit)
            self._units[quantity] = unit
            self.distanceUnitChanged.emit(unit)
            self._emit()
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

    # ── club group ──────────────────────────────────────────────────
    def club_spec(self) -> ClubSpec:
        """The library spec under the current loft and curvature overrides."""
        base = get_club(self._club_combo.currentText())
        curved = self._curvature_check.isChecked()
        return replace(
            base,
            loft_deg=self._loft_spin.value(),
            face_bulge_radius_m=(self._bulge_spin.value() / 1000.0 if curved else None),
            face_roll_radius_m=self._roll_spin.value() / 1000.0 if curved else None,
        )

    def set_club_name(self, name: str) -> None:
        """Select one library club through the panel's canonical control."""
        get_club(name)  # fail closed before touching the current selection
        self._club_combo.setCurrentText(name)

    def _clear_assembly_binding(self, _value: object = None) -> None:
        """Discard a binding when any identity-defining selected input changes."""
        self.clear_assembly_binding(
            "Assembly binding cleared — selected club specification changed."
        )

    def clear_assembly_binding(self, reason: str) -> None:
        """Discard the owned binding and publish one authoritative invalidation."""
        if self._assembly_binding is None:
            return
        self._assembly_binding = None
        self.assemblyBindingChanged.emit(None)
        self._binding_status.setText(reason)

    def _on_import_assembly(self) -> None:
        """Import a qualified binding for the exact current club selection."""
        binding = import_club_assembly_binding(
            self, self.club_spec(), self._binding_status
        )
        if binding is not None:
            self._assembly_binding = binding
            self.assemblyBindingChanged.emit(binding)
        else:
            self._assembly_binding = None
            self.assemblyBindingChanged.emit(None)

    def _on_export_head(self) -> None:
        """Export the complete current club specification as binary STL."""
        export_clubhead_stl(self, self.club_spec(), self._export_status)

    def _on_export_engineering(self) -> None:
        """Export the selected head's strict engineering JSON sidecar."""
        export_clubhead_engineering_sidecar(
            self,
            self.club_spec(),
            self._export_status,
            self._assembly_binding,
        )

    def _on_club_changed(self, name: str) -> None:
        """Adopt a library club: loft/curvature defaults, scenario plumbing.

        GC-to-face and lie are driven from the spec (the CG lies within
        a few millimetres of the geometric center, so ``cg_depth`` is
        the representative GC-to-face distance); the spins stay fully
        editable afterwards, preserving user overrides.
        """
        self._clear_assembly_binding()
        spec = get_club(name)
        self._loft_spin.setValue(spec.loft_deg)
        self._curvature_check.setChecked(spec.has_curved_face)
        if spec.face_bulge_radius_m is not None:
            self._bulge_spin.setValue(spec.face_bulge_radius_m * 1000.0)
        if spec.face_roll_radius_m is not None:
            self._roll_spin.setValue(spec.face_roll_radius_m * 1000.0)
        self._on_curvature_toggled(self._curvature_check.isChecked())
        for field_name, canonical in (
            ("com_to_face_mm", spec.cg_depth_m * 1000.0),
            ("lie_angle_deg", spec.lie_deg),
        ):
            spin = self._spins.get(field_name)
            if spin is None:  # construction order: scenario spins come later
                continue
            quantity = self._quantity_of(field_name)
            spin.setValue(
                canonical
                if quantity is None
                else convert_from_canonical(quantity, self._units[quantity], canonical)
            )

    def _on_curvature_toggled(self, enabled: bool) -> None:
        self._bulge_spin.setEnabled(enabled)
        self._roll_spin.setEnabled(enabled)

    def _on_generate_head(self) -> None:
        self.clubHeadRequested.emit(self.club_spec())

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
