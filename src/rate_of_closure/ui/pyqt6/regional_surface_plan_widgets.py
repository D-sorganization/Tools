"""Reusable PyQt6 inputs for the regional surface-plan editor."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from rate_of_closure.application.regional_surface_plan import (
    RegionalOverlayDraft,
    SurfaceMaterialDraft,
)
from rate_of_closure.ui.pyqt6.engineering_number_input import (
    NumberInputSpec,
    engineering_number_input,
)
from shared.python.swing_sim.canonical_numeric_json import (
    MAX_CANONICAL_SAFE_INTEGER,
)

_MAX_SAFE_FLOAT = float(MAX_CANONICAL_SAFE_INTEGER)
_MAX_SAFE_INTEGER_TEXT = str(MAX_CANONICAL_SAFE_INTEGER)

_MATERIAL_FIELDS = (
    ("normal_restitution", "Normal restitution", "", 0.01, 0.0, 1.0),
    ("static_friction", "Static friction", "", 0.01, 0.0, 5.0),
    ("kinetic_friction", "Kinetic friction", "", 0.01, 0.0, 5.0),
    ("rolling_resistance", "Rolling resistance", "", 0.01, 0.0, 1.0),
    ("firmness_pa", "Firmness", " Pa", 1_000.0, 1e-11, _MAX_SAFE_FLOAT),
    ("hardness_fraction", "Hardness", " fraction", 0.01, 0.0, 1.0),
    ("grass_height_m", "Grass height", " m", 0.001, 0.0, _MAX_SAFE_FLOAT),
    ("compressibility_fraction", "Compressibility", " fraction", 0.01, 0.0, 1.0),
    (
        "compression_damping_fraction",
        "Compression damping",
        " fraction",
        0.01,
        0.0,
        1.0,
    ),
    (
        "turf_density_kg_m3",
        "Turf density",
        " kg/m³",
        1.0,
        0.0,
        _MAX_SAFE_FLOAT,
    ),
    ("moisture_fraction", "Moisture", " fraction", 0.01, 0.0, 1.0),
)


def number_input(
    name: str,
    value: float,
    spec: NumberInputSpec | None = None,
) -> QDoubleSpinBox:
    """Create one canonical-precision accessible SI number input."""
    resolved_spec = spec or NumberInputSpec(
        minimum=-_MAX_SAFE_FLOAT,
        maximum=_MAX_SAFE_FLOAT,
        decimals=11,
    )
    return cast(QDoubleSpinBox, engineering_number_input(name, value, resolved_spec))


def coordinate_input(name: str, value: float) -> QDoubleSpinBox:
    """Create one canonical-precision coordinate editor."""
    return number_input(
        name,
        value,
        NumberInputSpec(
            suffix=" m",
            minimum=-_MAX_SAFE_FLOAT,
            maximum=_MAX_SAFE_FLOAT,
            decimals=11,
        ),
    )


class _SafeIntegerValidator(QValidator):
    """Accept one decimal integer in the shared JSON-safe nonnegative range."""

    def validate(
        self, input_text: str | None, position: int
    ) -> tuple[QValidator.State, str, int]:
        input_text = input_text or ""
        if input_text == "":
            state = QValidator.State.Intermediate
        elif not input_text.isascii() or not input_text.isdecimal():
            state = QValidator.State.Invalid
        elif len(input_text) < len(_MAX_SAFE_INTEGER_TEXT):
            state = QValidator.State.Acceptable
        elif input_text <= _MAX_SAFE_INTEGER_TEXT:
            state = QValidator.State.Acceptable
        else:
            state = QValidator.State.Invalid
        return state, input_text, position


class SafeIntegerEdit(QLineEdit):
    """Exact integer editor that does not narrow the v1 safe-integer contract."""

    def __init__(self, name: str, value: int) -> None:
        super().__init__()
        self._name = name
        self.setAccessibleName(name)
        self.setMaxLength(len(_MAX_SAFE_INTEGER_TEXT))
        self.setValidator(_SafeIntegerValidator(self))
        self.set_value(value)

    def set_value(self, value: int) -> None:
        """Set one already validated exact safe integer without float conversion."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{self._name} must be an integer")
        if not 0 <= value <= MAX_CANONICAL_SAFE_INTEGER:
            raise ValueError(f"{self._name} must lie within cross-runtime safe range")
        self.setText(str(value))

    def value(self) -> int:
        """Return the exact integer or reject an incomplete editor value."""
        if not self.hasAcceptableInput():
            raise ValueError(f"{self._name} must lie within cross-runtime safe range")
        return int(self.text())


class MaterialEditor(QGroupBox):
    """Editable surface identity and full v1 material parameter collection."""

    def __init__(self, title: str, value: SurfaceMaterialDraft) -> None:
        super().__init__(title)
        self.surface_id = QLineEdit(value.surface_id)
        self.surface_id.setAccessibleName(f"{title} surface ID")
        self.surface_id.setToolTip(
            f"Stable identifier for {title.lower()}; included in validated readback."
        )
        self.fields: dict[str, QDoubleSpinBox] = {}
        layout = QFormLayout(self)
        layout.addRow("Surface ID", self.surface_id)
        for name, label, suffix, step, minimum, maximum in _MATERIAL_FIELDS:
            field = number_input(
                f"{title} {label}",
                getattr(value, name),
                NumberInputSpec(suffix, step, minimum, maximum, decimals=11),
            )
            self.fields[name] = field
            layout.addRow(label, field)

    def draft(self) -> SurfaceMaterialDraft:
        """Read the current widgets without applying separate UI validation."""
        values = {name: field.value() for name, field in self.fields.items()}
        return SurfaceMaterialDraft(self.surface_id.text(), **values)

    def set_draft(self, value: SurfaceMaterialDraft) -> None:
        """Replace all visible material values from one validated draft."""
        self.surface_id.setText(value.surface_id)
        for name, field in self.fields.items():
            field.setValue(getattr(value, name))


class RegionalOverlayRow(QGroupBox):
    """One removable bounded regional overlay row."""

    def __init__(
        self,
        ordinal: int,
        value: RegionalOverlayDraft,
        remove: Callable[[RegionalOverlayRow], None],
    ) -> None:
        super().__init__(f"Regional overlay {ordinal}")
        self.region_id = QLineEdit(value.region_id)
        self.region_id.setToolTip(
            "Stable overlay identifier; it must be unique within the regional plan."
        )
        self.precedence = SafeIntegerEdit(
            f"Overlay {ordinal} precedence", value.precedence
        )
        self.precedence.setToolTip(
            "Overlay selection precedence. Higher values win when intervals overlap."
        )
        self.lower_coordinate = coordinate_input(
            f"Overlay {ordinal} lower coordinate",
            value.lower_coordinate_m,
        )
        self.upper_coordinate = coordinate_input(
            f"Overlay {ordinal} upper coordinate",
            value.upper_coordinate_m,
        )
        self.material = MaterialEditor(f"Overlay {ordinal} material", value.surface)
        self.remove_button = QPushButton(f"Remove overlay {ordinal}")
        self.remove_button.setToolTip(
            "Remove this overlay from the unvalidated draft; one overlay is required."
        )
        self.remove_button.clicked.connect(lambda: remove(self))
        form = QFormLayout()
        form.addRow("Region ID", self.region_id)
        form.addRow("Precedence", self.precedence)
        form.addRow("Lower coordinate", self.lower_coordinate)
        form.addRow("Upper coordinate", self.upper_coordinate)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.material)
        layout.addWidget(self.remove_button, alignment=Qt.AlignmentFlag.AlignRight)

    def draft(self) -> RegionalOverlayDraft:
        """Read one overlay for authoritative contract validation."""
        return RegionalOverlayDraft(
            self.region_id.text(),
            self.precedence.value(),
            self.lower_coordinate.value(),
            self.upper_coordinate.value(),
            self.material.draft(),
        )


__all__ = [
    "MaterialEditor",
    "RegionalOverlayRow",
    "SafeIntegerEdit",
    "coordinate_input",
    "number_input",
]
