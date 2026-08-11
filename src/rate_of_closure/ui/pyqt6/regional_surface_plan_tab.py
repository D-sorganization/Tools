"""PyQt6 editor/readback for strict regional surface-plan requests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.regional_surface_plan import (
    MAX_EDITOR_REGIONS,
    RegionalOverlayDraft,
    RegionalSurfacePlanDraft,
    SurfaceMaterialDraft,
    illustrative_regional_surface_plan_draft,
    validate_regional_surface_plan_draft,
)

_MATERIAL_FIELDS = (
    ("normal_restitution", "Normal restitution", "", 0.01, 0.0, 1.0),
    ("static_friction", "Static friction", "", 0.01, 0.0, 5.0),
    ("kinetic_friction", "Kinetic friction", "", 0.01, 0.0, 5.0),
    ("rolling_resistance", "Rolling resistance", "", 0.01, 0.0, 1.0),
    ("firmness_pa", "Firmness", " Pa", 1_000.0, 0.001, 1e9),
    ("hardness_fraction", "Hardness", " fraction", 0.01, 0.0, 1.0),
    ("grass_height_m", "Grass height", " m", 0.001, 0.0, 1.0),
    ("compressibility_fraction", "Compressibility", " fraction", 0.01, 0.0, 1.0),
    (
        "compression_damping_fraction",
        "Compression damping",
        " fraction",
        0.01,
        0.0,
        1.0,
    ),
    ("turf_density_kg_m3", "Turf density", " kg/m³", 1.0, 0.0, 10_000.0),
    ("moisture_fraction", "Moisture", " fraction", 0.01, 0.0, 1.0),
)


def _number_input(
    name: str,
    value: float,
    suffix: str = "",
    step: float = 0.1,
    minimum: float = -1e9,
    maximum: float = 1e9,
) -> QDoubleSpinBox:
    """Create one consistently configured accessible SI number input."""
    field = QDoubleSpinBox()
    field.setAccessibleName(name)
    field.setDecimals(6)
    field.setRange(minimum, maximum)
    field.setSingleStep(step)
    field.setSuffix(suffix)
    field.setValue(value)
    field.setToolTip(
        f"{name}. Edit this SI draft value, then validate the surface plan."
    )
    return field


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
            field = _number_input(
                f"{title} {label}", getattr(value, name), suffix, step, minimum, maximum
            )
            self.fields[name] = field
            layout.addRow(label, field)

    def draft(self) -> SurfaceMaterialDraft:
        """Read the current widgets without applying separate UI validation."""
        values = {name: field.value() for name, field in self.fields.items()}
        return SurfaceMaterialDraft(self.surface_id.text(), **values)


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
        self.precedence = QSpinBox()
        self.precedence.setRange(0, 1_000_000)
        self.precedence.setValue(value.precedence)
        self.precedence.setToolTip(
            "Overlay selection precedence. Higher values win when intervals overlap."
        )
        self.lower_coordinate = _number_input(
            f"Overlay {ordinal} lower coordinate", value.lower_coordinate_m, " m"
        )
        self.upper_coordinate = _number_input(
            f"Overlay {ordinal} upper coordinate", value.upper_coordinate_m, " m"
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


class RegionalSurfacePlanTab(QWidget):
    """Session-only regional surface editor with strict canonical readback."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[RegionalOverlayRow] = []
        self._initial = illustrative_regional_surface_plan_draft()
        self._build_ui()
        self._connect_static_changes()
        self._append_row(self._initial.regions[0])
        self._update_row_actions()

    def _connect_material_changes(self, editor: MaterialEditor) -> None:
        """Invalidate canonical output when any material draft value changes."""
        editor.surface_id.textChanged.connect(self._mark_dirty)
        for field in editor.fields.values():
            field.valueChanged.connect(self._mark_dirty)

    def _connect_static_changes(self) -> None:
        """Connect the fixed identity, domain, and base-material inputs."""
        self.request_id.textChanged.connect(self._mark_dirty)
        self.source_revision.textChanged.connect(self._mark_dirty)
        self.domain_lower.valueChanged.connect(self._mark_dirty)
        self.domain_upper.valueChanged.connect(self._mark_dirty)
        self._connect_material_changes(self.base_material)

    def _connect_row_changes(self, row: RegionalOverlayRow) -> None:
        """Connect one dynamic overlay to the shared invalidation boundary."""
        row.region_id.textChanged.connect(self._mark_dirty)
        row.precedence.valueChanged.connect(self._mark_dirty)
        row.lower_coordinate.valueChanged.connect(self._mark_dirty)
        row.upper_coordinate.valueChanged.connect(self._mark_dirty)
        self._connect_material_changes(row.material)

    def _mark_dirty(self) -> None:
        """Remove stale validation evidence after any draft mutation."""
        self.status_label.setText("Changes not validated")
        self.status_label.setAccessibleName("Regional surface plan validation pending")
        self.readback.clear()

    def _build_ui(self) -> None:
        """Build the scrollable form and always-visible validation output."""
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        title = QLabel("Regional Surface Plan")
        title.setObjectName("sectionTitle")
        self.warning_label = QLabel(
            "Illustrative, unvalidated values are loaded for discovery. They are "
            "not measured course data. This slice validates only; it does not run "
            "physics."
        )
        self.warning_label.setWordWrap(True)
        self.warning_label.setAccessibleName("Regional surface qualification")
        self.content_layout.addWidget(title)
        self.content_layout.addWidget(self.warning_label)
        self.content_layout.addWidget(self._identity_box())
        self.content_layout.addWidget(self._base_box())
        self.rows_layout = QVBoxLayout()
        self.content_layout.addLayout(self.rows_layout)
        self.content_layout.addLayout(self._action_row())
        self.status_label = QLabel("Not validated")
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Regional surface plan validation status")
        self.content_layout.addWidget(self.status_label)
        self.readback = QPlainTextEdit()
        self.readback.setReadOnly(True)
        self.readback.setAccessibleName("Regional surface plan canonical readback")
        self.readback.setPlaceholderText("Validated canonical request appears here.")
        self.content_layout.addWidget(self.readback)
        self.content_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

    def _identity_box(self) -> QGroupBox:
        """Create request identity, evidence, and fixed calibration controls."""
        box = QGroupBox("Plan identity and provenance")
        self.request_id = QLineEdit(self._initial.request_id)
        self.request_id.setToolTip(
            "Stable request identifier included in canonical validation output."
        )
        self.source_revision = QLineEdit(self._initial.source_revision)
        self.source_revision.setToolTip(
            "Revision of the source evidence used to define this draft."
        )
        self.calibration_combo = QComboBox()
        self.calibration_combo.addItem("Unvalidated", "unvalidated")
        self.calibration_combo.setEnabled(False)
        self.calibration_combo.setToolTip(
            "Calibration remains unvalidated until measured course evidence exists."
        )
        geometry = QLabel(
            "Frame: target x-downrange, y-up, z-right. Flat static coplanar geometry."
        )
        geometry.setWordWrap(True)
        session = QLabel(
            "Session-only draft: current workspace file commands do not persist "
            "model inputs."
        )
        session.setWordWrap(True)
        layout = QFormLayout(box)
        layout.addRow("Request ID", self.request_id)
        layout.addRow("Source revision", self.source_revision)
        layout.addRow("Calibration", self.calibration_combo)
        layout.addRow("Qualified geometry", geometry)
        layout.addRow("Persistence", session)
        return box

    def _base_box(self) -> QGroupBox:
        """Create the base interval and complete base material editor."""
        box = QGroupBox("Base surface and domain")
        self.domain_lower = _number_input(
            "Base domain lower coordinate", self._initial.lower_coordinate_m, " m"
        )
        self.domain_upper = _number_input(
            "Base domain upper coordinate", self._initial.upper_coordinate_m, " m"
        )
        self.base_material = MaterialEditor("Base material", self._initial.base_surface)
        layout = QVBoxLayout(box)
        form = QFormLayout()
        form.addRow("Lower coordinate", self.domain_lower)
        form.addRow("Upper coordinate", self.domain_upper)
        layout.addLayout(form)
        layout.addWidget(self.base_material)
        return box

    def _action_row(self) -> QHBoxLayout:
        """Create bounded row and strict validation actions."""
        layout = QHBoxLayout()
        self.add_button = QPushButton("Add overlay")
        self.add_button.setAccessibleName("Add regional overlay")
        self.add_button.setToolTip(
            f"Add an illustrative overlay row, up to {MAX_EDITOR_REGIONS} total."
        )
        self.add_button.clicked.connect(self._add_default_row)
        self.validate_button = QPushButton("Validate and preview")
        self.validate_button.setAccessibleName("Validate surface plan")
        self.validate_button.setToolTip(
            "Validate the complete draft and display its canonical SI request."
        )
        self.validate_button.clicked.connect(self.validate_plan)
        layout.addWidget(self.add_button)
        layout.addStretch(1)
        layout.addWidget(self.validate_button)
        return layout

    def _add_default_row(self) -> None:
        """Append one unique in-domain illustrative row if capacity remains."""
        if len(self._rows) >= MAX_EDITOR_REGIONS:
            return
        ordinal = len(self._rows) + 1
        template = self._initial.regions[0]
        lower = min(280.0, 120.0 + (ordinal - 1) * 20.0)
        surface = replace(
            template.surface, surface_id=f"illustrative-surface-{ordinal}"
        )
        self._append_row(
            RegionalOverlayDraft(
                f"illustrative-region-{ordinal}",
                ordinal * 10,
                lower,
                min(295.0, lower + 15.0),
                surface,
            )
        )
        self._update_row_actions()
        self._mark_dirty()

    def _append_row(self, draft: RegionalOverlayDraft) -> None:
        """Create and register a presentation row."""
        row = RegionalOverlayRow(len(self._rows) + 1, draft, self._remove_row)
        self._connect_row_changes(row)
        self._rows.append(row)
        self.rows_layout.addWidget(row)

    def _remove_row(self, row: RegionalOverlayRow) -> None:
        """Remove a row while preserving the contract's nonempty invariant."""
        if len(self._rows) <= 1:
            return
        self._rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.deleteLater()
        self._update_row_actions()
        self._mark_dirty()

    def _update_row_actions(self) -> None:
        """Expose row bounds through disabled states and removal availability."""
        self.add_button.setEnabled(len(self._rows) < MAX_EDITOR_REGIONS)
        removable = len(self._rows) > 1
        for row in self._rows:
            row.remove_button.setEnabled(removable)

    def region_count(self) -> int:
        """Return the current bounded number of overlay rows."""
        return len(self._rows)

    def region_rows(self) -> tuple[RegionalOverlayRow, ...]:
        """Return a stable snapshot for integration and accessibility tests."""
        return tuple(self._rows)

    def draft(self) -> RegionalSurfacePlanDraft:
        """Read the current presentation state into a UI-neutral draft."""
        return RegionalSurfacePlanDraft(
            self.request_id.text(),
            self.domain_lower.value(),
            self.domain_upper.value(),
            self.source_revision.text(),
            str(self.calibration_combo.currentData()),
            self.base_material.draft(),
            tuple(row.draft() for row in self._rows),
        )

    def validate_plan(self) -> None:
        """Validate through the strict contract and render canonical readback."""
        try:
            request = validate_regional_surface_plan_draft(self.draft())
        except (TypeError, ValueError) as exc:
            self.status_label.setText(str(exc))
            self.status_label.setAccessibleName(
                "Regional surface plan validation error"
            )
            self.readback.clear()
            return
        self.status_label.setText(
            f"Validated {len(request.regions)} overlay(s) in SI. No physics executed."
        )
        self.status_label.setAccessibleName("Regional surface plan validation success")
        self.readback.setPlainText(request.to_json())


__all__ = ["RegionalSurfacePlanTab"]
