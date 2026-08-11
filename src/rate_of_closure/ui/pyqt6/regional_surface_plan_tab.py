"""PyQt6 editor/readback for strict regional surface-plan requests."""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.regional_surface_plan import (
    MAX_EDITOR_REGIONS,
    RegionalOverlayDraft,
    RegionalSurfacePlanDraft,
    editor_draft_from_regional_surface_plan_request,
    illustrative_regional_surface_plan_draft,
    regional_surface_plan_request_for_draft,
)
from rate_of_closure.ui.pyqt6.regional_surface_plan_io import (
    RegionalSurfacePlanFileActions,
)
from rate_of_closure.ui.pyqt6.regional_surface_plan_widgets import (
    MaterialEditor,
    RegionalOverlayRow,
    number_input,
)
from shared.python.swing_sim.ground.regional_plan_records import (
    GroundRegionalMaterialPlanRequest,
)


class RegionalSurfacePlanTab(QWidget):
    """Session-only regional surface editor with strict canonical readback."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[RegionalOverlayRow] = []
        self._initial = illustrative_regional_surface_plan_draft()
        self._imported_request: GroundRegionalMaterialPlanRequest | None = None
        self.file_actions = RegionalSurfacePlanFileActions(self, self)
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
            "Open/Save As persists this canonical request only. Workspace "
            "persistence remains a separate contract."
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
        self.domain_lower = number_input(
            "Base domain lower coordinate", self._initial.lower_coordinate_m, " m"
        )
        self.domain_upper = number_input(
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
        self.open_button = QPushButton("Open JSON")
        self.open_button.setAccessibleName("Open regional surface plan JSON")
        self.open_button.setToolTip(
            "Open and fully validate an editor-qualified canonical request."
        )
        self.open_button.clicked.connect(self.file_actions.open)
        self.save_button = QPushButton("Save As JSON")
        self.save_button.setAccessibleName("Save regional surface plan JSON as")
        self.save_button.setToolTip(
            "Atomically save the validated canonical request to a chosen file."
        )
        self.save_button.clicked.connect(self.file_actions.save_as)
        layout.addWidget(self.add_button)
        layout.addWidget(self.open_button)
        layout.addWidget(self.save_button)
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

    def current_request(self) -> GroundRegionalMaterialPlanRequest:
        """Return exact imported evidence unless the visible draft has changed."""
        return regional_surface_plan_request_for_draft(
            self.draft(), self._imported_request
        )

    def apply_imported_request(
        self, request: GroundRegionalMaterialPlanRequest
    ) -> None:
        """Populate widgets only after complete strict editor qualification."""
        draft = editor_draft_from_regional_surface_plan_request(request)
        self.request_id.setText(draft.request_id)
        self.source_revision.setText(draft.source_revision)
        self.domain_lower.setValue(draft.lower_coordinate_m)
        self.domain_upper.setValue(draft.upper_coordinate_m)
        self.base_material.set_draft(draft.base_surface)
        for row in self._rows:
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        for region in draft.regions:
            self._append_row(region)
        self._update_row_actions()
        self._imported_request = request

    def validate_plan(self) -> None:
        """Validate through the strict contract and render canonical readback."""
        try:
            request = self.current_request()
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
