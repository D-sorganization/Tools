"""Accessible editor for the canonical versioned three-dimensional target."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.responsive_layout import HeightForWidthGroupBox
from rate_of_closure.ui.pyqt6.spatial_target_panel_access import (
    SpatialTargetPanelAccessMixin,
)
from rate_of_closure.ui.pyqt6.spatial_target_panel_text import (
    COORDINATE_LABELS,
    DEFAULT_GROUND_SOURCE,
    FRAME_ITEMS,
    KIND_ITEMS,
    TOLERANCE_ITEMS,
    finite_number,
    miss_summary,
    target_summary,
)
from rate_of_closure.ui.pyqt6.spatial_target_trajectory import (
    validate_landing_surface,
)
from shared.python.swing_sim.solver import (
    BoxTolerance,
    SpatialTarget,
    SphereTolerance,
    SurfaceCircleTolerance,
    SurfaceCorridorTolerance,
    TargetMiss,
    TargetPoint,
)


class SpatialTargetPanel(SpatialTargetPanelAccessMixin, HeightForWidthGroupBox):
    """Edit one canonical target without silently coercing invalid text."""

    targetChanged = pyqtSignal(object)  # noqa: N815 - Qt signal convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Spatial Target", parent)
        self.setAccessibleName("Spatial Target Editor")
        self._loading = True
        self._external_error: str | None = None
        self._valid = False
        self._last_frame = "app"
        self._last_target = self._default_target()
        self._coordinate_edits: dict[str, QLineEdit] = {}
        self._coordinate_labels: dict[str, QLabel] = {}
        self._tolerance_edits: dict[str, QLineEdit] = {}
        self._tolerance_labels: dict[str, QLabel] = {}
        self._build_ui()
        self.set_target(self._last_target, emit=False)
        self._loading = False
        self._validate_and_emit(emit=False)

    @staticmethod
    def _default_target() -> SpatialTarget:
        return SpatialTarget(
            label="Landing target",
            kind="landing_area",
            point=TargetPoint(230.0, 0.0, 0.0),
            tolerance=SurfaceCircleTolerance(10.0),
            elevation_source="course_surface",
            ground_source=DEFAULT_GROUND_SOURCE,
        )

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._label_edit = QLineEdit()
        self._label_edit.setAccessibleName("Spatial Target Label")
        self._label_edit.setToolTip(
            "Enter a descriptive target name used in summaries and exported JSON."
        )
        form.addRow("Target label", self._label_edit)
        self._kind_combo = self._combo(KIND_ITEMS, "Spatial Target Kind")
        self._kind_combo.setToolTip(
            "Choose a landing target on the course surface or a 3D aerial waypoint."
        )
        form.addRow("Target kind", self._kind_combo)
        self._frame_combo = self._combo(FRAME_ITEMS, "Target Authoring Frame")
        self._frame_combo.setToolTip(
            "Choose the coordinate convention for entry; changing it preserves "
            "the same canonical physical point."
        )
        form.addRow("Entry frame", self._frame_combo)
        for key in ("x", "second", "third"):
            label = QLabel()
            edit = QLineEdit()
            edit.setToolTip(
                "Enter a finite coordinate in metres using the selected entry frame."
            )
            self._coordinate_labels[key] = label
            self._coordinate_edits[key] = edit
            form.addRow(label, edit)
        self._ground_label = QLabel("Ground source")
        self._ground_edit = QLineEdit(DEFAULT_GROUND_SOURCE)
        self._ground_edit.setAccessibleName("Course Surface Source")
        self._ground_edit.setToolTip(
            "Identify the course-surface elevation source for this landing target."
        )
        form.addRow(self._ground_label, self._ground_edit)
        self._tolerance_combo = QComboBox()
        self._tolerance_combo.setAccessibleName("Spatial Target Tolerance Geometry")
        self._tolerance_combo.setToolTip(
            "Choose the accepted target region: circle/corridor on the surface "
            "or sphere/box in 3D."
        )
        form.addRow("Tolerance", self._tolerance_combo)
        for key in ("primary", "secondary", "tertiary"):
            label = QLabel()
            edit = QLineEdit()
            edit.setToolTip(
                "Enter a positive tolerance dimension in metres when this field "
                "is used."
            )
            self._tolerance_labels[key] = label
            self._tolerance_edits[key] = edit
            form.addRow(label, edit)
        layout.addLayout(form)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setAccessibleName("Current Spatial Target")
        layout.addWidget(self._summary)
        self._validation = QLabel()
        self._validation.setWordWrap(True)
        self._validation.setAccessibleName("Spatial Target Validation")
        layout.addWidget(self._validation)
        self._miss = QLabel("Run a flight to calculate target residuals.")
        self._miss.setWordWrap(True)
        self._miss.setAccessibleName("Spatial Target Miss Residuals")
        layout.addWidget(self._miss)
        buttons = QHBoxLayout()
        self._copy_button = QPushButton("Copy Target JSON")
        self._copy_button.setAccessibleName("Copy Spatial Target JSON")
        self._copy_button.setToolTip(
            "Copy the validated, versioned spatial-target JSON to the clipboard."
        )
        self._paste_button = QPushButton("Paste Target JSON")
        self._paste_button.setAccessibleName("Paste Spatial Target JSON")
        self._paste_button.setToolTip(
            "Load versioned spatial-target JSON from the clipboard; invalid input "
            "leaves the last valid target unchanged."
        )
        buttons.addWidget(self._copy_button)
        buttons.addWidget(self._paste_button)
        layout.addLayout(buttons)
        self._connect_controls()

    @staticmethod
    def _combo(items: tuple[tuple[str, str], ...], accessible_name: str) -> QComboBox:
        combo = QComboBox()
        for label, data in items:
            combo.addItem(label, data)
        combo.setAccessibleName(accessible_name)
        return combo

    def _connect_controls(self) -> None:
        self._kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        self._frame_combo.currentIndexChanged.connect(self._on_frame_changed)
        self._tolerance_combo.currentIndexChanged.connect(self._on_tolerance_changed)
        self._label_edit.textChanged.connect(self._on_control_edited)
        self._ground_edit.textChanged.connect(self._on_control_edited)
        for edit in (*self._coordinate_edits.values(), *self._tolerance_edits.values()):
            edit.textChanged.connect(self._on_control_edited)
        self._copy_button.clicked.connect(self._copy_json)
        self._paste_button.clicked.connect(self._paste_json)

    def _on_control_edited(self, *_args: object) -> None:
        if self._loading:
            return
        self._external_error = None
        self._miss.setText("Target changed — run a flight to refresh residuals.")
        self._validate_and_emit()

    def _on_frame_changed(self, *_args: object) -> None:
        if self._loading:
            return
        if not self._valid:
            self._loading = True
            self._set_combo_data(self._frame_combo, self._last_frame)
            self._loading = False
            self._set_error("Correct invalid entries before changing entry frame")
            return
        new_frame = str(self._frame_combo.currentData())
        coordinates = self._last_target.point.coordinates_in(new_frame)
        self._loading = True
        self._populate_coordinates(coordinates)
        self._loading = False
        self._last_frame = new_frame
        self._sync_labels()
        self._validate_and_emit()

    def _on_kind_changed(self, *_args: object) -> None:
        if self._loading:
            return
        self._configure_tolerances(str(self._kind_combo.currentData()))
        self._sync_labels()
        self._validate_and_emit()

    def _on_tolerance_changed(self, *_args: object) -> None:
        if self._loading:
            return
        self._sync_labels()
        self._validate_and_emit()

    def _configure_tolerances(self, kind: str) -> None:
        was_loading = self._loading
        self._loading = True
        self._tolerance_combo.clear()
        for label, data in TOLERANCE_ITEMS[kind]:
            self._tolerance_combo.addItem(label, data)
        defaults = (
            ("10", "10", "10") if kind == "aerial_waypoint" else ("10", "16", "10")
        )
        for edit, value in zip(self._tolerance_edits.values(), defaults, strict=True):
            edit.setText(value)
        self._loading = was_loading

    def _sync_labels(self) -> None:
        frame = str(self._frame_combo.currentData())
        for key, text in zip(
            self._coordinate_labels, COORDINATE_LABELS[frame], strict=True
        ):
            self._coordinate_labels[key].setText(text)
            self._coordinate_edits[key].setAccessibleName(text.removesuffix(" [m]"))
            self._coordinate_edits[key].setToolTip(
                f"Enter {text}; signed direction follows the selected entry frame."
            )
        landing = self._kind_combo.currentData() == "landing_area"
        self._ground_label.setVisible(landing)
        self._ground_edit.setVisible(landing)
        tolerance = self._tolerance_combo.currentData()
        labels = {
            "surface_circle": ("Radius [m]",),
            "surface_corridor": ("Half-length [m]", "Half-width [m]"),
            "sphere": ("Radius [m]",),
            "box": ("Half downrange [m]", "Half elevation [m]", "Half right [m]"),
        }[tolerance]
        for index, key in enumerate(self._tolerance_labels):
            visible = index < len(labels)
            self._tolerance_labels[key].setVisible(visible)
            self._tolerance_edits[key].setVisible(visible)
            if visible:
                self._tolerance_labels[key].setText(labels[index])
                self._tolerance_edits[key].setAccessibleName(
                    labels[index].removesuffix(" [m]")
                )
                self._tolerance_edits[key].setToolTip(
                    f"Enter a positive {labels[index].lower()} for the accepted region."
                )

    def _build_target(self) -> SpatialTarget:
        frame = str(self._frame_combo.currentData())
        coordinate_labels = COORDINATE_LABELS[frame]
        coordinates = tuple(
            finite_number(edit, label.removesuffix(" [m]"))
            for edit, label in zip(
                self._coordinate_edits.values(), coordinate_labels, strict=True
            )
        )
        point = TargetPoint.from_frame(coordinates, source_frame=frame)
        tolerance = self._build_tolerance()
        kind = str(self._kind_combo.currentData())
        ground_source = self._ground_edit.text()
        return SpatialTarget(
            label=self._label_edit.text(),
            kind=kind,
            point=point,
            tolerance=tolerance,
            elevation_source="course_surface" if kind == "landing_area" else "absolute",
            ground_source=ground_source if kind == "landing_area" else None,
        )

    def _build_tolerance(self):  # type: ignore[no-untyped-def]
        kind = self._tolerance_combo.currentData()
        values = [
            finite_number(edit, label.text().removesuffix(" [m]"), positive=True)
            for label, edit in zip(
                self._tolerance_labels.values(),
                self._tolerance_edits.values(),
                strict=True,
            )
            if not edit.isHidden()
        ]
        if kind == "surface_circle":
            return SurfaceCircleTolerance(values[0])
        if kind == "surface_corridor":
            return SurfaceCorridorTolerance(*values)
        if kind == "sphere":
            return SphereTolerance(values[0])
        return BoxTolerance(tuple(values))

    def _validate_and_emit(self, *, emit: bool = True) -> None:
        if self._external_error is not None:
            self._set_error(self._external_error)
            return
        try:
            target = self._build_target()
            validate_landing_surface(target)
        except (TypeError, ValueError) as exc:
            self._set_error(str(exc))
            return
        self._valid = True
        self._last_target = target
        self._clear_errors()
        self._validation.setText(
            "Target valid — plot and residual calculation are current."
        )
        self._validation.setProperty("validationState", "valid")
        self._summary.setText(target_summary(target))
        self._copy_button.setEnabled(True)
        if emit:
            self.targetChanged.emit(target)

    def _set_error(self, message: str) -> None:
        self._valid = False
        self._validation.setText(
            f"Invalid target: {message}. Plot remains at the last valid target."
        )
        self._validation.setAccessibleDescription(self._validation.text())
        self._validation.setProperty("validationState", "error")
        self._copy_button.setEnabled(False)
        self._mark_matching_field(message)

    def _clear_errors(self) -> None:
        for edit in (
            self._label_edit,
            self._ground_edit,
            *self._coordinate_edits.values(),
            *self._tolerance_edits.values(),
        ):
            edit.setProperty("validationState", "valid")
            edit.setAccessibleDescription("Valid target input")

    def _mark_matching_field(self, message: str) -> None:
        self._clear_errors()
        for label, edit in zip(
            COORDINATE_LABELS[str(self._frame_combo.currentData())],
            self._coordinate_edits.values(),
            strict=True,
        ):
            if message.startswith(label.removesuffix(" [m]")):
                edit.setProperty("validationState", "error")
                edit.setAccessibleDescription(f"Invalid input: {message}")
                return

    def set_miss(self, miss: TargetMiss, *, landing: bool) -> None:
        """Display signed canonical miss components for the latest trajectory."""
        if not isinstance(miss, TargetMiss):
            raise TypeError("miss must be a TargetMiss")
        self._miss.setText(miss_summary(miss, landing=landing))

    def set_miss_unavailable(self, reason: str) -> None:
        """Explain why no residual exists instead of retaining stale output."""
        self._miss.setText(f"Target residual unavailable: {reason}")

    def _copy_json(self) -> None:
        if self._valid:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(self.target_json())

    def _paste_json(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is None:
            self._set_error("System clipboard is unavailable")
            return
        self.load_target_json(clipboard.text())


__all__ = ["SpatialTargetPanel"]
