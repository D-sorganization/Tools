"""Narrow public accessors for the spatial-target editor."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PyQt6.QtWidgets import QComboBox, QLabel, QLineEdit

from rate_of_closure.ui.pyqt6.spatial_target_panel_text import DEFAULT_GROUND_SOURCE
from shared.python.swing_sim.solver import (
    BoxTolerance,
    SpatialTarget,
    SphereTolerance,
    SurfaceCircleTolerance,
    spatial_target_from_json,
    spatial_target_to_json,
)


class SpatialTargetPanelAccessMixin:
    """Expose labeled controls without leaking layout internals."""

    _coordinate_edits: dict[str, QLineEdit]
    _coordinate_labels: dict[str, QLabel]
    _frame_combo: QComboBox
    _ground_edit: QLineEdit
    _kind_combo: QComboBox
    _label_edit: QLineEdit
    _last_frame: str
    _last_target: SpatialTarget
    _loading: bool
    _miss: QLabel
    _external_error: str | None
    _summary: QLabel
    _tolerance_combo: QComboBox
    _tolerance_edits: dict[str, QLineEdit]
    _validation: QLabel
    _valid: bool

    if TYPE_CHECKING:

        def _configure_tolerances(self, kind: str) -> None: ...

        def _set_error(self, message: str) -> None: ...

        def _sync_labels(self) -> None: ...

        def _validate_and_emit(self, *, emit: bool = True) -> None: ...

    def target(self) -> SpatialTarget:
        """Return the target described by valid entries or raise explicitly."""
        if not self._valid:
            raise ValueError(self._validation.text().removeprefix("Invalid target: "))
        return self._last_target

    def current_target(self) -> SpatialTarget:
        """Return the last valid target while an edit is incomplete or invalid."""
        return self._last_target

    def is_valid(self) -> bool:
        """Whether every visible field currently satisfies the target contract."""
        return self._valid

    def target_json(self) -> str:
        """Serialize the valid target with the shared versioned contract."""
        return cast(str, spatial_target_to_json(self.target()))

    @staticmethod
    def serialize_target(target: SpatialTarget) -> str:
        """Serialize ``target`` through the shared deterministic contract."""
        return cast(str, spatial_target_to_json(target))

    def load_target_json(self, text: str) -> None:
        """Load serialized target text, preserving the prior target on failure."""
        try:
            target = spatial_target_from_json(text)
        except (TypeError, ValueError) as exc:
            self._external_error = f"Could not load target JSON: {exc}"
            self._set_error(self._external_error)
            return
        self.set_target(target)

    def set_target(self, target: SpatialTarget, *, emit: bool = True) -> None:
        """Populate every editor from an already validated canonical target."""
        if not isinstance(target, SpatialTarget):
            raise TypeError("target must be a SpatialTarget")
        self._loading = True
        self._external_error = None
        self._label_edit.setText(target.label)
        self._set_combo_data(self._kind_combo, target.kind)
        self._configure_tolerances(target.kind)
        self._set_combo_data(self._frame_combo, target.point.source_frame)
        self._last_frame = target.point.source_frame
        self._populate_coordinates(
            target.point.coordinates_in(target.point.source_frame)
        )
        self._ground_edit.setText(target.ground_source or DEFAULT_GROUND_SOURCE)
        self._populate_tolerance(target)
        self._sync_labels()
        self._loading = False
        self._validate_and_emit(emit=emit)

    @staticmethod
    def _set_combo_data(combo: QComboBox, data: str) -> None:
        index = combo.findData(data)
        if index < 0:
            raise ValueError(f"unsupported combo value {data!r}")
        combo.setCurrentIndex(index)

    def _populate_coordinates(self, values: tuple[float, float, float]) -> None:
        for key, value in zip(("x", "second", "third"), values, strict=True):
            self._coordinate_edits[key].setText(f"{value:.12g}")

    def _populate_tolerance(self, target: SpatialTarget) -> None:
        tolerance = target.tolerance
        kind: str
        values: tuple[float, ...]
        if isinstance(tolerance, (SphereTolerance, SurfaceCircleTolerance)):
            kind = (
                "sphere" if isinstance(tolerance, SphereTolerance) else "surface_circle"
            )
            values = (tolerance.radius_m,)
        elif isinstance(tolerance, BoxTolerance):
            kind, values = "box", tolerance.half_extents_m
        else:
            kind = "surface_corridor"
            values = (tolerance.half_length_m, tolerance.half_width_m)
        self._set_combo_data(self._tolerance_combo, kind)
        for key, value in zip(
            ("primary", "secondary", "tertiary"), values, strict=False
        ):
            self._tolerance_edits[key].setText(f"{value:.12g}")

    def coordinate_edit(self, key: str) -> QLineEdit:
        """Return one coordinate editor."""
        return self._coordinate_edits[key]

    def coordinate_label(self, key: str) -> QLabel:
        """Return one frame-dependent coordinate label."""
        return self._coordinate_labels[key]

    def tolerance_edit(self, key: str) -> QLineEdit:
        """Return one tolerance editor."""
        return self._tolerance_edits[key]

    def kind_combo(self) -> QComboBox:
        """Return the target-kind selector."""
        return self._kind_combo

    def frame_combo(self) -> QComboBox:
        """Return the authoring-frame selector."""
        return self._frame_combo

    def tolerance_combo(self) -> QComboBox:
        """Return the tolerance-geometry selector."""
        return self._tolerance_combo

    def summary_label(self) -> QLabel:
        """Return the current-target summary."""
        return self._summary

    def validation_label(self) -> QLabel:
        """Return the visible validation status."""
        return self._validation

    def miss_label(self) -> QLabel:
        """Return the visible signed residual summary."""
        return self._miss


__all__ = ["SpatialTargetPanelAccessMixin"]
