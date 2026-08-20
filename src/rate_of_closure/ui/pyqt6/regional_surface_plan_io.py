"""Native Open/Save As controller for the regional surface-plan editor."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PyQt6.QtWidgets import QFileDialog, QLabel, QPlainTextEdit, QWidget

from rate_of_closure.application.regional_surface_plan_files import (
    read_regional_surface_plan_request,
    write_regional_surface_plan_request_atomic,
)
from shared.python.swing_sim.ground.regional_plan_records import (
    GroundRegionalMaterialPlanRequest,
)


class RegionalSurfacePlanFileHost(Protocol):
    """Minimal host surface needed by the native file controller."""

    status_label: QLabel
    readback: QPlainTextEdit

    def current_request(self) -> GroundRegionalMaterialPlanRequest: ...

    def apply_imported_request(
        self, request: GroundRegionalMaterialPlanRequest
    ) -> None: ...


class RegionalSurfacePlanFileActions:
    """Transactional native file commands with one successful recent path."""

    def __init__(self, host: RegionalSurfacePlanFileHost, parent: QWidget) -> None:
        self._host = host
        self._parent = parent
        self.recent_path: Path | None = None

    def open(self) -> None:
        """Choose, fully validate, and only then replace the editor draft."""
        selected, _filter = QFileDialog.getOpenFileName(
            self._parent,
            "Open Regional Surface Plan",
            self._initial_location(),
            "JSON files (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        try:
            request = read_regional_surface_plan_request(path)
            self._host.apply_imported_request(request)
        except (OSError, TypeError, ValueError) as exc:
            self._set_error(f"Open failed: {exc}")
            return
        self.recent_path = path
        self._host.readback.setPlainText(request.to_json())
        self._set_success(f"Opened {path.name}. No physics executed.")

    def save_as(self) -> None:
        """Validate before choosing a destination, then atomically replace it."""
        try:
            request = self._host.current_request()
        except (TypeError, ValueError) as exc:
            self._set_error(f"Save failed: {exc}")
            return
        selected, _filter = QFileDialog.getSaveFileName(
            self._parent,
            "Save Regional Surface Plan As",
            self._initial_location("regional-surface-plan.json"),
            "JSON files (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        try:
            write_regional_surface_plan_request_atomic(request, path)
        except OSError as exc:
            self._set_error(f"Save failed: {exc}")
            return
        self.recent_path = path
        self._host.readback.setPlainText(request.to_json())
        self._set_success(f"Saved {path.name} atomically. No physics executed.")

    def _initial_location(self, filename: str = "") -> str:
        if self.recent_path is None:
            return filename
        parent = self.recent_path.parent
        return str(parent / filename) if filename else str(parent)

    def _set_error(self, message: str) -> None:
        self._host.status_label.setText(message)
        self._host.status_label.setAccessibleName("Regional surface plan file error")

    def _set_success(self, message: str) -> None:
        self._host.status_label.setText(message)
        self._host.status_label.setAccessibleName("Regional surface plan file success")


__all__ = ["QFileDialog", "RegionalSurfacePlanFileActions"]
