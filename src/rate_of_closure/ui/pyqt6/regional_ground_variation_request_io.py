"""Native File commands for one combined regional-ground variation request."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PyQt6.QtWidgets import QFileDialog, QWidget

from rate_of_closure.application.regional_ground_variation_request import (
    read_regional_ground_variation_request,
    regional_ground_variation_request_to_json,
    write_regional_ground_variation_request_atomic,
)
from rate_of_closure.variation.regional_ground_variation import (
    GroundRegionalVariationRequest,
)


class RegionalGroundVariationRequestFileHost(Protocol):
    """Narrow application-shell boundary required by native File commands."""

    def current_regional_ground_variation_request(
        self,
    ) -> GroundRegionalVariationRequest: ...

    def apply_regional_ground_variation_request(
        self, request: GroundRegionalVariationRequest
    ) -> None: ...

    def show_regional_ground_variation_file_status(
        self, message: str, *, error: bool
    ) -> None: ...


class RegionalGroundVariationRequestFileActions:
    """Cancel-safe native dialogs over the strict application persistence port."""

    def __init__(
        self, host: RegionalGroundVariationRequestFileHost, parent: QWidget
    ) -> None:
        self._host = host
        self._parent = parent
        self.recent_path: Path | None = None

    def open(self) -> None:
        """Fully validate a selected request before applying it to the owner."""
        selected, _filter = QFileDialog.getOpenFileName(
            self._parent,
            "Open Regional-Ground Variation Request",
            self._initial_location(),
            "JSON files (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        try:
            request = read_regional_ground_variation_request(path)
            self._host.apply_regional_ground_variation_request(request)
        except (OSError, TypeError, ValueError) as exc:
            self._show(f"Open failed: {exc}", error=True)
            return
        self.recent_path = path
        self._show(f"Opened {path.name}. No physics executed.", error=False)

    def save_as(self) -> None:
        """Validate the snapshot before choosing and atomically replacing a file."""
        try:
            request = self._host.current_regional_ground_variation_request()
        except (TypeError, ValueError) as exc:
            self._show(f"Save failed: {exc}", error=True)
            return
        selected, _filter = QFileDialog.getSaveFileName(
            self._parent,
            "Save Regional-Ground Variation Request As",
            self._initial_location("regional-ground-variation-request.json"),
            "JSON files (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        try:
            write_regional_ground_variation_request_atomic(request, path)
        except (OSError, TypeError, ValueError) as exc:
            self._show(f"Save failed: {exc}", error=True)
            return
        self.recent_path = path
        self._show(f"Saved {path.name} atomically. No physics executed.", error=False)

    def _initial_location(self, filename: str = "") -> str:
        if self.recent_path is None:
            return filename
        parent = self.recent_path.parent
        return str(parent / filename) if filename else str(parent)

    def _show(self, message: str, *, error: bool) -> None:
        self._host.show_regional_ground_variation_file_status(message, error=error)


__all__ = [
    "QFileDialog",
    "RegionalGroundVariationRequestFileActions",
    "regional_ground_variation_request_to_json",
]
