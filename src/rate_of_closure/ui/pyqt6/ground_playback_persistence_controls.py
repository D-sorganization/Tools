"""File controls for the PyQt ground playback workspace."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QIODevice, QSaveFile
from PyQt6.QtWidgets import QFileDialog, QGroupBox, QPushButton, QVBoxLayout, QWidget

from rate_of_closure.simulation.ground_playback_workspace_v2 import (
    GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2,
)


def write_atomic_text(path: Path, text: str) -> None:
    """Replace one UTF-8 text file atomically or preserve its prior contents."""
    if not isinstance(path, Path):
        raise TypeError("export path must be a pathlib Path")
    if type(text) is not str:
        raise TypeError("export content must be text")
    output = QSaveFile(str(path))
    if not output.open(QIODevice.OpenModeFlag.WriteOnly):
        raise OSError(output.errorString())
    encoded = text.encode("utf-8")
    if output.write(encoded) != len(encoded) or not output.commit():
        raise OSError(output.errorString())


class GroundPlaybackPersistenceControls(QGroupBox):
    """Accessible import/export controls backed by injected document functions."""

    def __init__(
        self,
        parent: QWidget,
        *,
        import_workspace: Callable[[str, str], None],
        exports: dict[str, tuple[str, Callable[[], str]]],
        report_error: Callable[[str], None],
    ) -> None:
        super().__init__("Workspace & exports", parent)
        self._import_workspace = import_workspace
        self._exports = exports
        self._report_error = report_error
        layout = QVBoxLayout(self)
        self.import_workspace_button = QPushButton("Import Workspace JSON…")
        self.import_workspace_button.setAccessibleName(
            "Import ground playback workspace"
        )
        self.import_workspace_button.setToolTip(
            "Restore strict workspace v1 or v2 and retain every last-good "
            "state on error."
        )
        self.import_workspace_button.clicked.connect(self._choose_workspace)
        layout.addWidget(self.import_workspace_button)
        self.export_buttons: dict[str, QPushButton] = {}
        labels = {
            "workspace": "Save Workspace JSON…",
            "result": "Export Result JSON…",
            "trajectory": "Export Trajectory CSV…",
            "events": "Export Events CSV…",
        }
        accessible = {
            "workspace": "Save ground playback workspace",
            "result": "Export ground result JSON",
            "trajectory": "Export ground trajectory CSV",
            "events": "Export ground events CSV",
        }
        tooltips = {
            "workspace": (
                "Save primary, optional comparison, playback, and view as "
                "strict v2 JSON."
            ),
            "result": (
                "Export the loaded flight-to-ground-result/v1 record as canonical JSON."
            ),
            "trajectory": (
                "Export every trajectory sample and state-vector field as CSV."
            ),
            "events": "Export every event and pre/post state-vector field as CSV.",
        }
        for export_id, label in labels.items():
            button = QPushButton(label)
            button.setAccessibleName(accessible[export_id])
            button.setToolTip(tooltips[export_id])
            button.clicked.connect(
                lambda _checked=False, name=export_id: self._choose_export(name)
            )
            self.export_buttons[export_id] = button
            layout.addWidget(button)
        self.set_exports_enabled(False)

    def set_exports_enabled(self, enabled: bool) -> None:
        """Enable result-dependent operations together."""
        for button in self.export_buttons.values():
            button.setEnabled(enabled)

    def _choose_workspace(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Import Ground Playback Workspace", "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            file_path = Path(path)
            if file_path.stat().st_size > GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2:
                raise ValueError(
                    "ground playback workspace exceeds the import size limit"
                )
            self._import_workspace(
                file_path.read_text(encoding="utf-8"), file_path.name
            )
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            self._report_error(f"Could not import {Path(path).name}: {exc}")

    def _choose_export(self, export_id: str) -> None:
        default_name, render = self._exports[export_id]
        suffix = Path(default_name).suffix
        file_filter = (
            "JSON files (*.json)" if suffix == ".json" else "CSV files (*.csv)"
        )
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export Ground Playback Data", default_name, file_filter
        )
        if not path:
            return
        try:
            write_atomic_text(Path(path), render())
        except (OSError, TypeError, ValueError) as exc:
            self._report_error(f"Could not export {Path(path).name}: {exc}")


__all__ = ["GroundPlaybackPersistenceControls", "write_atomic_text"]
