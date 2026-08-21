"""Restricted-data export interaction for the player workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from rate_of_closure.launch_monitor_workspace import export_analysis_bundle
from rate_of_closure.launch_monitor_workspace_v3 import WorkspaceExportAuthorization


class PlayerWorkspaceExportMixin:
    """Require an explicit restricted-row decision for every full export."""

    def export_dialog(self: Any) -> None:
        if not self._export_payload:
            return
        selected = QFileDialog.getExistingDirectory(self, "Choose Full Export Parent")
        if not selected:
            return
        approval = QMessageBox.question(
            self,
            "Export Restricted Backing Rows?",
            "The project and aggregate results are row-free. Include retained source "
            "rows only if this destination is approved for restricted data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        export_analysis_bundle(
            Path(selected) / "launch-monitor-analysis-bundle",
            self.project(),
            self._export_payload,
            self._frame,
            WorkspaceExportAuthorization(
                include_backing_rows=True,
                restricted_data_approved=approval == QMessageBox.StandardButton.Yes,
            ),
        )
        if approval != QMessageBox.StandardButton.Yes:
            self.status.setText(
                "Row-free v3 bundle exported; backing rows unavailable because "
                "restricted approval was not granted."
            )


__all__ = ["PlayerWorkspaceExportMixin"]
