"""Native import-only readback for frozen regional execution evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QWidget,
)

from rate_of_closure.application.regional_execution_readback import (
    RegionalExecutionReadback,
    read_regional_execution_evidence,
)
from shared.python.swing_sim.ground import GroundRegionalMaterialPlanRequest


class RegionalExecutionEvidenceHost(Protocol):
    """Minimal host boundary for exact plan binding."""

    def current_request(self) -> GroundRegionalMaterialPlanRequest: ...


class RegionalExecutionEvidenceBox(QGroupBox):
    """Compact evidence controller and accessible readback surface."""

    def __init__(self, host: RegionalExecutionEvidenceHost, parent: QWidget) -> None:
        super().__init__("Regional execution evidence", parent)
        self._host = host
        self.recent_path: Path | None = None
        self.description = QLabel(
            "Import a canonical Python-produced result for this exact plan. "
            "This readback does not execute or modify physics."
        )
        self.description.setWordWrap(True)
        self.open_button = QPushButton("Open execution JSON")
        self.open_button.setAccessibleName("Open regional execution evidence JSON")
        self.open_button.setToolTip(
            "Strictly validate Python-produced evidence and bind it to the "
            "visible plan."
        )
        self.open_button.clicked.connect(self.open)
        self.status_label = QLabel("No execution evidence loaded.")
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Regional execution evidence status")
        self.readback_label = QLabel("No accepted evidence")
        self.readback_label.setWordWrap(True)
        self.readback_label.setAccessibleName("Regional execution evidence readback")
        layout = QFormLayout(self)
        layout.addRow(self.description)
        layout.addRow(self.open_button)
        layout.addRow("Import status", self.status_label)
        layout.addRow("Validated result", self.readback_label)

    def clear(self) -> None:
        """Remove evidence made stale by a visible plan edit."""
        self.readback_label.setText("No accepted evidence")
        self.status_label.setText("Plan changed; execution evidence must be reloaded.")

    def open(self) -> None:
        """Replace readback only after strict parsing and exact plan binding."""
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Regional Execution Evidence",
            "" if self.recent_path is None else str(self.recent_path.parent),
            "JSON files (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        try:
            evidence = read_regional_execution_evidence(
                path, self._host.current_request()
            )
        except (OSError, TypeError, ValueError) as exc:
            self.status_label.setText(
                f"Import failed: {exc}. Prior accepted evidence was preserved."
            )
            self.status_label.setAccessibleName("Regional execution evidence error")
            return
        self.recent_path = path
        self.readback_label.setText(_format_readback(evidence.readback))
        self.status_label.setText(f"Loaded {path.name}. No physics executed.")
        self.status_label.setAccessibleName("Regional execution evidence success")


def _metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f} m"


def _format_readback(value: RegionalExecutionReadback) -> str:
    terminal = value.termination_reason or value.failure_reason or "unavailable"
    return (
        f"{value.status} · {terminal} · plan {value.plan_id} · surface "
        f"{value.surface_id} · model {value.model_id} {value.model_version} · "
        f"skid {_metric(value.skid_distance_m)} · "
        f"roll {_metric(value.roll_distance_m)} · "
        f"total {_metric(value.total_distance_m)} · "
        f"{value.transition_count} transition(s) · "
        f"source {value.executor_source_revision} · "
        f"input {value.executor_input_sha256} · "
        f"limits {', '.join(value.limitations)}"
    )


__all__ = ["QFileDialog", "RegionalExecutionEvidenceBox"]
