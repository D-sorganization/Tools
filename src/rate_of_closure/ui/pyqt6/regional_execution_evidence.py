"""Native import-only readback for frozen regional execution evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.regional_execution_readback import (
    RegionalExecutionEventReadback,
    RegionalExecutionEvidence,
    RegionalExecutionReadback,
    RegionalExecutionTransitionReadback,
    read_regional_execution_evidence,
)
from shared.python.swing_sim.ground import GroundRegionalMaterialPlanRequest

MAX_VISIBLE_LEDGER_ROWS = 256


class RegionalExecutionEvidenceHost(Protocol):
    """Minimal host boundary for exact plan binding."""

    def current_request(self) -> GroundRegionalMaterialPlanRequest: ...


class RegionalExecutionEvidenceBox(QGroupBox):
    """Compact evidence controller and accessible readback surface."""

    def __init__(self, host: RegionalExecutionEvidenceHost, parent: QWidget) -> None:
        super().__init__("Regional execution evidence", parent)
        self._host = host
        self.recent_path: Path | None = None
        self._accepted_evidence: RegionalExecutionEvidence | None = None
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
        self.readback_label = QPlainTextEdit("No accepted evidence")
        self.readback_label.setReadOnly(True)
        self.readback_label.setMinimumHeight(180)
        self.readback_label.setAccessibleName("Regional execution evidence readback")
        self.event_summary = QLabel("Events: no accepted evidence")
        self.event_table = _ledger_table(
            (
                "Seq",
                "Event",
                "t (s)",
                "Position (m)",
                "v before (m/s)",
                "v after (m/s)",
                "omega before (rad/s)",
                "omega after (rad/s)",
                "Frame",
            ),
            "Ground execution events",
        )
        self.transition_summary = QLabel("Transitions: no accepted evidence")
        self.transition_table = _ledger_table(
            (
                "Event seq",
                "t (s)",
                "Position (m)",
                "From region / surface",
                "To region / surface",
            ),
            "Regional surface transitions",
        )
        self.ledger_tabs = QTabWidget()
        self.ledger_tabs.setAccessibleName("Regional execution ledger inspection")
        self.ledger_tabs.addTab(
            _ledger_tab(self.event_summary, self.event_table), "Events"
        )
        self.ledger_tabs.addTab(
            _ledger_tab(self.transition_summary, self.transition_table), "Transitions"
        )
        layout = QFormLayout(self)
        layout.addRow(self.description)
        layout.addRow(self.open_button)
        layout.addRow("Import status", self.status_label)
        layout.addRow("Validated result", self.readback_label)
        layout.addRow("Validated ledgers", self.ledger_tabs)

    def clear(self) -> None:
        """Remove evidence made stale by a visible plan edit."""
        self._accepted_evidence = None
        self.readback_label.setPlainText("No accepted evidence")
        _populate_event_table(self.event_table, ())
        _populate_transition_table(self.transition_table, ())
        self.event_summary.setText("Events: no accepted evidence")
        self.transition_summary.setText("Transitions: no accepted evidence")
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
        self._accepted_evidence = evidence
        self.readback_label.setPlainText(_format_readback(evidence.readback))
        _populate_event_table(self.event_table, evidence.readback.events)
        _populate_transition_table(self.transition_table, evidence.readback.transitions)
        self.event_summary.setText(
            _ledger_summary("Events", len(evidence.readback.events))
        )
        self.transition_summary.setText(
            _ledger_summary("Transitions", len(evidence.readback.transitions))
        )
        self.status_label.setText(f"Loaded {path.name}. No physics executed.")
        self.status_label.setAccessibleName("Regional execution evidence success")


def _metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f} m"


def _ledger_table(headers: tuple[str, ...], accessible_name: str) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setAccessibleName(accessible_name)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    vertical_header = table.verticalHeader()
    horizontal_header = table.horizontalHeader()
    if vertical_header is None or horizontal_header is None:
        raise RuntimeError("ledger table headers must be available")
    vertical_header.setVisible(False)
    horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.setMinimumHeight(160)
    return table


def _ledger_tab(summary: QLabel, table: QTableWidget) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(summary)
    layout.addWidget(table)
    return page


def _populate_event_table(
    table: QTableWidget, values: tuple[RegionalExecutionEventReadback, ...]
) -> None:
    visible = values[:MAX_VISIBLE_LEDGER_ROWS]
    table.setRowCount(len(visible))
    for row, value in enumerate(visible):
        cells = (
            str(value.sequence),
            value.event_type,
            f"{value.time_s:.6f}",
            _vector(value.position_m),
            _vector(value.velocity_before_m_s),
            _vector(value.velocity_after_m_s),
            _vector(value.angular_velocity_before_rad_s),
            _vector(value.angular_velocity_after_rad_s),
            value.frame,
        )
        for column, text in enumerate(cells):
            table.setItem(row, column, QTableWidgetItem(text))


def _populate_transition_table(
    table: QTableWidget, values: tuple[RegionalExecutionTransitionReadback, ...]
) -> None:
    visible = values[:MAX_VISIBLE_LEDGER_ROWS]
    table.setRowCount(len(visible))
    for row, value in enumerate(visible):
        cells = (
            str(value.event_sequence),
            f"{value.time_s:.6f}",
            _vector(value.position_m),
            _region_surface(value.from_region_id, value.from_surface_id),
            _region_surface(value.to_region_id, value.to_surface_id),
        )
        for column, text in enumerate(cells):
            table.setItem(row, column, QTableWidgetItem(text))


def _vector(value: tuple[float, float, float]) -> str:
    return "(" + ", ".join(f"{item:.6f}" for item in value) + ")"


def _region_surface(region_id: str | None, surface_id: str) -> str:
    return f"{region_id or 'base'} / {surface_id}"


def _ledger_summary(label: str, total: int) -> str:
    shown = min(total, MAX_VISIBLE_LEDGER_ROWS)
    if shown == total:
        return f"{label}: {total} validated row(s)."
    return f"{label}: showing first {shown} of {total} validated rows."


def _format_readback(value: RegionalExecutionReadback) -> str:
    terminal = value.termination_reason or value.failure_reason or "unavailable"
    lines = [
        f"Status: {value.status} · termination: {terminal} · "
        f"ground time: {_seconds(value.ground_time_s)}",
        f"Plan: {value.plan_id} · surface: {value.surface_id} · "
        f"provider: {value.surface_provider_id} {value.surface_provider_version} · "
        f"model: {value.model_id} {value.model_version} · units: {value.unit_system}",
        f"Carry {_metric(value.carry_distance_m)} · "
        f"bounce {_metric(value.bounce_air_distance_m)} · "
        f"skid {_metric(value.skid_distance_m)} · "
        f"roll {_metric(value.roll_distance_m)}",
        f"Surface path {_metric(value.surface_path_distance_m)} · "
        f"total {_metric(value.total_distance_m)} · "
        f"final downrange {_metric(value.final_downrange_m)} · "
        f"final offline {_metric(value.final_offline_m)}",
        f"Bounces: {_count(value.bounce_count)} · transitions: "
        f"{value.transition_count} · phases: {_phases(value.observed_phases)}",
        f"Calibration: {_calibration(value)}",
        f"Executor source: {value.executor_source_revision} · "
        f"input: {value.executor_input_sha256}",
        f"Qualification limits: {', '.join(value.limitations)}",
    ]
    lines.extend(
        f"Warning {item.code} [{item.severity}]: {item.message}"
        for item in value.warnings
    )
    return "\n".join(lines)


def _seconds(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f} s"


def _count(value: int | None) -> str:
    return "unavailable" if value is None else str(value)


def _phases(values: tuple[str, ...]) -> str:
    return "unavailable" if not values else " -> ".join(values)


def _calibration(value: RegionalExecutionReadback) -> str:
    if value.calibration_kind is None:
        return "unavailable"
    return (
        f"{value.calibration_kind} · {value.calibration_id} · "
        f"{value.calibration_source} · confidence {value.calibration_confidence}"
    )


__all__ = ["QFileDialog", "RegionalExecutionEvidenceBox"]
