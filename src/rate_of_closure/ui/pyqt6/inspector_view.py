"""Run-data inspector: sortable time-series table + summary + export.

Shows one :class:`~rate_of_closure.simulation.session.SimulationRun` as
a sortable table of the phase-tagged time series (the same rows the CSV
export writes) with the impact/launch/flight summary numbers above it,
and offers CSV / JSON export through the shared
:mod:`rate_of_closure.simulation.export` functions.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.simulation import (
    SimulationRun,
    write_csv,
    write_json,
    write_screw_csv,
)
from rate_of_closure.simulation.export import CSV_COLUMNS, series_rows

logger = logging.getLogger(__name__)

__all__ = ["InspectorView"]

_SUMMARY_ORDER: tuple[tuple[str, str, str], ...] = (
    ("ball_speed_mph", "Ball Speed", "mph"),
    ("launch_angle_deg", "Launch Angle", "°"),
    ("launch_azimuth_deg", "Launch Direction", "°"),
    ("spin_rpm", "Spin", "rpm"),
    ("carry_m", "Carry", "m"),
    ("max_height_m", "Apex", "m"),
    ("flight_time_s", "Flight Time", "s"),
    ("landing_angle_deg", "Landing Angle", "°"),
)


class _NumericItem(QTableWidgetItem):
    """Table item sorting numerically on its stored value."""

    def __init__(self, value: float, text: str) -> None:
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole, value)
        self.setFlags(self.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        mine = self.data(Qt.ItemDataRole.UserRole)
        theirs = other.data(Qt.ItemDataRole.UserRole)
        if isinstance(mine, float) and isinstance(theirs, float):
            return mine < theirs
        return bool(super().__lt__(other))


def _series_item(value: Any) -> QTableWidgetItem:
    """Build a sortable numeric item or an honest unavailable cell."""
    if value is None:
        item = QTableWidgetItem("—")
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setToolTip("Unavailable for this phase or contact outcome")
        return item
    numeric = float(value)
    return _NumericItem(numeric, f"{numeric:.4f}")


class InspectorView(QWidget):
    """Sortable inspector over one simulation run, with export buttons."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._run: SimulationRun | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QHBoxLayout()
        self._summary_label = QLabel("Run a simulation to populate the inspector.")
        self._summary_label.setWordWrap(True)
        header.addWidget(self._summary_label, stretch=1)

        self._export_csv_button = QPushButton("Export CSV…")
        self._export_csv_button.setToolTip(
            "Write the phase-tagged time series (swing + flight) as CSV."
        )
        self._export_csv_button.setEnabled(False)
        self._export_csv_button.clicked.connect(self._on_export_csv)
        header.addWidget(self._export_csv_button)

        self._export_screw_csv_button = QPushButton("Export Screw CSV…")
        self._export_screw_csv_button.setToolTip(
            "Write the club screw-axis decomposition, velocity components, "
            "and reconstruction residual at every swing sample."
        )
        self._export_screw_csv_button.setEnabled(False)
        self._export_screw_csv_button.clicked.connect(self._on_export_screw_csv)
        header.addWidget(self._export_screw_csv_button)

        self._export_json_button = QPushButton("Export JSON…")
        self._export_json_button.setToolTip(
            "Write parameters, delivery/launch summaries, and the time "
            "series as a JSON document."
        )
        self._export_json_button.setEnabled(False)
        self._export_json_button.clicked.connect(self._on_export_json)
        header.addWidget(self._export_json_button)
        layout.addLayout(header)

        self._table = QTableWidget(0, len(CSV_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(CSV_COLUMNS))
        self._table.setSortingEnabled(True)
        vertical_header = self._table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        layout.addWidget(self._table)

    # ── public API ──────────────────────────────────────────────────
    def set_run(self, run: SimulationRun | None) -> None:
        """Populate (or clear, with ``None``) the inspector."""
        self._run = run
        self._export_csv_button.setEnabled(run is not None)
        self._export_screw_csv_button.setEnabled(run is not None)
        self._export_json_button.setEnabled(run is not None)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        if run is None:
            self._summary_label.setText("Run a simulation to populate the inspector.")
            return

        launch = run.launch
        source = run.config.source_kind.replace("_", " ")
        if launch is None:
            outcome = run.impact_outcome
            self._summary_label.setText(
                f"{run.config.club.name} — {source} swing completed with no "
                f"impact. Closest sampled approach "
                f"{outcome.closest_approach_m * 1000.0:.1f} mm at "
                f"{outcome.candidate_time_s:.3f} s; launch and flight are absent."
            )
        else:
            parts = [
                f"{label} {launch[key]:.1f} {unit}"
                for key, label, unit in _SUMMARY_ORDER
            ]
            assert run.impact_time_s is not None
            self._summary_label.setText(
                f"{run.config.club.name} — {source} swing, impact at "
                f"{run.impact_time_s:.3f} s.  " + " · ".join(parts)
            )

        rows = series_rows(run)
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            phase_item = QTableWidgetItem(str(row[0]))
            phase_item.setFlags(phase_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 0, phase_item)
            for c, value in enumerate(row[1:], start=1):
                self._table.setItem(r, c, _series_item(value))
        self._table.setSortingEnabled(True)

    def run(self) -> SimulationRun | None:
        """The run currently inspected, if any."""
        return self._run

    # ── internals ──────────────────────────────────────────────────
    def _export(self, kind: str) -> None:
        if self._run is None:
            return
        caption = f"Export Simulation Run ({kind.upper()})"
        pattern = f"{kind.upper()} files (*.{kind});;All files (*)"
        path, _selected = QFileDialog.getSaveFileName(
            self, caption, f"simulation_run.{kind}", pattern
        )
        if not path:
            return
        try:
            if kind == "csv":
                write_csv(self._run, path)
            else:
                write_json(self._run, path)
        except OSError as exc:  # surface disk/permission problems
            logger.warning("export failed: %s", exc)
            QMessageBox.warning(self, "Export Failed", str(exc))

    def _on_export_csv(self) -> None:
        self._export("csv")

    def _on_export_json(self) -> None:
        self._export("json")

    def _on_export_screw_csv(self) -> None:
        """Prompt for the dedicated screw-motion CSV destination."""
        if self._run is None:
            return
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Export Club Screw Motion (CSV)",
            "club_screw_motion.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            write_screw_csv(self._run, path)
        except OSError as exc:
            logger.warning("screw export failed: %s", exc)
            QMessageBox.warning(self, "Export Failed", str(exc))
