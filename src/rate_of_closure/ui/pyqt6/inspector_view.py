"""Run-data inspector: sortable time-series table + summary + export.

Shows one :class:`~rate_of_closure.simulation.session.SimulationRun` as
a sortable table of the phase-tagged time series (the same rows the CSV
export writes) with the impact/launch/flight summary numbers above it,
and offers CSV / JSON export through the shared
:mod:`rate_of_closure.simulation.export` functions.
"""

from __future__ import annotations

import logging

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

from rate_of_closure.simulation import SimulationRun, write_csv, write_json
from rate_of_closure.simulation.export import CSV_COLUMNS, series_rows

logger = logging.getLogger(__name__)

__all__ = ["InspectorView"]

_SUMMARY_ORDER: tuple[tuple[str, str, str], ...] = (
    ("ball_speed_mph", "Ball Speed", "mph"),
    ("launch_angle_deg", "Launch Angle", "°"),
    ("launch_azimuth_deg", "Launch Azimuth", "°"),
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
        return super().__lt__(other)


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
        self._table.verticalHeader().setVisible(False)  # type: ignore[union-attr]
        layout.addWidget(self._table)

    # ── public API ──────────────────────────────────────────────────
    def set_run(self, run: SimulationRun | None) -> None:
        """Populate (or clear, with ``None``) the inspector."""
        self._run = run
        self._export_csv_button.setEnabled(run is not None)
        self._export_json_button.setEnabled(run is not None)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        if run is None:
            self._summary_label.setText("Run a simulation to populate the inspector.")
            return

        launch = run.launch
        parts = [
            f"{label} {launch[key]:.1f} {unit}" for key, label, unit in _SUMMARY_ORDER
        ]
        self._summary_label.setText(
            f"{run.config.club.name} — {run.config.source_kind.replace('_', ' ')} "
            f"swing, impact at {run.impact_time_s:.3f} s.  " + " · ".join(parts)
        )

        rows = series_rows(run)
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            phase_item = QTableWidgetItem(str(row[0]))
            phase_item.setFlags(phase_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 0, phase_item)
            for c, value in enumerate(row[1:], start=1):
                self._table.setItem(r, c, _NumericItem(float(value), f"{value:.4f}"))
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
