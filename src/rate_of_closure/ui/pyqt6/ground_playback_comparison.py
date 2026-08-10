"""Atomic comparison controls shared by the PyQt ground playback tab."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.simulation.ground_playback import (
    DEFAULT_IMPORT_MAX_BYTES,
    GroundPlaybackTimeline,
    load_ground_result_json,
)
from rate_of_closure.simulation.ground_playback_comparison import (
    GroundPlaybackComparison,
    ground_comparison_csv,
    ground_comparison_json,
)


class GroundPlaybackComparisonMixin:
    """Own one optional comparison without weakening primary last-good behavior."""

    _timeline: GroundPlaybackTimeline | None
    _comparison: GroundPlaybackComparison | None
    current_time_s: float
    view: Any
    pause: Callable[[], None]
    set_time: Callable[[float], None]

    @property
    def has_comparison(self) -> bool:
        return self._comparison is not None

    @property
    def comparison(self) -> GroundPlaybackComparison:
        if self._comparison is None:
            raise RuntimeError("no comparison result has been imported")
        return self._comparison

    @property
    def comparison_is_shown(self) -> bool:
        return self.has_comparison and self.show_comparison_checkbox.isChecked()

    @property
    def _start_time_s(self) -> float:
        return float(
            self.comparison.start_time_s
            if self.comparison_is_shown
            else self._require_primary.start_time_s
        )

    @property
    def _end_time_s(self) -> float:
        return float(
            self.comparison.end_time_s
            if self.comparison_is_shown
            else self._require_primary.end_time_s
        )

    @property
    def _duration_s(self) -> float:
        return self._end_time_s - self._start_time_s

    @property
    def _require_primary(self) -> GroundPlaybackTimeline:
        if self._timeline is None:
            raise RuntimeError("no primary result has been imported")
        return self._timeline

    def _step_time(self, time_s: float, direction: int) -> float:
        if self.comparison_is_shown:
            return float(self.comparison.step_time(time_s, direction))
        return float(self._require_primary.step_time(time_s, direction))

    def attach_comparison_controls(self, panel: QWidget, layout: QVBoxLayout) -> None:
        """Attach import, visibility, export, status, and complete scalar table."""
        self._comparison = None
        group = QGroupBox("Comparison")
        controls = QGridLayout(group)
        self.import_comparison_button = QPushButton("Import Comparison JSON…")
        self.import_comparison_button.setAccessibleName(
            "Import ground comparison result JSON"
        )
        self.import_comparison_button.setToolTip(
            "Import one exact result as a comparison without replacing the primary."
        )
        self.import_comparison_button.clicked.connect(self._choose_comparison_file)
        self.show_comparison_checkbox = QCheckBox("Show comparison")
        self.show_comparison_checkbox.setAccessibleName("Show comparison overlay")
        self.show_comparison_checkbox.setToolTip(
            "Show or hide the synchronized dashed comparison path and event markers."
        )
        self.show_comparison_checkbox.toggled.connect(self._toggle_comparison)
        self.export_comparison_json_button = QPushButton("Comparison JSON")
        self.export_comparison_json_button.setAccessibleName(
            "Export ground comparison JSON"
        )
        self.export_comparison_json_button.setToolTip(
            "Export both exact results and direct comparison-minus-primary deltas."
        )
        self.export_comparison_json_button.clicked.connect(
            lambda: self._save_comparison(
                "ground-comparison.json", self.comparison_json
            )
        )
        self.export_comparison_csv_button = QPushButton("Comparison CSV")
        self.export_comparison_csv_button.setAccessibleName(
            "Export ground comparison CSV"
        )
        self.export_comparison_csv_button.setToolTip(
            "Export the complete direct scalar comparison table as deterministic CSV."
        )
        self.export_comparison_csv_button.clicked.connect(
            lambda: self._save_comparison("ground-comparison.csv", self.comparison_csv)
        )
        controls.addWidget(self.import_comparison_button, 0, 0, 1, 2)
        controls.addWidget(self.show_comparison_checkbox, 1, 0, 1, 2)
        controls.addWidget(self.export_comparison_json_button, 2, 0)
        controls.addWidget(self.export_comparison_csv_button, 2, 1)
        self.comparison_status_label = QLabel("No comparison loaded.")
        self.comparison_status_label.setAccessibleName("Ground comparison status")
        self.comparison_status_label.setWordWrap(True)
        controls.addWidget(self.comparison_status_label, 3, 0, 1, 2)
        layout.addWidget(group)
        self._set_comparison_enabled(False)

    def attach_comparison_detail_tabs(self, tabs: QTabWidget) -> None:
        """Place wide comparison evidence in the resizable detail workspace."""
        self.comparison_table = QTableWidget(0, 4)
        self.comparison_table.setHorizontalHeaderLabels(
            ("Metric", "Primary", "Comparison", "Comparison − primary")
        )
        self.comparison_table.setAccessibleName("Ground result comparison table")
        self.comparison_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.comparison_provenance_table = QTableWidget(0, 3)
        self.comparison_provenance_table.setHorizontalHeaderLabels(
            ("Field", "Primary", "Comparison")
        )
        self.comparison_provenance_table.setAccessibleName(
            "Ground comparison identity status and provenance"
        )
        self.comparison_provenance_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        tabs.addTab(self.comparison_table, "Comparison")
        tabs.addTab(self.comparison_provenance_table, "Comparison provenance")

    def import_comparison_json_text(
        self, text: str, *, source_name: str = "comparison result JSON"
    ) -> None:
        """Atomically replace only the comparison after complete validation."""
        if self._timeline is None:
            raise RuntimeError("import a primary result before a comparison")
        try:
            candidate_timeline = GroundPlaybackTimeline(load_ground_result_json(text))
            candidate = GroundPlaybackComparison(self._timeline, candidate_timeline)
        except (TypeError, ValueError) as exc:
            retained = (
                " Last valid comparison remains loaded." if self.has_comparison else ""
            )
            self.comparison_status_label.setText(
                f"Could not import {source_name}: {exc}.{retained}"
            )
            self.comparison_status_label.setProperty("state", "error")
            raise
        self.pause()
        self._comparison = candidate
        self.view.set_comparison_timeline(candidate.comparison)
        self._populate_comparison(candidate)
        self._set_comparison_enabled(True)
        self.show_comparison_checkbox.setChecked(True)
        self.set_time(self.current_time_s)
        self.comparison_status_label.setText(
            f"Loaded comparison {source_name} — "
            f"{candidate.comparison.result.status.value}; "
            f"{len(candidate.comparison.result.trajectory)} samples. "
            "Deltas are comparison minus primary."
        )
        self.comparison_status_label.setProperty("state", "ready")

    def on_primary_timeline_applied(self) -> None:
        """Clear an old comparison only after a new primary commits successfully."""
        if not self.has_comparison:
            return
        self._comparison = None
        self.view.clear_comparison()
        self.comparison_table.setRowCount(0)
        self.comparison_provenance_table.setRowCount(0)
        self._set_comparison_enabled(False)
        self.comparison_status_label.setText(
            "Comparison cleared after the primary result changed."
        )

    def comparison_json(self) -> str:
        return cast(str, ground_comparison_json(self.comparison))

    def comparison_csv(self) -> str:
        return cast(str, ground_comparison_csv(self.comparison))

    def _populate_comparison(self, comparison: GroundPlaybackComparison) -> None:
        rows = comparison.metric_rows
        self.comparison_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                f"{row.label} [{row.unit}]",
                f"{row.primary:.6g}",
                f"{row.comparison:.6g}",
                f"{row.delta:+.6g}",
            )
            for column, value in enumerate(values):
                self.comparison_table.setItem(
                    row_index, column, QTableWidgetItem(value)
                )
        self.comparison_table.resizeColumnsToContents()
        evidence = comparison.provenance_rows
        self.comparison_provenance_table.setRowCount(len(evidence))
        for row_index, row in enumerate(evidence):
            for column, value in enumerate((row.field, row.primary, row.comparison)):
                self.comparison_provenance_table.setItem(
                    row_index, column, QTableWidgetItem(value)
                )
        self.comparison_provenance_table.resizeColumnsToContents()

    def _set_comparison_enabled(self, enabled: bool) -> None:
        self.show_comparison_checkbox.setEnabled(enabled)
        self.show_comparison_checkbox.setChecked(enabled)
        self.export_comparison_json_button.setEnabled(enabled)
        self.export_comparison_csv_button.setEnabled(enabled)

    def _toggle_comparison(self, shown: bool) -> None:
        self.view.set_comparison_visible(shown and self.has_comparison)
        if self._timeline is not None:
            self.set_time(self.current_time_s)

    def _choose_comparison_file(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            cast(QWidget, self), "Import Ground Comparison", "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            file_path = Path(path)
            if file_path.stat().st_size > DEFAULT_IMPORT_MAX_BYTES:
                raise ValueError("ground result JSON exceeds the import size limit")
            self.import_comparison_json_text(
                file_path.read_text(encoding="utf-8"), source_name=file_path.name
            )
        except (OSError, UnicodeError, RuntimeError, TypeError, ValueError) as exc:
            self.comparison_status_label.setText(
                f"Could not import {Path(path).name}: {exc}"
            )

    def _save_comparison(self, name: str, producer: Callable[[], str]) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            cast(QWidget, self), "Export Ground Comparison", name
        )
        if not path:
            return
        try:
            Path(path).write_text(producer(), encoding="utf-8", newline="")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.comparison_status_label.setText(f"Could not export {name}: {exc}")


__all__ = ["GroundPlaybackComparisonMixin"]
