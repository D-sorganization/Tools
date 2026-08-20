"""Club Tester and Heavy Hit PyQt6 primary tab (C6, H4)."""

from __future__ import annotations

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.club_tester_controls import ClubTesterControlsPanel
from rate_of_closure.ui.pyqt6.club_tester_models import (
    ClubTesterExecutionResult,
    execute_club_tester_study,
    execute_heavy_hit_sweep,
)
from rate_of_closure.ui.pyqt6.club_tester_results import ClubTesterResultsPanel
from shared.python.golf_club.fitting_engine import fitting_report_to_json

logger = logging.getLogger(__name__)

__all__ = ["ClubTesterTab"]


class ClubTesterTab(QWidget):
    """Primary workbench tab for club fitting counterfactuals and impact coupling."""

    glossaryRequested = pyqtSignal(str)
    studyCompleted = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_result: ClubTesterExecutionResult | None = None
        self._last_json = ""

        self._controls = ClubTesterControlsPanel()
        self._results = ClubTesterResultsPanel()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(self._controls)
        left_scroll.setMinimumWidth(320)

        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self._status_lbl = QLabel("Ready.")
        self._status_lbl.setAccessibleName("Club Tester Status")
        self._status_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        right_layout.addWidget(self._status_lbl)
        right_layout.addWidget(self._results, stretch=1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setWidget(right_content)

        splitter = QSplitter()
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        self._controls.runRequested.connect(self.run_now)
        self._controls.exportRequested.connect(self.export_json)
        self._controls.sweepRequested.connect(self.run_sweep)

        # Run initial evaluation
        self.run_now()

    def run_now(self) -> None:
        """Execute fitting evaluation and update results view."""
        try:
            state = self._controls.state()
            result = execute_club_tester_study(state)
            self._last_result = result
            report_json = fitting_report_to_json(result.report)
            self._last_json = report_json
            self._results.display_results(result, report_json)
            delta_speed = (
                result.report.counterfactuals[0].ball_speed_mps
                - result.report.baseline.ball_speed_mps
            )
            self._status_lbl.setText(
                f"Evaluation completed for {state.preset_club}. "
                f"Ball speed delta: {delta_speed:+.2f} m/s"
            )
            self.studyCompleted.emit(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Club Tester evaluation failed: %s", exc, exc_info=True)
            self._status_lbl.setText(f"Error: {exc}")

    def run_sweep(self) -> None:
        """Run heavy hit multi-axis decoupling sweep."""
        try:
            state = self._controls.state()
            report_json = execute_heavy_hit_sweep(state)
            self._last_json = report_json
            if self._last_result:
                self._results.display_results(self._last_result, report_json)
            self._status_lbl.setText("Heavy hit sweep completed successfully.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Heavy hit sweep failed: %s", exc, exc_info=True)
            self._status_lbl.setText(f"Sweep error: {exc}")

    def export_json(self) -> None:
        """Save report JSON to user-selected file."""
        if not self._last_json:
            self.run_now()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fitting Report JSON",
            "fitting_report.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._last_json)
            self._status_lbl.setText(f"Exported report to {path}")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to export report JSON: %s", exc)
            self._status_lbl.setText(f"Export error: {exc}")

    def last_result(self) -> ClubTesterExecutionResult | None:
        """Return the most recent execution result."""
        return self._last_result
