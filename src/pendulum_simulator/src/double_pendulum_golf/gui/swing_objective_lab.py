"""Swing Objective Lab — compare what a downswing is optimized *for*.

Presentation only. Every number shown here is produced by
:mod:`double_pendulum_golf.swing_objectives`; this module builds controls, runs
the engine on a worker thread, and renders the answer. It deliberately contains
no dynamics, so the same engine serves this surface, the CLI, and the notebooks
without the equations living in two places.

Closes #4771.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from double_pendulum_golf.swing_objectives.comparison import (
    SwingComparison,
    compare_objectives,
)
from double_pendulum_golf.swing_objectives.objectives import SWING_OBJECTIVES
from double_pendulum_golf.swing_objectives.presets import (
    DEFAULT_PRESET,
    SwingBudget,
    build_config,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SwingObjectiveLabWidget",
    "SwingObjectiveLabWindow",
    "ComparisonWorker",
    "format_comparison_matrix",
]

_WINDOW_TITLE = "Swing Objective Lab"
_MIN_SELECTED_OBJECTIVES = 2
_PERCENT_DECIMALS = 1
_METRIC_COLUMNS = (
    "Objective",
    "Value",
    "Units",
    "Clubhead speed (m/s)",
    "Wrist cock at impact (deg)",
    "Hub torque saturated (%)",
    "Max defect",
)


def format_comparison_matrix(comparison: SwingComparison) -> list[list[str]]:
    """Render the cross-evaluation matrix as labelled percentage strings.

    Pure so it can be tested without Qt, and so every cell carries a visible
    value — colour must never be the only encoding.

    Args:
        comparison: A completed comparison.

    Returns:
        Row-major table of formatted cells.
    """
    return [[f"{value:.{_PERCENT_DECIMALS}f}%" for value in row] for row in comparison.matrix]


class ComparisonWorker(QThread):
    """Runs the objective comparison off the UI thread.

    A multi-objective comparison takes seconds; running it inline would freeze
    the interface.
    """

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: Any, objective_keys: tuple[str, ...]) -> None:
        """Store the immutable job description."""
        super().__init__()
        self._config = config
        self._objective_keys = objective_keys

    def run(self) -> None:
        """Solve the comparison and emit the result or the failure reason."""
        try:
            comparison = compare_objectives(self._config, self._objective_keys)
        except (ValueError, KeyError, ArithmeticError) as error:
            logger.exception("Swing objective comparison failed")
            self.failed.emit(str(error))
            return
        self.finished.emit(comparison)


class SwingObjectiveLabWidget(QWidget):
    """Controls and results for the swing objective comparison."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the surface with the default golfer preset loaded."""
        super().__init__(parent)
        self._worker: ComparisonWorker | None = None
        self._status = QLabel("Choose objectives and run a comparison.")
        self._status.setWordWrap(True)
        self._build_ui()

    # --- Construction ---------------------------------------------------------

    def _build_ui(self) -> None:
        """Assemble the control column and the results tables."""
        layout = QHBoxLayout(self)
        layout.addWidget(self._build_controls(), stretch=0)
        layout.addWidget(self._build_results(), stretch=1)

    def _build_controls(self) -> QWidget:
        """Build the objective selector and the shared-budget inputs."""
        panel = QGroupBox("Conditions held identical across objectives")
        form = QFormLayout(panel)

        self.objective_list = QListWidget()
        self.objective_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for key, objective in SWING_OBJECTIVES.items():
            item = QListWidgetItem(objective.name)
            item.setData(0x0100, key)
            item.setToolTip(objective.description)
            self.objective_list.addItem(item)
            item.setSelected(True)
        form.addRow("Objectives", self.objective_list)

        self._add_budget_rows(form)

        self.run_button = QPushButton("Run comparison")
        self.run_button.clicked.connect(self.run_comparison)
        form.addRow(self.run_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        form.addRow(self.progress)
        form.addRow(self._status)
        return panel

    def _add_budget_rows(self, form: QFormLayout) -> None:
        """Add the shared effort-budget inputs held identical across objectives."""
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.20, 0.80)
        self.duration_spin.setSingleStep(0.01)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setValue(DEFAULT_PRESET.duration_s)
        self.duration_spin.setSuffix(" s")
        form.addRow("Downswing duration", self.duration_spin)

        self.hub_torque_spin = QDoubleSpinBox()
        self.hub_torque_spin.setRange(50.0, 600.0)
        self.hub_torque_spin.setSingleStep(10.0)
        self.hub_torque_spin.setValue(DEFAULT_PRESET.hub_torque_nm)
        self.hub_torque_spin.setSuffix(" N·m")
        form.addRow("Hub torque limit", self.hub_torque_spin)

        self.wrist_torque_spin = QDoubleSpinBox()
        self.wrist_torque_spin.setRange(2.0, 80.0)
        self.wrist_torque_spin.setSingleStep(1.0)
        self.wrist_torque_spin.setValue(DEFAULT_PRESET.wrist_torque_nm)
        self.wrist_torque_spin.setSuffix(" N·m")
        form.addRow("Wrist torque limit", self.wrist_torque_spin)

        self.node_spin = QSpinBox()
        self.node_spin.setRange(9, 61)
        self.node_spin.setSingleStep(2)
        self.node_spin.setValue(DEFAULT_PRESET.node_count)
        form.addRow("Collocation nodes", self.node_spin)

    def _build_results(self) -> QWidget:
        """Build the per-objective metric table and the cross-evaluation table."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.result_table = QTableWidget(0, len(_METRIC_COLUMNS))
        self.result_table.setHorizontalHeaderLabels(list(_METRIC_COLUMNS))
        self._stretch(self.result_table)
        layout.addWidget(QLabel("Swing produced by each objective"))
        layout.addWidget(self.result_table)

        self.matrix_table = QTableWidget(0, 0)
        self._stretch(self.matrix_table)
        layout.addWidget(
            QLabel(
                "Cross-evaluation: row = optimized for, column = scored on, "
                "as a percent of the best any swing achieves"
            )
        )
        layout.addWidget(self.matrix_table)
        return panel

    @staticmethod
    def _stretch(table: QTableWidget) -> None:
        """Make a table fill its available width."""
        header = table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    # --- Behaviour ------------------------------------------------------------

    def status_text(self) -> str:
        """Return the current status message, for tests and screen readers."""
        return self._status.text()

    def selected_objective_keys(self) -> tuple[str, ...]:
        """Return the objective keys currently ticked, in registry order."""
        selected = {item.data(0x0100) for item in self.objective_list.selectedItems()}
        return tuple(key for key in SWING_OBJECTIVES if key in selected)

    def run_comparison(self) -> None:
        """Start a comparison on a worker thread using the current settings."""
        keys = self.selected_objective_keys()
        if len(keys) < _MIN_SELECTED_OBJECTIVES:
            self._status.setText("Select at least two objectives to compare.")
            return
        try:
            config = build_config(
                SwingBudget(
                    duration_s=self.duration_spin.value(),
                    hub_torque_nm=self.hub_torque_spin.value(),
                    wrist_torque_nm=self.wrist_torque_spin.value(),
                    node_count=self.node_spin.value(),
                )
            )
        except ValueError as error:
            self._status.setText(str(error))
            return

        self._set_running(True)
        self._worker = ComparisonWorker(config, keys)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _set_running(self, running: bool) -> None:
        """Toggle the busy affordances."""
        self.run_button.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self._status.setText("Solving one downswing per objective…")

    def _on_failed(self, message: str) -> None:
        """Report a solver or configuration failure without losing prior results."""
        self._set_running(False)
        self._status.setText(f"Comparison failed: {message}")

    def _on_finished(self, comparison: SwingComparison) -> None:
        """Render a completed comparison."""
        self._set_running(False)
        self.display_comparison(comparison)

    # --- Rendering ------------------------------------------------------------

    def display_comparison(self, comparison: SwingComparison) -> None:
        """Populate both tables from a completed comparison.

        Args:
            comparison: The comparison to render. Supplying one directly keeps
                presentation testable without running the optimizer.
        """
        self._fill_result_table(comparison)
        self._fill_matrix_table(comparison)
        self._status.setText(self._summarize(comparison))

    def _fill_result_table(self, comparison: SwingComparison) -> None:
        """Write one row per objective into the metric table."""
        keys = comparison.objective_keys
        self.result_table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            for column, text in enumerate(self._metric_row(comparison, key)):
                self.result_table.setItem(row, column, QTableWidgetItem(text))

    @staticmethod
    def _metric_row(comparison: SwingComparison, key: str) -> list[str]:
        """Format one objective's summary row."""
        objective = SWING_OBJECTIVES[key]
        diagnostics = comparison.diagnostics[key]
        result = comparison.results.get(key)
        speed = f"{result.signals.clubhead_speed[-1]:.2f}" if result else "n/a"
        wrist_cock = f"{float(result.states[-1, 1]) * 57.29578:.2f}" if result else "n/a"
        return [
            objective.name,
            f"{float(diagnostics['objective_value']):.3f}",
            objective.units,
            speed,
            wrist_cock,
            f"{100.0 * float(comparison.torque_saturation[key][0]):.0f}",
            f"{float(diagnostics['max_defect']):.1e}",
        ]

    def _fill_matrix_table(self, comparison: SwingComparison) -> None:
        """Write the labelled cross-evaluation matrix."""
        labels = [SWING_OBJECTIVES[key].name for key in comparison.objective_keys]
        self.matrix_table.setRowCount(len(labels))
        self.matrix_table.setColumnCount(len(labels))
        self.matrix_table.setHorizontalHeaderLabels(labels)
        self.matrix_table.setVerticalHeaderLabels(labels)
        self._stretch(self.matrix_table)

        for row, cells in enumerate(format_comparison_matrix(comparison)):
            for column, text in enumerate(cells):
                self.matrix_table.setItem(row, column, QTableWidgetItem(text))

    @staticmethod
    def _summarize(comparison: SwingComparison) -> str:
        """Describe what the table does and does not support concluding."""
        if comparison.is_degenerate:
            return (
                "This comparison is DEGENERATE: the constraints pin the swing, so "
                "every objective returned the same trajectory and the matrix is "
                "all 100%. That is a property of the configuration, not evidence "
                "that the mechanisms agree. Lengthen the downswing or raise the "
                "torque limit to give the objectives room to differ."
            )
        return (
            f"Compared {len(comparison.objective_keys)} objectives. Largest "
            f"difference between swings: {comparison.max_swing_distance:.3f} "
            f"(RMS torque, as a fraction of the budget)."
        )


class SwingObjectiveLabWindow(QMainWindow):
    """Standalone window the launcher tile opens."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the window around a single lab widget."""
        super().__init__(parent)
        self.setWindowTitle(_WINDOW_TITLE)
        self.setCentralWidget(SwingObjectiveLabWidget(self))
