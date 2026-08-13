"""PyQt wind-strategy ensemble controls, summary, scatter, and export."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.plot_canvas_pane import PlotCanvasPane
from rate_of_closure.ui.pyqt6.wind_strategy_basis import format_wind_strategy_basis
from rate_of_closure.ui.pyqt6.wind_strategy_launch import (
    WindStrategyLaunchContext,
    WindStrategySettings,
    build_strategy_request,
    scalar_ensemble_csv,
    target_hold_note,
)
from rate_of_closure.ui.pyqt6.wind_strategy_lifecycle import WindStrategyGroupBox
from rate_of_closure.ui.pyqt6.wind_strategy_plot import (
    draw_wind_strategy_scatter,
    scatter_availability_text,
)
from rate_of_closure.ui.pyqt6.wind_strategy_worker import WindStrategyWorker
from rate_of_closure.variation.scalar_ensemble_contract import ScalarEnsembleDataset
from rate_of_closure.variation.wind_strategy_plot_adapter import (
    build_wind_strategy_plot_dataset,
)
from shared.python.swing_sim.flight import StrategyAnalysisRequest, WindStrategyAnalysis

logger = logging.getLogger(__name__)
_CONTEXT_POLL_MS = 250


class WindStrategyPanel(WindStrategyGroupBox):
    """Run and inspect current-launch performance under uncertain wind."""

    def __init__(
        self,
        context_provider: Callable[[], WindStrategyLaunchContext],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Wind Strategy Ensemble", parent)
        self._context_provider = context_provider
        self._worker: WindStrategyWorker | None = None
        self._request: StrategyAnalysisRequest | None = None
        self._dataset: ScalarEnsembleDataset | None = None
        self._active_context: WindStrategyLaunchContext | None = None
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._plot = PlotCanvasPane("Wind Strategy Trial Scatter")
        self._build_ui()
        self._connect_invalidation()
        self._context_timer = QTimer(self)
        self._context_timer.setInterval(_CONTEXT_POLL_MS)
        self._context_timer.timeout.connect(self._check_context)
        self._context_timer.start()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._workspace = QTabWidget()
        self._workspace.setAccessibleName("Wind Strategy Setup and Results")
        setup = QWidget()
        setup_layout = QVBoxLayout(setup)
        self._input_grid = self._build_input_form()
        setup_layout.addLayout(self._input_grid)
        setup_layout.addStretch(1)
        self._workspace.addTab(setup, "Setup")
        results = QWidget()
        results_layout = QVBoxLayout(results)
        self._workspace.addTab(results, "Results")
        layout.addWidget(self._workspace, stretch=1)
        layout.addLayout(self._build_run_row())
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._status = QLabel("Ready. Set launch inputs, then analyze.")
        self._status.setWordWrap(True)
        self._status.setAccessibleName("Wind Strategy Status")
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        self._basis = QLabel("Calculation basis: no current result.")
        self._basis.setWordWrap(True)
        self._basis.setAccessibleName("Wind Strategy Calculation Basis")
        results_layout.addWidget(self._basis)
        self._summary = QTableWidget(0, 7)
        self._summary.setHorizontalHeaderLabels(
            ["Strategy", "Complete", "Failed", "Expected Cost"]
            + ["Hold %", "CVaR m", "Info Delta"]
        )
        self._summary.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._summary.setAccessibleName("Wind Strategy Summary")
        self._summary.setMaximumHeight(125)
        results_layout.addWidget(self._summary)
        results_layout.addLayout(self._build_axis_form())
        self._availability = QLabel("No ensemble result yet.")
        self._availability.setWordWrap(True)
        self._availability.setAccessibleName("Wind Scatter Availability")
        results_layout.addWidget(self._availability)
        results_layout.addWidget(self._plot, stretch=1)

    def _build_input_form(self) -> QGridLayout:
        grid = QGridLayout()
        self._trials = QSpinBox()
        self._trials.setRange(1, 100_000)
        self._trials.setValue(100)
        self._trials.setAccessibleName("Wind Strategy Trial Count")
        self._trials.setToolTip(
            "Number of deterministic paired trials, from 1 to 100000."
        )
        self._seed = QLineEdit("4199")
        self._seed.setAccessibleName("Wind Strategy Random Seed")
        self._seed.setToolTip("Reproducible uint32 seed, from 0 to 4294967295.")
        fields: list[tuple[str, QWidget]] = [
            ("Trials", self._trials),
            ("Seed", self._seed),
        ]
        specifications = (
            ("true_speed", "True Wind Speed", 0.0, 80.0, 5.0, " m/s", 2),
            ("true_bearing", "True Wind From Bearing", -180.0, 180.0, 90.0, "°", 1),
            ("speed_bias", "Speed Estimate Bias", -30.0, 30.0, 0.0, " m/s", 2),
            ("speed_std", "Speed Estimate Std Dev", 0.0, 30.0, 1.0, " m/s", 2),
            ("bearing_bias", "Bearing Estimate Bias", -180.0, 180.0, 0.0, "°", 1),
            ("bearing_std", "Bearing Estimate Std Dev", 0.0, 180.0, 5.0, "°", 1),
            ("correlation", "Error Correlation", -1.0, 1.0, 0.0, "", 2),
            ("aim_gain", "Crosswind Aim Gain", -10.0, 10.0, 0.0, "°/(m/s)", 3),
        )
        for key, label, low, high, value, suffix, decimals in specifications:
            spin = self._double_spin(low, high, value, suffix, decimals)
            spin.setAccessibleName(label)
            spin.setToolTip(f"Set {label.lower()}; displayed units are explicit.")
            self._spins[key] = spin
            fields.append((label, spin))
        for index, (label, field) in enumerate(fields):
            row, pair = divmod(index, 2)
            grid.addWidget(QLabel(label), row, pair * 2)
            grid.addWidget(field, row, pair * 2 + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return grid

    def _connect_invalidation(self) -> None:
        self._trials.valueChanged.connect(self._invalidate_settings)
        self._seed.textChanged.connect(self._invalidate_settings)
        for spin in self._spins.values():
            spin.valueChanged.connect(self._invalidate_settings)

    @staticmethod
    def _double_spin(
        low: float,
        high: float,
        value: float,
        suffix: str,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    def _build_run_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._run = QPushButton("Analyze Wind Strategy")
        self._run.setAccessibleName("Run Wind Strategy Analysis")
        self._run.setToolTip("Run paired true/estimated-wind trials off the UI thread.")
        self._run.clicked.connect(self._on_run)
        self._cancel = QPushButton("Cancel")
        self._cancel.setEnabled(False)
        self._cancel.setToolTip(
            "Cancel cooperatively after the in-flight outcome finishes."
        )
        self._cancel.clicked.connect(self._on_cancel)
        self._export = QPushButton("Export All Rows CSV…")
        self._export.setEnabled(False)
        self._export.setToolTip(
            "Export every raw trial, status, scalar, and unavailable cell."
        )
        self._export.clicked.connect(self._on_export)
        row.addWidget(self._run)
        row.addWidget(self._cancel)
        row.addWidget(self._export)
        return row

    def _build_axis_form(self) -> QFormLayout:
        form = QFormLayout()
        self._x_axis = QComboBox()
        self._y_axis = QComboBox()
        self._x_axis.setToolTip("Choose any declared scalar for the horizontal axis.")
        self._y_axis.setToolTip("Choose any declared scalar for the vertical axis.")
        self._x_axis.currentIndexChanged.connect(self._redraw)
        self._y_axis.currentIndexChanged.connect(self._redraw)
        form.addRow("Horizontal Axis", self._x_axis)
        form.addRow("Vertical Axis", self._y_axis)
        return form

    def settings(self) -> WindStrategySettings:
        """Return the validated user-authored settings without hidden defaults."""
        value = lambda key: self._spins[key].value()  # noqa: E731
        try:
            seed = int(self._seed.text().strip())
        except ValueError as exc:
            raise ValueError("seed must be a uint32 integer") from exc
        return WindStrategySettings(
            self._trials.value(),
            value("true_speed"),
            value("true_bearing"),
            value("speed_bias"),
            value("speed_std"),
            value("bearing_bias"),
            value("bearing_std"),
            value("correlation"),
            value("aim_gain"),
            seed,
        )

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            context = self._context_provider()
            request = build_strategy_request(context, self.settings())
        except (TypeError, ValueError) as exc:
            self._status.setText(f"Cannot analyze: {exc}")
            return
        self._clear_result()
        self._request = request
        self._active_context = context
        self._set_running(True)
        self._status.setText(f"Running… {target_hold_note(context.target)}")
        worker = WindStrategyWorker(request)
        worker.progressed.connect(self._on_progress)
        worker.succeeded.connect(self._on_succeeded)
        worker.cancelled.connect(self._on_cancelled)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        worker.start()

    def _set_running(self, running: bool) -> None:
        self._run.setEnabled(not running)
        self._cancel.setEnabled(running)
        if running:
            total = self._trials.value()
            self._progress.setRange(0, total)
            self._progress.setValue(0)

    def _on_progress(self, completed: int, total: int) -> None:
        if self._request is None:
            return
        self._progress.setRange(0, total)
        self._progress.setValue(completed)
        self._status.setText(f"Running wind strategies: {completed}/{total} outcomes.")

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("Cancelling after the current outcome…")

    def _on_cancelled(self) -> None:
        if self._request is not None:
            self._status.setText("Cancelled. No partial ensemble was published.")

    def _on_failed(self, message: str) -> None:
        if self._request is None:
            return
        self._status.setText(f"Wind strategy analysis failed: {message}")

    def _on_succeeded(self, result: WindStrategyAnalysis) -> None:
        if self._request is None:
            logger.info("discarded stale wind strategy result after invalidation")
            return
        self.apply_result(self._request, result)

    def _on_finished(self) -> None:
        self._set_running(False)
        self._worker = None

    def apply_result(
        self,
        request: StrategyAnalysisRequest,
        result: WindStrategyAnalysis,
    ) -> None:
        """Publish one completed result into every null-safe consumer."""
        try:
            dataset = build_wind_strategy_plot_dataset(request, result)
        except (TypeError, ValueError) as exc:
            logger.warning("wind strategy result adaptation failed: %s", exc)
            self._on_failed(str(exc))
            return
        self._request = request
        self._dataset = dataset
        try:
            self._active_context = self._context_provider()
        except (TypeError, ValueError):
            self._active_context = None
        self._populate_summary(result)
        self._basis.setText(format_wind_strategy_basis(request, result))
        self._populate_axes(dataset)
        self._export.setEnabled(True)
        self._progress.setRange(0, len(result.outcomes))
        self._progress.setValue(len(result.outcomes))
        self._status.setText(
            f"Completed {len(result.outcomes)} outcomes across "
            f"{len(result.summaries)} strategy."
        )
        self._workspace.setCurrentIndex(1)

    def _populate_summary(self, result: WindStrategyAnalysis) -> None:
        self._summary.setRowCount(len(result.summaries))
        for row, summary in enumerate(result.summaries):
            values = (
                summary.label,
                str(summary.completed_trials),
                str(summary.failed_trials),
                f"{summary.expected_cost:.4g}",
                f"{100.0 * summary.target_hold_probability:.1f}",
                f"{summary.miss_distance_cvar_m:.3f}",
                f"{summary.expected_information_cost_delta:.4g}",
            )
            for column, value in enumerate(values):
                self._summary.setItem(row, column, QTableWidgetItem(value))
        self._summary.resizeColumnsToContents()

    def _populate_axes(self, dataset: ScalarEnsembleDataset) -> None:
        for combo in (self._x_axis, self._y_axis):
            combo.blockSignals(True)
            combo.clear()
            for variable in dataset.variables:
                combo.addItem(f"{variable.label} [{variable.unit}]", variable.key)
            combo.blockSignals(False)
        self._select_axis(self._x_axis, "true_wind_left_mps")
        self._select_axis(self._y_axis, "actual_landing_right_m")
        self._redraw()

    @staticmethod
    def _select_axis(combo: QComboBox, key: str) -> None:
        combo.setCurrentIndex(max(0, combo.findData(key)))

    def _redraw(self, *_args: object) -> None:
        if self._dataset is None or self._x_axis.currentIndex() < 0:
            return
        scatter = self._dataset.scatter(
            str(self._x_axis.currentData()), str(self._y_axis.currentData())
        )
        self._plot.render_custom(partial(draw_wind_strategy_scatter, scatter=scatter))
        self._availability.setText(scatter_availability_text(scatter))

    def _on_export(self) -> None:
        if self._dataset is None:
            self._status.setText("Nothing to export: run the analysis first.")
            return
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Export Wind Strategy Rows",
            "wind_strategy_rows.csv",
            "CSV files (*.csv)",
        )
        if not selected:
            return
        try:
            Path(selected).write_text(
                scalar_ensemble_csv(self._dataset) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            self._status.setText(f"CSV export failed: {exc}")
        else:
            self._status.setText(
                f"Exported {len(self._dataset.rows)} raw rows to {selected}."
            )


__all__ = ["WindStrategyPanel"]
