"""Selectable scalar scatter view over canonical variation plot data."""

from __future__ import annotations

from typing import cast

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QFormLayout, QLabel, QVBoxLayout, QWidget

from rate_of_closure.ui.pyqt6.variation_plot_canvas import VariationPlotCanvas
from rate_of_closure.ui.pyqt6.variation_plot_exports import (
    VariationPlotExportControls,
    distribution_matrix_csv,
    scatter_plot_definition,
)
from rate_of_closure.ui.pyqt6.variation_plot_helpers import (
    availability_text,
    axis_label,
    cohort_label,
    draw_scalar_study_scatter,
)
from rate_of_closure.ui.pyqt6.variation_trial_table import (
    create_trial_table,
    populate_trial_table,
)
from rate_of_closure.variation.plot_data import (
    EnsemblePlotDataset,
    ScalarPlotVariable,
    scalar_plot_variables,
)
from rate_of_closure.variation.plot_definition import PlotDefinition
from rate_of_closure.variation.simulation_types import TrialEvaluationStatus
from shared.python.swing_sim.variation import VariationDataset

_COHORT_COLORS = {
    TrialEvaluationStatus.EVALUATED_HIT: "#2f8bd6",
    TrialEvaluationStatus.EVALUATED_NO_IMPACT: "#eb9f3c",
    TrialEvaluationStatus.NUMERICAL_FAILURE: "#d35f5f",
}


class DatasetScatterView(QWidget):
    """Selectable input/impact/shot scatter with raw accessible rows."""

    selectionChanged = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: EnsemblePlotDataset | None = None
        self._variation: VariationDataset | None = None
        self._variables: tuple[ScalarPlotVariable, ...] = ()
        self._outcomes: tuple[str, ...] = ()
        self._selected_trial: int | None = None
        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        self._trial_combo = QComboBox()
        self._availability = QLabel("Run a trace-capable variation study.")
        self._availability.setWordWrap(True)
        self._canvas = VariationPlotCanvas()
        self._table = create_trial_table(
            "Selected scatter trial data",
            "Every raw trial value for the selected scatter axes.",
        )
        self._exports = VariationPlotExportControls(
            lambda: self._canvas.figure,
            self._definition,
            "variation-scatter",
            csv_data=self._selected_csv,
        )
        self._configure_controls()
        self._build_layout()
        self._connect_controls()
        self._clear()

    def _configure_controls(self) -> None:
        """Describe linked selection and scalar-axis behavior."""
        self._trial_combo.setToolTip(
            "Highlight one trial here and in every linked variation result view."
        )
        self._x_combo.setToolTip(
            "Select any sampled input or available contact, impact, or shot scalar "
            "for the horizontal axis."
        )
        self._y_combo.setToolTip(
            "Select any sampled input or available contact, impact, or shot scalar "
            "for the vertical axis."
        )
        self._exports.setEnabled(False)

    def _build_layout(self) -> None:
        """Assemble selectors, plot, exports, and raw-data alternative."""
        selectors = QFormLayout()
        selectors.addRow("Horizontal Axis", self._x_combo)
        selectors.addRow("Vertical Axis", self._y_combo)
        selectors.addRow("Highlighted Trial", self._trial_combo)
        layout = QVBoxLayout(self)
        layout.addLayout(selectors)
        layout.addWidget(self._availability)
        layout.addWidget(self._exports)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._table)

    def _connect_controls(self) -> None:
        """Connect selectors and table to one linked trial state."""
        self._x_combo.currentIndexChanged.connect(self._redraw)
        self._y_combo.currentIndexChanged.connect(self._redraw)
        self._trial_combo.currentIndexChanged.connect(self._selection_changed)
        self._table.cellClicked.connect(self._table_selected)

    def set_plot_dataset(self, dataset: EnsemblePlotDataset) -> None:
        """Populate selectors and render the default paired scatter."""
        self._dataset = dataset
        self._variation = dataset.result.variation
        self._outcomes = tuple(cohort.value for cohort in dataset.cohorts)
        self._set_trials(dataset.result.variation.plan.n_runs)
        self._set_variables(dataset.variables)
        self._exports.setEnabled(True)
        self._redraw()

    def set_variation_dataset(self, dataset: VariationDataset) -> None:
        """Render scalar-only delivery/launch studies with failure accounting."""
        self._dataset = None
        self._variation = dataset
        self._outcomes = tuple(
            "evaluated" if success else "failure" for success in dataset.success
        )
        self._set_trials(dataset.plan.n_runs)
        self._set_variables(scalar_plot_variables(dataset))
        self._exports.setEnabled(True)
        self._redraw()

    def _set_variables(self, variables: tuple[ScalarPlotVariable, ...]) -> None:
        """Populate both axis selectors from one canonical variable list."""
        self._variables = variables
        for combo in (self._x_combo, self._y_combo):
            combo.blockSignals(True)
            combo.clear()
            for variable in variables:
                combo.addItem(axis_label(variable), variable.key)
            combo.blockSignals(False)
        self._select_default(self._x_combo, "input:")
        self._select_default(self._y_combo, "output:carry_m")

    def _set_trials(self, count: int) -> None:
        """Populate the stable all-trials/highlight selector."""
        self._trial_combo.blockSignals(True)
        self._trial_combo.clear()
        self._trial_combo.addItem("All Trials", None)
        for trial_index in range(count):
            self._trial_combo.addItem(f"Trial {trial_index + 1}", trial_index)
        self._trial_combo.blockSignals(False)

    def set_selected_trial(self, trial_index: int | None) -> None:
        """Apply linked trial selection without emitting a signal loop."""
        self._selected_trial = trial_index
        index = self._trial_combo.findData(trial_index)
        self._trial_combo.blockSignals(True)
        self._trial_combo.setCurrentIndex(max(index, 0))
        self._trial_combo.blockSignals(False)
        if trial_index is None:
            self._table.clearSelection()
        elif trial_index < self._table.rowCount():
            self._table.selectRow(trial_index)
        self._redraw()

    def _selection_changed(self) -> None:
        self._selected_trial = self._trial_combo.currentData()
        self.selectionChanged.emit(self._selected_trial)
        self._redraw()

    def _table_selected(self, row: int, _column: int) -> None:
        """Link a raw table row to every variation visualization."""
        self._selected_trial = row
        self.selectionChanged.emit(row)
        self.set_selected_trial(row)

    @staticmethod
    def _select_default(combo: QComboBox, prefix: str) -> None:
        """Select the first exact/prefixed stable key when present."""
        index = next(
            (
                item
                for item in range(combo.count())
                if str(combo.itemData(item)).startswith(prefix)
            ),
            0,
        )
        combo.setCurrentIndex(index)

    def _selected_variables(self) -> tuple[ScalarPlotVariable, ...]:
        """Return descriptors for the two currently selected scalar axes."""
        by_key = {variable.key: variable for variable in self._variables}
        return (
            by_key[str(self._x_combo.currentData())],
            by_key[str(self._y_combo.currentData())],
        )

    def _selected_csv(self) -> str:
        """Serialize every raw row for the selected scatter axes."""
        return cast(
            str,
            distribution_matrix_csv(
                self._variation,
                tuple(variable.key for variable in self._selected_variables()),
                self._outcomes,
            ),
        )

    def _definition(self) -> PlotDefinition:
        """Build the current versioned scatter definition."""
        return scatter_plot_definition(
            self._dataset,
            self._variation,
            str(self._x_combo.currentData()),
            str(self._y_combo.currentData()),
            self._selected_trial,
        )

    def _clear(self) -> None:
        self._canvas.axes.clear()
        self._canvas.apply_theme()
        self._canvas.axes.set_title("Input, Impact, and Shot-Outcome Scatter")
        self._canvas.draw_idle()

    def _redraw(self, *_args: object) -> None:
        variation = self._variation
        if variation is None or self._x_combo.currentIndex() < 0:
            return
        populate_trial_table(
            self._table, variation, self._selected_variables(), self._outcomes
        )
        if self._dataset is None:
            self._redraw_scalar(variation)
            return
        self._redraw_ensemble(self._dataset)

    def _redraw_ensemble(self, dataset: EnsemblePlotDataset) -> None:
        """Render cohort-colored, paired-finite complete-simulation rows."""
        scatter = dataset.scatter(
            str(self._x_combo.currentData()), str(self._y_combo.currentData())
        )
        axes = self._canvas.axes
        axes.clear()
        self._canvas.apply_theme()
        for cohort in TrialEvaluationStatus:
            mask = np.fromiter((item is cohort for item in scatter.cohorts), dtype=bool)
            if np.any(mask):
                axes.scatter(
                    scatter.x[mask],
                    scatter.y[mask],
                    s=20,
                    alpha=0.72,
                    label=cohort_label(cohort),
                    color=_COHORT_COLORS[cohort],
                    edgecolors="none",
                )
        self._draw_selected(scatter.trial_indices, scatter.x, scatter.y)
        axes.set_xlabel(axis_label(scatter.x_variable))
        axes.set_ylabel(axis_label(scatter.y_variable))
        axes.set_title("Variation Effects Across Typed Trial Outcomes")
        if axes.collections:
            axes.legend(loc="best", fontsize=8)
        self._availability.setText(availability_text(scatter.cohort_summaries.values()))
        self._canvas.draw_idle()

    def _draw_selected(
        self, trial_indices: np.ndarray, x_values: np.ndarray, y_values: np.ndarray
    ) -> None:
        """Emphasize one available trial without fabricating missing points."""
        if self._selected_trial is None:
            return
        selected = trial_indices == self._selected_trial
        self._canvas.axes.scatter(
            x_values[selected],
            y_values[selected],
            s=72,
            facecolors="none",
            edgecolors="#f2f4f8",
            linewidths=1.8,
            label=f"Trial {self._selected_trial + 1}",
        )

    def _redraw_scalar(self, dataset: VariationDataset) -> None:
        """Render finite paired rows from a scalar-only variation dataset."""
        x_variable, y_variable = self._selected_variables()
        self._availability.setText(
            draw_scalar_study_scatter(
                self._canvas,
                dataset,
                x_variable,
                y_variable,
            )
        )


__all__ = ["DatasetScatterView"]
