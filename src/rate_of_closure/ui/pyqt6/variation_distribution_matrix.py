"""Selectable PyQt scatter matrix with honest marginal distributions."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas
from rate_of_closure.ui.pyqt6.variation_plot_exports import (
    VariationPlotExportControls,
    distribution_matrix_csv,
    distribution_matrix_plot_definition,
)
from rate_of_closure.ui.pyqt6.variation_plot_helpers import axis_label, dataset_values
from rate_of_closure.variation.plot_data import (
    EnsemblePlotDataset,
    ScalarPlotVariable,
    scalar_plot_variables,
)
from shared.python.swing_sim.variation import VariationDataset


class DistributionMatrixView(QWidget):
    """Four-variable matrix: histograms on diagonal, paired scatter elsewhere."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._variation: VariationDataset | None = None
        self._variables: tuple[ScalarPlotVariable, ...] = ()
        self._selectors = [QComboBox() for _ in range(4)]
        form = QFormLayout()
        for index, selector in enumerate(self._selectors, start=1):
            selector.setToolTip(
                "Select an input, contact, impact, or shot variable for this "
                "matrix row and column."
            )
            selector.currentIndexChanged.connect(self._redraw)
            form.addRow(f"Matrix Variable {index}", selector)
        self._status = QLabel("Run a variation study to populate the matrix.")
        self._status.setWordWrap(True)
        self._figure = Figure(figsize=(7.2, 7.2), layout="constrained")
        self._canvas = LifecycleSafeFigureCanvas(self._figure)
        self._canvas.setAccessibleName(
            "Scatter matrix with diagonal marginal histograms"
        )
        self._table = QTableWidget()
        self._table.setAccessibleName("Selected scatter matrix trial data")
        self._table.setToolTip(
            "Accessible trial-by-trial values for the four selected matrix variables."
        )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setMaximumHeight(180)
        self._exports = VariationPlotExportControls(
            lambda: self._figure,
            lambda: distribution_matrix_plot_definition(
                self._variation, self._selected_keys()
            ),
            "variation-distribution-matrix",
            csv_data=lambda: distribution_matrix_csv(
                self._variation, self._selected_keys()
            ),
        )
        self._exports.setEnabled(False)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._status)
        layout.addWidget(self._exports)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._table)

    def set_plot_dataset(self, dataset: EnsemblePlotDataset) -> None:
        """Populate from the universal trace-capable facade."""
        self._set_dataset(dataset.result.variation, dataset.variables)

    def set_variation_dataset(self, dataset: VariationDataset) -> None:
        """Populate from a scalar-only variation result."""
        self._set_dataset(dataset, scalar_plot_variables(dataset))

    def _set_dataset(
        self,
        dataset: VariationDataset,
        variables: tuple[ScalarPlotVariable, ...],
    ) -> None:
        self._variation = dataset
        self._variables = variables
        self._exports.setEnabled(True)
        defaults = _default_indices(variables)
        for selector_index, selector in enumerate(self._selectors):
            selector.blockSignals(True)
            selector.clear()
            for variable in variables:
                selector.addItem(axis_label(variable), variable.key)
            selector.setCurrentIndex(defaults[selector_index])
            selector.blockSignals(False)
        self._redraw()

    def _redraw(self, *_args: object) -> None:
        if self._variation is None or not self._variables:
            return
        by_key = {variable.key: variable for variable in self._variables}
        selected = [by_key[str(selector.currentData())] for selector in self._selectors]
        values = [dataset_values(self._variation, variable) for variable in selected]
        self._figure.clear()
        axes = self._figure.subplots(4, 4, squeeze=False)
        missing_total = 0
        for row in range(4):
            for column in range(4):
                axis = axes[row, column]
                x_values = values[column]
                y_values = values[row]
                if row == column:
                    finite = np.isfinite(x_values)
                    axis.hist(x_values[finite], bins=12, color="#2f8bd6", alpha=0.78)
                    missing_total += int(np.count_nonzero(~finite))
                else:
                    finite = np.isfinite(x_values) & np.isfinite(y_values)
                    axis.scatter(
                        x_values[finite],
                        y_values[finite],
                        s=8,
                        alpha=0.55,
                        color="#2f8bd6",
                        edgecolors="none",
                    )
                axis.tick_params(labelsize=6)
                if row == 3:
                    axis.set_xlabel(axis_label(selected[column]), fontsize=7)
                if column == 0:
                    axis.set_ylabel(axis_label(selected[row]), fontsize=7)
        self._status.setText(
            f"Four-variable matrix across {self._variation.plan.n_runs} trials; "
            f"{missing_total} diagonal values unavailable. Off-diagonal cells "
            "plot finite pairs only; canonical exports retain every miss/failure row."
        )
        self._populate_accessible_table(selected, values)
        self._canvas.draw_idle()

    def _selected_keys(self) -> tuple[str, ...]:
        """Return the stable variable keys currently defining the matrix."""
        return tuple(str(selector.currentData()) for selector in self._selectors)

    def _populate_accessible_table(
        self,
        selected: list[ScalarPlotVariable],
        values: list[np.ndarray],
    ) -> None:
        """Expose every plotted or unavailable trial value without recomputation."""
        variation = self._variation
        if variation is None:
            raise RuntimeError("no variation result is loaded")
        self._table.setRowCount(variation.plan.n_runs)
        self._table.setColumnCount(2 + len(selected))
        self._table.setHorizontalHeaderLabels(
            ["Trial", "Status", *(axis_label(variable) for variable in selected)]
        )
        for trial_index, success in enumerate(variation.success):
            self._table.setItem(trial_index, 0, QTableWidgetItem(str(trial_index + 1)))
            self._table.setItem(
                trial_index,
                1,
                QTableWidgetItem("Evaluated" if success else "Failure"),
            )
            for column_index, column in enumerate(values, start=2):
                value = column[trial_index]
                text = f"{value:.8g}" if np.isfinite(value) else "Unavailable"
                self._table.setItem(trial_index, column_index, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()


def _default_indices(
    variables: tuple[ScalarPlotVariable, ...],
) -> tuple[int, int, int, int]:
    preferred = (
        "input:",
        "output:clubhead_speed_mps",
        "output:carry_m",
        "output:lateral_m",
    )
    indices: list[int] = []
    for key in preferred:
        index = next(
            (i for i, variable in enumerate(variables) if variable.key.startswith(key)),
            min(len(indices), len(variables) - 1),
        )
        indices.append(index)
    return tuple(indices)  # type: ignore[return-value]


__all__ = ["DistributionMatrixView"]
