"""Accessible raw-trial tables shared by variation plot views."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QAbstractItemView, QTableWidget, QTableWidgetItem

from rate_of_closure.ui.pyqt6.variation_plot_helpers import axis_label, dataset_values
from rate_of_closure.variation.plot_data import ScalarPlotVariable
from shared.python.swing_sim.variation import VariationDataset


def create_trial_table(accessible_name: str, tooltip: str) -> QTableWidget:
    """Create a bounded, read-only table for raw variation rows."""
    table = QTableWidget()
    table.setAccessibleName(accessible_name)
    table.setToolTip(tooltip)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setMaximumHeight(180)
    return table


def populate_trial_table(
    table: QTableWidget,
    dataset: VariationDataset,
    variables: tuple[ScalarPlotVariable, ...],
    outcomes: tuple[str, ...],
) -> None:
    """Populate every raw trial, preserving unavailable scalar values."""
    if len(outcomes) != dataset.plan.n_runs:
        raise ValueError("outcomes must align with variation trials")
    columns = tuple(dataset_values(dataset, variable) for variable in variables)
    table.setRowCount(dataset.plan.n_runs)
    table.setColumnCount(2 + len(variables))
    table.setHorizontalHeaderLabels(
        ["Trial", "Status", *(axis_label(variable) for variable in variables)]
    )
    for trial_index, outcome in enumerate(outcomes):
        table.setItem(trial_index, 0, QTableWidgetItem(str(trial_index + 1)))
        table.setItem(
            trial_index,
            1,
            QTableWidgetItem(outcome.replace("_", " ").title()),
        )
        _populate_values(table, trial_index, columns)
    table.resizeColumnsToContents()


def _populate_values(
    table: QTableWidget,
    trial_index: int,
    columns: tuple[np.ndarray, ...],
) -> None:
    """Populate selected raw values for one trial row."""
    for column_index, column in enumerate(columns, start=2):
        value = column[trial_index]
        text = f"{value:.8g}" if np.isfinite(value) else "Unavailable"
        table.setItem(trial_index, column_index, QTableWidgetItem(text))


__all__ = ["create_trial_table", "populate_trial_table"]
