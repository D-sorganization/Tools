"""Construction and population of PyQt Variation result surfaces."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QTabWidget

from rate_of_closure.ui.pyqt6.variation_results import (
    LandingCanvas,
    SensitivityTable,
    SummaryTable,
)
from rate_of_closure.ui.pyqt6.variation_visualizations import (
    ArcOverlayView,
    DatasetScatterView,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    SensitivityResult,
    VariationDataset,
    dispersion_ellipse,
    spearman_matrix,
    summary_stats,
)


class VariationTabResultsMixin:
    """Attach the standard result widgets to a concrete variation tab."""

    def _build_results_tabs(self) -> QTabWidget:
        (
            tabs,
            self._summary_table,
            self._sensitivity_table,
            self._spearman_table,
            self._landing,
            self._ensemble_scatter,
            self._arc_overlay,
        ) = build_result_tabs()
        return tabs


def build_result_tabs() -> tuple[
    QTabWidget,
    SummaryTable,
    SensitivityTable,
    SensitivityTable,
    LandingCanvas,
    DatasetScatterView,
    ArcOverlayView,
]:
    """Build every result view and return stable widget references."""
    tabs = QTabWidget()
    summary = SummaryTable()
    sensitivity = SensitivityTable()
    sensitivity.setToolTip(
        "One-at-a-time sensitivity: std induced in each output (column) "
        "when only that input's (row) noise is active. Hot cells dominate."
    )
    spearman = SensitivityTable()
    spearman.setToolTip(
        "Spearman rank correlation between each sampled input and each output "
        "over the full study."
    )
    landing = LandingCanvas()
    scatter = DatasetScatterView()
    arcs = ArcOverlayView()
    scatter.selectionChanged.connect(arcs.set_selected_trial)
    arcs.selectionChanged.connect(scatter.set_selected_trial)
    for widget, label in (
        (summary, "Summary"),
        (sensitivity, "Sensitivity"),
        (spearman, "Rank Correlation"),
        (landing, "Landing Dispersion"),
        (scatter, "Impact / Shot Scatter"),
        (arcs, "All Swing Arcs"),
    ):
        tabs.addTab(widget, label)
    return tabs, summary, sensitivity, spearman, landing, scatter, arcs


def populate_result_views(
    dataset: VariationDataset,
    sensitivity: SensitivityResult | None,
    summary: SummaryTable,
    sensitivity_table: SensitivityTable,
    spearman_table: SensitivityTable,
    landing: LandingCanvas,
) -> None:
    """Populate standard scalar, sensitivity, and landing result surfaces."""
    summary.set_stats(summary_stats(dataset))
    rho = spearman_matrix(dataset)
    spearman_table.set_matrix(
        dataset.input_names,
        dataset.output_names,
        rho,
        np.abs(rho),
        value_format="{:+.2f}",
    )
    if sensitivity is not None:
        sensitivity_table.set_matrix(
            sensitivity.input_keys,
            sensitivity.output_names,
            sensitivity.matrix,
            sensitivity.normalized,
        )
    try:
        ellipse = dispersion_ellipse(dataset)
    except (ContractViolationError, ValueError):
        ellipse = None
    landing.set_dataset(dataset, ellipse)


__all__ = [
    "VariationTabResultsMixin",
    "build_result_tabs",
    "populate_result_views",
]
