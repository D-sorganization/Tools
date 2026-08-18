"""Construction and population of PyQt Variation result surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtWidgets import QSizePolicy, QTabWidget

from rate_of_closure.ui.pyqt6.variation_distribution_matrix import (
    DistributionMatrixView,
)
from rate_of_closure.ui.pyqt6.variation_results import (
    LandingCanvas,
    SensitivityTable,
    SummaryTable,
)
from rate_of_closure.ui.pyqt6.variation_visualizations import (
    ArcOverlayView,
    DatasetScatterView,
)
from rate_of_closure.ui.pyqt6.visual_state_frame import VisualStateFrame
from rate_of_closure.variation.plot_data import EnsemblePlotDataset
from rate_of_closure.variation.simulation_types import SimulationEnsembleResult
from rate_of_closure.variation_visual_state import (
    VariationVisualEvent,
    variation_visual_state,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    DispersionEllipse,
    OutputStats,
    SensitivityResult,
    VariationDataset,
    dispersion_ellipse,
    spearman_matrix,
    summary_stats,
)


@dataclass(frozen=True)
class PreparedResultViews:
    """Immutable, fully-derived scalar presentation for one accepted result."""

    dataset: VariationDataset
    stats: tuple[OutputStats, ...]
    spearman: np.ndarray
    spearman_magnitude: np.ndarray
    sensitivity: SensitivityResult | None
    ellipse: DispersionEllipse | None


class VariationTabResultsMixin:
    """Attach the standard result widgets to a concrete variation tab."""

    _visual_frame: VisualStateFrame

    def _build_results_tabs(self) -> VisualStateFrame:
        (
            tabs,
            self._summary_table,
            self._sensitivity_table,
            self._spearman_table,
            self._landing,
            self._ensemble_scatter,
            self._distribution_matrix,
            self._arc_overlay,
        ) = build_result_tabs()
        self._visual_frame = VisualStateFrame(tabs)
        self._visual_frame.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        self._visual_frame.set_state(
            variation_visual_state(VariationVisualEvent.INVALIDATE), "Ready."
        )
        return self._visual_frame

    def _apply_prepared_result(
        self,
        prepared: PreparedResultViews,
        ensemble: SimulationEnsembleResult | None,
        plot_dataset: EnsemblePlotDataset | None,
    ) -> None:
        dataset = prepared.dataset
        if ensemble is not None:
            self._landing.set_outcomes(tuple(item.status for item in ensemble.outcomes))
            if plot_dataset is None:
                raise ValueError("trace result has no prepared plot dataset")
            self._ensemble_scatter.set_plot_dataset(plot_dataset)
            self._distribution_matrix.set_plot_dataset(plot_dataset)
            self._arc_overlay.set_plot_dataset(plot_dataset)
        else:
            self._ensemble_scatter.set_variation_dataset(dataset)
            self._distribution_matrix.set_variation_dataset(dataset)
        populate_result_views(
            prepared,
            self._summary_table,
            self._sensitivity_table,
            self._spearman_table,
            self._landing,
        )

    def _restore_prepared_result(
        self,
        previous: tuple[
            VariationDataset | None,
            SimulationEnsembleResult | None,
            PreparedResultViews | None,
            EnsemblePlotDataset | None,
        ],
    ) -> None:
        dataset, ensemble, prepared, plot_dataset = previous
        if dataset is None or prepared is None:
            self._clear_result_widgets()
            return
        try:
            self._apply_prepared_result(prepared, ensemble, plot_dataset)
        except Exception:
            pass

    def _clear_result_widgets(self) -> None:
        self._summary_table.setRowCount(0)
        for table in (self._sensitivity_table, self._spearman_table):
            table.setRowCount(0)
            table.setColumnCount(0)
        self._landing.clear_view()
        self._ensemble_scatter.clear_view()
        self._distribution_matrix.clear_view()
        self._arc_overlay.clear_view()


def build_result_tabs() -> tuple[
    QTabWidget,
    SummaryTable,
    SensitivityTable,
    SensitivityTable,
    LandingCanvas,
    DatasetScatterView,
    DistributionMatrixView,
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
    matrix = DistributionMatrixView()
    arcs = ArcOverlayView()
    scatter.selectionChanged.connect(arcs.set_selected_trial)
    scatter.selectionChanged.connect(matrix.set_selected_trial)
    arcs.selectionChanged.connect(scatter.set_selected_trial)
    arcs.selectionChanged.connect(matrix.set_selected_trial)
    matrix.selectionChanged.connect(scatter.set_selected_trial)
    matrix.selectionChanged.connect(arcs.set_selected_trial)
    for widget, label in (
        (summary, "Summary"),
        (sensitivity, "Sensitivity"),
        (spearman, "Rank Correlation"),
        (landing, "Landing Dispersion"),
        (scatter, "Impact / Shot Scatter"),
        (matrix, "Scatter Matrix / Marginals"),
        (arcs, "All Swing Arcs"),
    ):
        tabs.addTab(widget, label)
    tabs.setCurrentWidget(landing)
    return tabs, summary, sensitivity, spearman, landing, scatter, matrix, arcs


def prepare_result_views(
    dataset: VariationDataset,
    sensitivity: SensitivityResult | None,
) -> PreparedResultViews:
    """Derive every fallible scalar view model before any widget mutation."""
    prepared_sensitivity = _prepare_sensitivity(dataset, sensitivity)
    rho = np.array(spearman_matrix(dataset), copy=True)
    magnitude = np.abs(rho)
    rho.setflags(write=False)
    magnitude.setflags(write=False)
    try:
        ellipse = dispersion_ellipse(dataset)
    except (ContractViolationError, ValueError):
        ellipse = None
    return PreparedResultViews(
        dataset=dataset,
        stats=summary_stats(dataset),
        spearman=rho,
        spearman_magnitude=magnitude,
        sensitivity=prepared_sensitivity,
        ellipse=ellipse,
    )


def _prepare_sensitivity(
    dataset: VariationDataset, sensitivity: SensitivityResult | None
) -> SensitivityResult | None:
    if sensitivity is None:
        return None
    expected_inputs = tuple(spec.variable_key for spec in dataset.plan.noise)
    expected_shape = (len(expected_inputs), len(dataset.output_names))
    if sensitivity.input_keys != expected_inputs:
        raise ValueError("sensitivity inputs do not match the accepted plan")
    if sensitivity.output_names != dataset.output_names:
        raise ValueError("sensitivity outputs do not match the accepted dataset")
    if not isinstance(sensitivity.matrix, np.ndarray) or not isinstance(
        sensitivity.normalized, np.ndarray
    ):
        raise ValueError("sensitivity matrices must be NumPy arrays")
    if sensitivity.matrix.shape != expected_shape or sensitivity.normalized.shape != (
        expected_shape
    ):
        raise ValueError("sensitivity matrix shape does not match its labels")
    matrix = np.asarray(sensitivity.matrix, dtype=float).copy()
    normalized = np.asarray(sensitivity.normalized, dtype=float).copy()
    if np.any(np.isinf(matrix)) or np.any(np.isinf(normalized)):
        raise ValueError("sensitivity matrices cannot contain infinities")
    finite_normalized = normalized[np.isfinite(normalized)]
    if np.any((finite_normalized < 0.0) | (finite_normalized > 1.0)):
        raise ValueError("normalized sensitivity values must be within [0, 1]")
    matrix.setflags(write=False)
    normalized.setflags(write=False)
    return SensitivityResult(
        input_keys=sensitivity.input_keys,
        output_names=sensitivity.output_names,
        matrix=matrix,
        normalized=normalized,
    )


def populate_result_views(
    prepared: PreparedResultViews,
    summary: SummaryTable,
    sensitivity_table: SensitivityTable,
    spearman_table: SensitivityTable,
    landing: LandingCanvas,
) -> None:
    """Commit one already-derived presentation to the visible widgets."""
    dataset = prepared.dataset
    summary.set_stats(prepared.stats)
    spearman_table.set_matrix(
        dataset.input_names,
        dataset.output_names,
        prepared.spearman,
        prepared.spearman_magnitude,
        value_format="{:+.2f}",
    )
    if prepared.sensitivity is not None:
        sensitivity = prepared.sensitivity
        sensitivity_table.set_matrix(
            sensitivity.input_keys,
            sensitivity.output_names,
            sensitivity.matrix,
            sensitivity.normalized,
        )
    landing.set_dataset(dataset, prepared.ellipse)


__all__ = [
    "VariationTabResultsMixin",
    "PreparedResultViews",
    "build_result_tabs",
    "prepare_result_views",
    "populate_result_views",
]
