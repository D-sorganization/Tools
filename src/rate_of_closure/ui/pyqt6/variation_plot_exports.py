"""Testable PNG, SVG, and plot-definition exports for variation figures."""

from __future__ import annotations

import csv
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import TypedDict

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QWidget

from rate_of_closure.ui.pyqt6.variation_plot_helpers import dataset_values
from rate_of_closure.variation.plot_data import (
    EnsemblePlotDataset,
    scalar_plot_variables,
)
from rate_of_closure.variation.plot_definition import (
    PlotDefinition,
    write_plot_definition,
)
from shared.python.swing_sim.variation import (
    ELLIPSOID_VOLUME,
    LowVariabilityMetricCriteria,
    VariationDataset,
)


class _DispersionDefinitionFields(TypedDict):
    """Exact keyword shape consumed by geometric plot definitions."""

    dispersion_metric: str
    dispersion_unit: str
    quiet_threshold: float
    confidence_level: float | None
    min_quiet_duration_s: float
    min_quiet_samples: int


def _dispersion_definition_fields(
    criteria: LowVariabilityMetricCriteria,
) -> _DispersionDefinitionFields:
    """Return versioned authority-unit state shared by both geometric exports."""
    return {
        "dispersion_metric": criteria.metric,
        "dispersion_unit": ("m^3" if criteria.metric == ELLIPSOID_VOLUME else "m"),
        "quiet_threshold": criteria.max_value,
        "confidence_level": (
            criteria.confidence_level if criteria.metric == ELLIPSOID_VOLUME else None
        ),
        "min_quiet_duration_s": criteria.min_duration_s,
        "min_quiet_samples": criteria.min_samples,
    }


def distribution_matrix_plot_definition(
    dataset: EnsemblePlotDataset | None,
    variation: VariationDataset | None,
    variable_keys: tuple[str, ...],
) -> PlotDefinition:
    """Build a reproducible definition for a selected distribution matrix."""
    if variation is None:
        raise RuntimeError("no variation result is loaded")
    return PlotDefinition(
        result_id=(
            dataset.result_id
            if dataset is not None
            else f"variation-{variation.plan.seed}-{variation.plan.n_runs}"
        ),
        plot_type="distribution_matrix",
        coordinate_frame=None,
        variable_keys=variable_keys,
    )


def distribution_matrix_csv(
    variation: VariationDataset | None,
    variable_keys: tuple[str, ...],
    outcomes: tuple[str, ...] | None = None,
) -> str:
    """Serialize every trial row for the selected matrix variables."""
    if variation is None:
        raise RuntimeError("no variation result is loaded")
    variables = {item.key: item for item in scalar_plot_variables(variation)}
    selected = [variables[key] for key in variable_keys]
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    cohort_labels = outcomes or tuple(
        "evaluated" if success else "failure" for success in variation.success
    )
    if len(cohort_labels) != variation.plan.n_runs:
        raise ValueError("outcomes must align with variation trials")
    writer.writerow(("trial_index", "outcome", *variable_keys))
    columns = [dataset_values(variation, variable) for variable in selected]
    for trial_index, outcome in enumerate(cohort_labels):
        writer.writerow(
            (
                trial_index,
                outcome,
                *(
                    "" if not np.isfinite(column[trial_index]) else column[trial_index]
                    for column in columns
                ),
            )
        )
    return output.getvalue().rstrip("\n")


def scatter_plot_definition(
    dataset: EnsemblePlotDataset | None,
    variation: VariationDataset | None,
    x_variable_key: str,
    y_variable_key: str,
    selected_trial_index: int | None,
) -> PlotDefinition:
    """Build a scalar definition from current selector state."""
    if variation is None:
        raise RuntimeError("no variation result is loaded")
    result_id = (
        dataset.result_id
        if dataset
        else (f"variation-{variation.plan.seed}-{variation.plan.n_runs}")
    )
    return PlotDefinition(
        result_id=result_id,
        plot_type="scalar_scatter",
        coordinate_frame=None,
        x_variable_key=x_variable_key,
        y_variable_key=y_variable_key,
        selected_trial_index=selected_trial_index,
    )


def arc_plot_definition(
    dataset: EnsemblePlotDataset | None,
    point_id: str,
    criteria: LowVariabilityMetricCriteria,
    selected_trial_index: int | None,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
    outcome_filter: str | None,
    phase_end_fraction: float,
    perturbation_source_key: str | None,
    perturbation_band: str | None,
    show_confidence_ellipsoids: bool,
) -> PlotDefinition:
    """Build a geometric definition from current selector and camera state."""
    if dataset is None:
        raise RuntimeError("no swing ensemble is loaded")
    return PlotDefinition(
        result_id=dataset.result_id,
        plot_type="swing_arc_overlay",
        coordinate_frame=dataset.coordinate_frame,
        point_id=point_id,
        position_unit="m",
        alignment_basis="common_simulation_time_s",
        **_dispersion_definition_fields(criteria),
        selected_trial_index=selected_trial_index,
        camera_yaw_deg=camera_yaw_deg,
        camera_pitch_deg=camera_pitch_deg,
        outcome_filter=outcome_filter,
        phase_end_fraction=phase_end_fraction,
        perturbation_source_key=perturbation_source_key,
        perturbation_band=perturbation_band,
        show_confidence_ellipsoids=show_confidence_ellipsoids,
    )


def geometric_variability_plot_definition(
    dataset: EnsemblePlotDataset | None,
    point_id: str,
    criteria: LowVariabilityMetricCriteria,
    outcome_filter: str | None,
    phase_end_fraction: float,
    perturbation_source_key: str | None,
    perturbation_band: str | None,
) -> PlotDefinition:
    """Build a definition for the selected dispersion/quiet-zone timeline."""
    if dataset is None:
        raise RuntimeError("no swing ensemble is loaded")
    return PlotDefinition(
        result_id=dataset.result_id,
        plot_type="geometric_variability",
        coordinate_frame=dataset.coordinate_frame,
        point_id=point_id,
        position_unit="m",
        alignment_basis="common_simulation_time_s",
        **_dispersion_definition_fields(criteria),
        outcome_filter=outcome_filter,
        phase_end_fraction=phase_end_fraction,
        perturbation_source_key=perturbation_source_key,
        perturbation_band=perturbation_band,
    )


class VariationPlotExportControls(QWidget):
    """Compact export bar whose write methods also support headless tests."""

    def __init__(
        self,
        figure: Callable[[], Figure],
        definition: Callable[[], PlotDefinition],
        stem: str,
        csv_data: Callable[[], str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._figure = figure
        self._definition = definition
        self._stem = stem
        self._csv_data = csv_data
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for label, kind, callback in (
            ("Export PNG…", "PNG (*.png)", self.write_png),
            ("Export SVG…", "SVG (*.svg)", self.write_svg),
            ("Plot Definition JSON…", "JSON (*.json)", self.write_definition),
        ):
            button = QPushButton(label)
            export_name = label.removesuffix("…")
            button.setToolTip(
                f"Save the current variation visualization as {export_name}."
            )
            button.clicked.connect(
                lambda _checked=False, file_filter=kind, writer=callback: self._choose(
                    file_filter, writer
                )
            )
            layout.addWidget(button)
        if csv_data is not None:
            button = QPushButton("Export Selected CSV…")
            button.setToolTip(
                "Save every trial row for the variables selected in this plot."
            )
            button.clicked.connect(
                lambda _checked=False: self._choose("CSV (*.csv)", self.write_csv)
            )
            layout.addWidget(button)
        layout.addStretch(1)

    def _choose(self, file_filter: str, writer: Callable[[str | Path], None]) -> None:
        suffix = file_filter.split("*", 1)[-1].split(")", 1)[0]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Variation Plot",
            f"{self._stem}{suffix}",
            file_filter,
        )
        if path:
            writer(path)

    def write_png(self, path: str | Path) -> None:
        """Write the current figure as a 200-dpi PNG."""
        self._figure().savefig(path, dpi=200, format="png")

    def write_svg(self, path: str | Path) -> None:
        """Write the current figure as a vector SVG."""
        self._figure().savefig(path, format="svg")

    def write_definition(self, path: str | Path) -> None:
        """Write the current versioned plot state."""
        write_plot_definition(self._definition(), path)

    def write_csv(self, path: str | Path) -> None:
        """Write selected raw plot rows without dropping unavailable values."""
        if self._csv_data is None:
            raise RuntimeError("this plot has no selected CSV export")
        Path(path).write_text(self._csv_data() + "\n", encoding="utf-8")


__all__ = [
    "VariationPlotExportControls",
    "arc_plot_definition",
    "distribution_matrix_csv",
    "distribution_matrix_plot_definition",
    "geometric_variability_plot_definition",
    "scatter_plot_definition",
]
