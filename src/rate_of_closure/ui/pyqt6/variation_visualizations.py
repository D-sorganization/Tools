"""Interactive PyQt views over canonical ensemble plot data."""

from __future__ import annotations

from typing import cast

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.variation_arc_filters import ArcFilterControls
from rate_of_closure.ui.pyqt6.variation_geometry_rendering import (
    clear_arc_views,
    draw_arc_trials,
    draw_principal_spread,
    draw_variability_timeline,
    set_app_frame_axes,
)
from rate_of_closure.ui.pyqt6.variation_plot_canvas import VariationPlotCanvas
from rate_of_closure.ui.pyqt6.variation_plot_exports import (
    VariationPlotExportControls,
    arc_plot_definition,
    geometric_variability_plot_definition,
    scatter_plot_definition,
)
from rate_of_closure.ui.pyqt6.variation_plot_helpers import (
    availability_text,
    axis_label,
    cohort_label,
    draw_scalar_study_scatter,
    equal_3d_axes,
    point_label,
)
from rate_of_closure.variation.plot_data import (
    EnsemblePlotDataset,
    ScalarPlotVariable,
    scalar_plot_variables,
)
from rate_of_closure.variation.simulation_types import TrialEvaluationStatus
from shared.python.swing_sim.variation import LowVariabilityCriteria, VariationDataset

_COHORT_COLORS = {
    TrialEvaluationStatus.EVALUATED_HIT: "#2f8bd6",
    TrialEvaluationStatus.EVALUATED_NO_IMPACT: "#eb9f3c",
    TrialEvaluationStatus.NUMERICAL_FAILURE: "#d35f5f",
}


class DatasetScatterView(QWidget):
    """Selectable input/impact/shot scatter with cohort availability counts."""

    selectionChanged = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: EnsemblePlotDataset | None = None
        self._variation: VariationDataset | None = None
        self._selected_trial: int | None = None
        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        self._trial_combo = QComboBox()
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
        self._availability = QLabel("Run a trace-capable variation study.")
        self._availability.setWordWrap(True)
        self._canvas = VariationPlotCanvas()
        self._exports = VariationPlotExportControls(
            lambda: self._canvas.figure,
            lambda: scatter_plot_definition(
                self._dataset,
                self._variation,
                str(self._x_combo.currentData()),
                str(self._y_combo.currentData()),
                self._selected_trial,
            ),
            "variation-scatter",
        )
        self._exports.setEnabled(False)
        selectors = QFormLayout()
        selectors.addRow("Horizontal Axis", self._x_combo)
        selectors.addRow("Vertical Axis", self._y_combo)
        selectors.addRow("Highlighted Trial", self._trial_combo)
        layout = QVBoxLayout(self)
        layout.addLayout(selectors)
        layout.addWidget(self._availability)
        layout.addWidget(self._exports)
        layout.addWidget(self._canvas, stretch=1)
        self._x_combo.currentIndexChanged.connect(self._redraw)
        self._y_combo.currentIndexChanged.connect(self._redraw)
        self._trial_combo.currentIndexChanged.connect(self._selection_changed)
        self._clear()

    def set_plot_dataset(self, dataset: EnsemblePlotDataset) -> None:
        """Populate selectors and render the default paired scatter."""
        self._dataset = dataset
        self._variation = dataset.result.variation
        self._set_trials(dataset.result.variation.plan.n_runs)
        self._set_variables(dataset.variables)
        self._exports.setEnabled(True)
        self._redraw()

    def set_variation_dataset(self, dataset: VariationDataset) -> None:
        """Render scalar-only delivery/launch studies with failure accounting."""
        self._dataset = None
        self._variation = dataset
        self._set_trials(dataset.plan.n_runs)
        self._set_variables(scalar_plot_variables(dataset))
        self._exports.setEnabled(True)
        self._redraw()

    def _set_variables(self, variables: tuple[ScalarPlotVariable, ...]) -> None:
        """Populate both axis selectors from one canonical variable list."""
        for combo in (self._x_combo, self._y_combo):
            combo.blockSignals(True)
            combo.clear()
            for variable in variables:
                combo.addItem(axis_label(variable), variable.key)
            combo.blockSignals(False)
        self._select_default(self._x_combo, "input:")
        self._select_default(self._y_combo, "output:carry_m")

    def _set_trials(self, count: int) -> None:
        self._trial_combo.blockSignals(True)
        self._trial_combo.clear()
        self._trial_combo.addItem("All Trials", None)
        for trial_index in range(count):
            self._trial_combo.addItem(f"Trial {trial_index + 1}", trial_index)
        self._trial_combo.blockSignals(False)

    def set_selected_trial(self, trial_index: int | None) -> None:
        self._selected_trial = trial_index
        index = self._trial_combo.findData(trial_index)
        self._trial_combo.blockSignals(True)
        self._trial_combo.setCurrentIndex(max(index, 0))
        self._trial_combo.blockSignals(False)
        self._redraw()

    def _selection_changed(self) -> None:
        self._selected_trial = self._trial_combo.currentData()
        self.selectionChanged.emit(self._selected_trial)
        self._redraw()

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

    def _clear(self) -> None:
        self._canvas.axes.clear()
        self._canvas.apply_theme()
        self._canvas.axes.set_title("Input, Impact, and Shot-Outcome Scatter")
        self._canvas.draw_idle()

    def _redraw(self, *_args: object) -> None:
        dataset = self._dataset
        variation = self._variation
        if variation is None or self._x_combo.currentIndex() < 0:
            return
        if dataset is None:
            self._redraw_scalar(variation)
            return
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
        if self._selected_trial is not None:
            selected = scatter.trial_indices == self._selected_trial
            axes.scatter(
                scatter.x[selected],
                scatter.y[selected],
                s=72,
                facecolors="none",
                edgecolors="#f2f4f8",
                linewidths=1.8,
                label=f"Trial {self._selected_trial + 1}",
            )
        axes.set_xlabel(axis_label(scatter.x_variable))
        axes.set_ylabel(axis_label(scatter.y_variable))
        axes.set_title("Variation Effects Across Typed Trial Outcomes")
        if axes.collections:
            axes.legend(loc="best", fontsize=8)
        self._availability.setText(availability_text(scatter.cohort_summaries.values()))
        self._canvas.draw_idle()

    def _redraw_scalar(self, dataset: VariationDataset) -> None:
        """Render finite paired rows from a scalar-only variation dataset."""
        variables = scalar_plot_variables(dataset)
        by_key = {variable.key: variable for variable in variables}
        x_variable = by_key[str(self._x_combo.currentData())]
        y_variable = by_key[str(self._y_combo.currentData())]
        self._availability.setText(
            draw_scalar_study_scatter(
                self._canvas,
                dataset,
                x_variable,
                y_variable,
            )
        )


class ArcOverlayView(QWidget):
    """Rotatable all-trial 3-D swing arcs with a median reference trace."""

    selectionChanged = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: EnsemblePlotDataset | None = None
        self._selected_trial: int | None = None
        self._point_combo = QComboBox()
        self._trial_combo = QComboBox()
        self._trial_combo.setToolTip(
            "Highlight one trial here and in every linked variation result view."
        )
        self._point_combo.setToolTip(
            "Choose which modeled point to overlay across every variation trial."
        )
        self._status = QLabel("Run a trace-capable swing variation study.")
        self._status.setWordWrap(True)
        self._canvas = VariationPlotCanvas(projection="3d")
        self._variability_canvas = VariationPlotCanvas()
        self._quiet_threshold = QDoubleSpinBox()
        self._quiet_threshold.setRange(0.01, 1000.0)
        self._quiet_threshold.setDecimals(2)
        self._quiet_threshold.setSuffix(" mm")
        self._quiet_threshold.setValue(5.0)
        self._quiet_threshold.setToolTip(
            "Maximum RMS positional radius used to identify contiguous quiet zones."
        )
        self._filters = ArcFilterControls()
        self._exports = VariationPlotExportControls(
            lambda: self._canvas.figure,
            lambda: arc_plot_definition(
                self._dataset,
                str(self._point_combo.currentData()),
                self._quiet_threshold.value() / 1000.0,
                self._selected_trial,
                float(cast(Axes3D, self._canvas.axes).azim),
                float(cast(Axes3D, self._canvas.axes).elev),
                self._filters.outcome_filter,
                self._filters.phase_percent / 100.0,
                self._filters.perturbation_source_key,
                self._filters.perturbation_band,
            ),
            "variation-swing-arcs",
        )
        self._exports.setEnabled(False)
        self._variability_exports = VariationPlotExportControls(
            lambda: self._variability_canvas.figure,
            lambda: geometric_variability_plot_definition(
                self._dataset,
                str(self._point_combo.currentData()),
                self._quiet_threshold.value() / 1000.0,
                self._filters.outcome_filter,
                self._filters.phase_percent / 100.0,
                self._filters.perturbation_source_key,
                self._filters.perturbation_band,
            ),
            "variation-geometric-variability",
        )
        self._variability_exports.setEnabled(False)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Modeled Point"))
        controls.addWidget(self._point_combo, stretch=1)
        controls.addWidget(QLabel("Highlighted Trial"))
        controls.addWidget(self._trial_combo, stretch=1)
        controls.addWidget(QLabel("Quiet Threshold"))
        controls.addWidget(self._quiet_threshold)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._status)
        layout.addWidget(self._filters)
        layout.addWidget(self._exports)
        layout.addWidget(self._variability_exports)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._variability_canvas, stretch=1)
        self._point_combo.currentIndexChanged.connect(self._redraw)
        self._quiet_threshold.valueChanged.connect(self._redraw)
        self._trial_combo.currentIndexChanged.connect(self._selection_changed)
        self._filters.changed.connect(self._redraw)
        clear_arc_views(self._canvas, self._variability_canvas)

    def set_plot_dataset(self, dataset: EnsemblePlotDataset) -> None:
        """Populate modeled points and render every valid trial arc."""
        self._dataset = dataset
        self._filters.set_dataset(dataset)
        self._exports.setEnabled(True)
        self._variability_exports.setEnabled(True)
        self._point_combo.blockSignals(True)
        self._point_combo.clear()
        for point_id in dataset.result.traces.point_ids:
            self._point_combo.addItem(point_label(point_id), point_id)
        self._point_combo.blockSignals(False)
        self._trial_combo.blockSignals(True)
        self._trial_combo.clear()
        self._trial_combo.addItem("All Trials", None)
        for trial_index in range(dataset.result.variation.plan.n_runs):
            self._trial_combo.addItem(f"Trial {trial_index + 1}", trial_index)
        self._trial_combo.blockSignals(False)
        clubhead = self._point_combo.findData("swing.clubhead.reference")
        self._point_combo.setCurrentIndex(max(clubhead, 0))
        self._redraw()

    def set_selected_trial(self, trial_index: int | None) -> None:
        self._selected_trial = trial_index
        index = self._trial_combo.findData(trial_index)
        self._trial_combo.blockSignals(True)
        self._trial_combo.setCurrentIndex(max(index, 0))
        self._trial_combo.blockSignals(False)
        self._redraw()

    def _selection_changed(self) -> None:
        self._selected_trial = self._trial_combo.currentData()
        self.selectionChanged.emit(self._selected_trial)
        self._redraw()

    def _redraw(self, *_args: object) -> None:
        if self._dataset is None or self._point_combo.currentIndex() < 0:
            return
        trial_indices = self._filters.trial_indices(self._dataset)
        sample_count = self._filters.sample_count(
            self._dataset.result.traces.sample_times_s.size
        )
        overlay = self._dataset.arc_overlay(
            str(self._point_combo.currentData()),
            trial_indices=trial_indices,
            sample_count=sample_count,
        )
        variability = self._dataset.geometric_variability(
            overlay.point_id,
            LowVariabilityCriteria(
                max_rms_radius_m=self._quiet_threshold.value() / 1000.0
            ),
            trial_indices=trial_indices,
            sample_count=sample_count,
        )
        axes = self._canvas.axes
        axes.clear()
        self._canvas.apply_theme()
        draw_arc_trials(axes, overlay, self._selected_trial, _COHORT_COLORS)
        draw_principal_spread(axes, variability)
        axes.set_title(f"All Trials — {point_label(overlay.point_id)}")
        set_app_frame_axes(axes)
        equal_3d_axes(axes, overlay)
        valid_trials = sum(bool(np.any(row)) for row in overlay.sample_valid)
        self._status.setText(
            f"{valid_trials}/{len(self._dataset.cohorts)} trials shown; "
            f"phase 0–{self._filters.phase_percent}%; "
            f"{overlay.rendered_vertex_count:,}/{overlay.raw_vertex_count:,} vertices. "
            f"{variability.n_quiet_samples}/{variability.sample_times_s.size} quiet "
            f"samples at <= {self._quiet_threshold.value():g} mm RMS. "
            f"Frame: {overlay.coordinate_frame}; alignment: common simulation time. "
            "Drag to rotate; scroll to zoom."
        )
        axes.legend(loc="best", fontsize=8)
        self._canvas.draw_idle()
        draw_variability_timeline(self._variability_canvas, variability)


__all__ = ["ArcOverlayView", "DatasetScatterView"]
