"""Interactive PyQt views over canonical ensemble plot data."""

from __future__ import annotations

from typing import cast

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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
)
from rate_of_closure.ui.pyqt6.variation_plot_helpers import (
    equal_3d_axes,
    point_label,
)
from rate_of_closure.ui.pyqt6.variation_scatter_view import DatasetScatterView
from rate_of_closure.variation.plot_data import EnsemblePlotDataset
from rate_of_closure.variation.plot_definition import PlotDefinition
from rate_of_closure.variation.simulation_types import TrialEvaluationStatus
from shared.python.swing_sim.variation import LowVariabilityCriteria

_COHORT_COLORS = {
    TrialEvaluationStatus.EVALUATED_HIT: "#2f8bd6",
    TrialEvaluationStatus.EVALUATED_NO_IMPACT: "#eb9f3c",
    TrialEvaluationStatus.NUMERICAL_FAILURE: "#d35f5f",
}


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
        self._configure_threshold()
        self._filters = ArcFilterControls()
        self._exports, self._variability_exports = self._build_exports()
        self._build_layout()
        self._connect_controls()
        clear_arc_views(self._canvas, self._variability_canvas)

    def _configure_threshold(self) -> None:
        """Configure the quiet-zone threshold editor."""
        self._quiet_threshold.setRange(0.01, 1000.0)
        self._quiet_threshold.setDecimals(2)
        self._quiet_threshold.setSuffix(" mm")
        self._quiet_threshold.setValue(5.0)
        self._quiet_threshold.setToolTip(
            "Maximum RMS positional radius used to identify contiguous quiet zones."
        )

    def _build_exports(
        self,
    ) -> tuple[VariationPlotExportControls, VariationPlotExportControls]:
        """Create image and definition exporters for both geometric views."""
        arcs = VariationPlotExportControls(
            lambda: self._canvas.figure,
            self._arc_definition,
            "variation-swing-arcs",
        )
        variability = VariationPlotExportControls(
            lambda: self._variability_canvas.figure,
            self._variability_definition,
            "variation-geometric-variability",
        )
        arcs.setEnabled(False)
        variability.setEnabled(False)
        return arcs, variability

    def _build_layout(self) -> None:
        """Assemble selectors, status, exports, and the paired canvases."""
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

    def _connect_controls(self) -> None:
        """Connect all geometric selectors to linked redraw behavior."""
        self._point_combo.currentIndexChanged.connect(self._redraw)
        self._quiet_threshold.valueChanged.connect(self._redraw)
        self._trial_combo.currentIndexChanged.connect(self._selection_changed)
        self._filters.changed.connect(self._redraw)

    def _arc_definition(self) -> PlotDefinition:
        """Build the current all-trial arc plot definition."""
        axes = cast(Axes3D, self._canvas.axes)
        return arc_plot_definition(
            self._dataset,
            str(self._point_combo.currentData()),
            self._quiet_threshold.value() / 1000.0,
            self._selected_trial,
            float(axes.azim),
            float(axes.elev),
            self._filters.outcome_filter,
            self._filters.phase_percent / 100.0,
            self._filters.perturbation_source_key,
            self._filters.perturbation_band,
        )

    def _variability_definition(self) -> PlotDefinition:
        """Build the current filtered variability plot definition."""
        return geometric_variability_plot_definition(
            self._dataset,
            str(self._point_combo.currentData()),
            self._quiet_threshold.value() / 1000.0,
            self._filters.outcome_filter,
            self._filters.phase_percent / 100.0,
            self._filters.perturbation_source_key,
            self._filters.perturbation_band,
        )

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
