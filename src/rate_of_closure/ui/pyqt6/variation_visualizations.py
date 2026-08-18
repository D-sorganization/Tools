"""Interactive PyQt views over canonical ensemble plot data."""

from __future__ import annotations

from typing import cast

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.variation_arc_filters import ArcFilterControls
from rate_of_closure.ui.pyqt6.variation_control_helpers import add_labeled_control
from rate_of_closure.ui.pyqt6.variation_geometry_rendering import (
    clear_arc_views,
    confidence_ellipsoid_legend,
    draw_arc_trials,
    draw_confidence_ellipsoid_mesh,
    draw_principal_spread,
    draw_variability_timeline,
    principal_spread_legend,
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
from rate_of_closure.ui.pyqt6.variation_trial_table import validated_trial_index
from rate_of_closure.variation.confidence_ellipsoid_mesh import (
    build_confidence_ellipsoid_mesh,
)
from rate_of_closure.variation.geometric_plot_data import (
    build_dispersion_metric_variability_view,
)
from rate_of_closure.variation.plot_data import EnsemblePlotDataset
from rate_of_closure.variation.plot_definition import PlotDefinition
from rate_of_closure.variation.simulation_types import TrialEvaluationStatus
from shared.python.swing_sim.variation import (
    ELLIPSOID_VOLUME,
    ESTIMABLE,
    INSUFFICIENT_SAMPLES,
    INVALID_COVARIANCE,
    LARGEST_PRINCIPAL_SIGMA,
    RANK_DEFICIENT,
    RMS_RADIUS,
    LowVariabilityMetricCriteria,
)

_COHORT_COLORS = {
    TrialEvaluationStatus.EVALUATED_HIT: "#2f8bd6",
    TrialEvaluationStatus.EVALUATED_NO_IMPACT: "#eb9f3c",
    TrialEvaluationStatus.NUMERICAL_FAILURE: "#d35f5f",
}
_METRIC_LABELS = {
    RMS_RADIUS: "RMS Radius",
    LARGEST_PRINCIPAL_SIGMA: "Largest Principal σ",
    ELLIPSOID_VOLUME: "Confidence-Ellipsoid Volume",
}
_DEFAULT_THRESHOLDS = {
    RMS_RADIUS: 5.0,
    LARGEST_PRINCIPAL_SIGMA: 5.0,
    ELLIPSOID_VOLUME: 1_000.0,
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
        self._metric_combo = QComboBox()
        self._confidence = QDoubleSpinBox()
        self._quiet_threshold = QDoubleSpinBox()
        self._min_duration = QDoubleSpinBox()
        self._min_samples = QSpinBox()
        self._ellipsoid_mesh = QCheckBox()
        self._active_metric = RMS_RADIUS
        self._metric_thresholds = dict(_DEFAULT_THRESHOLDS)
        self._configure_dispersion_controls()
        self._filters = ArcFilterControls()
        self._exports, self._variability_exports = self._build_exports()
        self._build_layout()
        self._connect_controls()
        clear_arc_views(self._canvas, self._variability_canvas)

    def _configure_dispersion_controls(self) -> None:
        """Configure metric-specific threshold, confidence, and continuity state."""
        for metric, label in _METRIC_LABELS.items():
            self._metric_combo.addItem(label, metric)
        self._metric_combo.setAccessibleName("Dispersion metric")
        self._metric_combo.setToolTip(
            "Choose the shared statistical authority used for the timeline "
            "and quiet zones."
        )
        self._confidence.setRange(50.0, 99.9)
        self._confidence.setDecimals(1)
        self._confidence.setSuffix(" %")
        self._confidence.setValue(95.0)
        self._confidence.setEnabled(False)
        self._confidence.setAccessibleName("Dispersion confidence percent")
        self._confidence.setToolTip(
            "Gaussian position-content probability; available only for "
            "ellipsoid volume."
        )
        self._quiet_threshold.setRange(0.001, 1000.0)
        self._quiet_threshold.setDecimals(2)
        self._quiet_threshold.setSuffix(" mm")
        self._quiet_threshold.setValue(5.0)
        self._quiet_threshold.setAccessibleName("Quiet-zone metric threshold")
        self._quiet_threshold.setToolTip(
            "Maximum selected dispersion value used to rank contiguous quiet zones."
        )
        self._min_duration.setRange(0.0, 10.0)
        self._min_duration.setDecimals(3)
        self._min_duration.setSuffix(" s")
        self._min_duration.setAccessibleName("Minimum quiet duration seconds")
        self._min_duration.setToolTip(
            "Minimum common-grid duration for a qualifying quiet interval."
        )
        self._min_samples.setRange(1, 100_000)
        self._min_samples.setValue(1)
        self._min_samples.setAccessibleName("Minimum quiet samples")
        self._min_samples.setToolTip(
            "Minimum number of common-grid samples in a quiet interval."
        )
        self._ellipsoid_mesh.setAccessibleName("Show confidence ellipsoid surfaces")
        self._ellipsoid_mesh.setToolTip(
            "Render bounded Gaussian position-content surfaces for estimable samples."
        )
        self._ellipsoid_mesh.setEnabled(False)

    def _metric_changed(self) -> None:
        """Persist each metric's display threshold and update relevant controls."""
        self._metric_thresholds[self._active_metric] = self._quiet_threshold.value()
        metric = str(self._metric_combo.currentData())
        self._active_metric = metric
        is_volume = metric == ELLIPSOID_VOLUME
        self._confidence.setEnabled(is_volume)
        self._ellipsoid_mesh.setEnabled(is_volume)
        if not is_volume:
            self._ellipsoid_mesh.setChecked(False)
        self._quiet_threshold.blockSignals(True)
        self._quiet_threshold.setRange(0.001, 1.0e12 if is_volume else 1000.0)
        self._quiet_threshold.setDecimals(3 if is_volume else 2)
        self._quiet_threshold.setSuffix(" mm³" if is_volume else " mm")
        self._quiet_threshold.setValue(self._metric_thresholds[metric])
        self._quiet_threshold.blockSignals(False)
        self._redraw()

    def _criteria(self) -> LowVariabilityMetricCriteria:
        """Return selected controls normalized into shared SI authority units."""
        metric = str(self._metric_combo.currentData())
        scale = 1.0e9 if metric == ELLIPSOID_VOLUME else 1.0e3
        return LowVariabilityMetricCriteria(
            metric=metric,
            max_value=self._quiet_threshold.value() / scale,
            confidence_level=self._confidence.value() / 100.0,
            min_duration_s=self._min_duration.value(),
            min_samples=self._min_samples.value(),
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
        selection_controls = QHBoxLayout()
        selection_controls.addWidget(QLabel("Modeled Point"))
        selection_controls.addWidget(self._point_combo, stretch=1)
        selection_controls.addWidget(QLabel("Highlighted Trial"))
        selection_controls.addWidget(self._trial_combo, stretch=1)
        analysis_controls = QHBoxLayout()
        add_labeled_control(
            analysis_controls, "Dispersion Metric", self._metric_combo, stretch=1
        )
        add_labeled_control(analysis_controls, "Confidence", self._confidence)
        add_labeled_control(analysis_controls, "Quiet Threshold", self._quiet_threshold)
        add_labeled_control(analysis_controls, "Min Duration", self._min_duration)
        add_labeled_control(analysis_controls, "Min Samples", self._min_samples)
        add_labeled_control(
            analysis_controls, "Ellipsoid Surfaces", self._ellipsoid_mesh
        )
        layout = QVBoxLayout(self)
        layout.addLayout(selection_controls)
        layout.addLayout(analysis_controls)
        layout.addWidget(self._status)
        layout.addWidget(self._filters)
        layout.addWidget(self._exports)
        layout.addWidget(self._variability_exports)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._variability_canvas, stretch=1)

    def _connect_controls(self) -> None:
        """Connect all geometric selectors to linked redraw behavior."""
        self._point_combo.currentIndexChanged.connect(self._redraw)
        self._metric_combo.currentIndexChanged.connect(self._metric_changed)
        self._confidence.valueChanged.connect(self._redraw)
        self._quiet_threshold.valueChanged.connect(self._redraw)
        self._min_duration.valueChanged.connect(self._redraw)
        self._min_samples.valueChanged.connect(self._redraw)
        self._ellipsoid_mesh.toggled.connect(self._redraw)
        self._trial_combo.currentIndexChanged.connect(self._selection_changed)
        self._filters.changed.connect(self._redraw)

    def _arc_definition(self) -> PlotDefinition:
        """Build the current all-trial arc plot definition."""
        axes = cast(Axes3D, self._canvas.axes)
        return arc_plot_definition(
            self._dataset,
            str(self._point_combo.currentData()),
            self._criteria(),
            self._selected_trial,
            float(axes.azim),
            float(axes.elev),
            self._filters.outcome_filter,
            self._filters.phase_percent / 100.0,
            self._filters.perturbation_source_key,
            self._filters.perturbation_band,
            self._ellipsoid_mesh.isChecked(),
        )

    def _variability_definition(self) -> PlotDefinition:
        """Build the current filtered variability plot definition."""
        return geometric_variability_plot_definition(
            self._dataset,
            str(self._point_combo.currentData()),
            self._criteria(),
            self._filters.outcome_filter,
            self._filters.phase_percent / 100.0,
            self._filters.perturbation_source_key,
            self._filters.perturbation_band,
        )

    def set_plot_dataset(self, dataset: EnsemblePlotDataset) -> None:
        """Populate modeled points and render every valid trial arc."""
        self._selected_trial = None
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

    def clear_view(self) -> None:
        """Remove every result and restore the honest empty state."""
        self._dataset = None
        self._selected_trial = None
        self._point_combo.clear()
        self._trial_combo.clear()
        self._exports.setEnabled(False)
        self._variability_exports.setEnabled(False)
        self._status.setText("Run a trace-capable swing variation study.")
        clear_arc_views(self._canvas, self._variability_canvas)

    def set_selected_trial(self, trial_index: int | None) -> None:
        trial_count = self._dataset.result.variation.plan.n_runs if self._dataset else 0
        selected = validated_trial_index(trial_index, trial_count)
        self._selected_trial = selected
        index = self._trial_combo.findData(selected)
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
        variability = build_dispersion_metric_variability_view(
            self._dataset.dispersion,
            self._dataset.result.traces,
            overlay.point_id,
            self._criteria(),
            trial_indices=trial_indices,
            sample_count=sample_count,
        )
        axes = self._canvas.axes
        axes.clear()
        self._canvas.apply_theme()
        draw_arc_trials(axes, overlay, self._selected_trial, _COHORT_COLORS)
        draw_principal_spread(axes, variability)
        mesh = None
        if self._ellipsoid_mesh.isChecked() and variability.metric == ELLIPSOID_VOLUME:
            mesh = build_confidence_ellipsoid_mesh(
                variability.mean_positions_m,
                variability.principal_axes,
                variability.confidence_semi_axis_lengths_m,
                variability.adequacy,
                variability.coordinate_frame,
            )
            draw_confidence_ellipsoid_mesh(axes, mesh)
        axes.set_title(f"All Trials — {point_label(overlay.point_id)}")
        set_app_frame_axes(axes)
        equal_3d_axes(axes, overlay, None if mesh is None else mesh.vertices_m)
        valid_trials = sum(bool(np.any(row)) for row in overlay.sample_valid)
        counts = variability.adequacy_counts
        ranked = (
            ", ".join(
                f"#{item.rank} {item.start_time_s:.3f}–{item.end_time_s:.3f} s "
                f"(score {item.score:.3f})"
                for item in variability.quiet_intervals[:3]
            )
            or "none"
        )
        interpretation = (
            f"{100.0 * variability.confidence_level:.1f}% Gaussian position-content "
            "region (plug-in sample covariance; not a confidence region for the mean)."
            if variability.confidence_level is not None
            else "Sample-position dispersion; confidence does not apply."
        )
        self._status.setText(
            f"{valid_trials}/{len(self._dataset.cohorts)} trials shown; "
            f"phase 0–{self._filters.phase_percent}%; "
            f"{overlay.rendered_vertex_count:,}/{overlay.raw_vertex_count:,} vertices. "
            f"{variability.n_quiet_samples}/{variability.sample_times_s.size} quiet "
            f"samples at <= {self._quiet_threshold.value():g} "
            f"{variability.display_unit} {_METRIC_LABELS[variability.metric]}. "
            f"Adequacy: {counts[ESTIMABLE]} estimable, "
            f"{counts[RANK_DEFICIENT]} rank-deficient, "
            f"{counts[INSUFFICIENT_SAMPLES]} insufficient, "
            f"{counts[INVALID_COVARIANCE]} invalid; "
            f"{variability.unavailable_count} unavailable. "
            f"Ranked intervals: {ranked}. {interpretation} "
            f"Frame: {overlay.coordinate_frame}; alignment: common simulation time. "
            "Sparse yellow 2σ principal-axis glyphs are not confidence ellipsoids. "
            + (
                f"Cyan surfaces show {len(mesh.sample_indices)} estimable "
                "Gaussian position-content ellipsoids (not mean CIs). "
                if mesh is not None
                else "Confidence-ellipsoid surfaces are off. "
            )
            + "Drag to rotate; scroll to zoom."
        )
        handles, labels = axes.get_legend_handles_labels()
        handles.append(principal_spread_legend())
        labels.append(handles[-1].get_label())
        if mesh is not None and variability.confidence_level is not None:
            handles.append(confidence_ellipsoid_legend(variability.confidence_level))
            labels.append(handles[-1].get_label())
        axes.legend(handles, labels, loc="best", fontsize=8)
        self._canvas.draw_idle()
        draw_variability_timeline(self._variability_canvas, variability)


__all__ = ["ArcOverlayView", "DatasetScatterView"]
