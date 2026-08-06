"""Interactive PyQt views over canonical ensemble plot data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas
from rate_of_closure.variation.plot_data import (
    ArcOverlayData,
    CohortAvailability,
    EnsemblePlotDataset,
    ScalarPlotVariable,
    scalar_plot_variables,
)
from rate_of_closure.variation.simulation_types import TrialEvaluationStatus
from shared.python.swing_sim.variation import VariationDataset

_COHORT_COLORS = {
    TrialEvaluationStatus.EVALUATED_HIT: "#2f8bd6",
    TrialEvaluationStatus.EVALUATED_NO_IMPACT: "#eb9f3c",
    TrialEvaluationStatus.NUMERICAL_FAILURE: "#d35f5f",
}


class _PlotCanvas(LifecycleSafeFigureCanvas):
    """Lifecycle-safe Matplotlib canvas exposing its one axes to tests/views."""

    def __init__(self, *, projection: str | None = None) -> None:
        figure = Figure(figsize=(6.0, 4.5), layout="constrained")
        super().__init__(figure)
        self.axes = figure.add_subplot(111, projection=projection)

    def apply_theme(self) -> None:
        """Apply the current Qt palette to figure, axes, and labels."""
        window = self.palette().window().color().name()
        text = self.palette().text().color().name()
        self.figure.set_facecolor(window)
        self.axes.set_facecolor(self.palette().window().color().lighter(105).name())
        self.axes.tick_params(colors=text, labelsize=8)
        axes = [self.axes.xaxis, self.axes.yaxis]
        if hasattr(self.axes, "zaxis"):
            axes.append(self.axes.zaxis)
        for axis in axes:
            axis.label.set_color(text)
        self.axes.title.set_color(text)


class DatasetScatterView(QWidget):
    """Selectable input/impact/shot scatter with cohort availability counts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: EnsemblePlotDataset | None = None
        self._variation: VariationDataset | None = None
        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
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
        self._canvas = _PlotCanvas()
        selectors = QFormLayout()
        selectors.addRow("Horizontal Axis", self._x_combo)
        selectors.addRow("Vertical Axis", self._y_combo)
        layout = QVBoxLayout(self)
        layout.addLayout(selectors)
        layout.addWidget(self._availability)
        layout.addWidget(self._canvas, stretch=1)
        self._x_combo.currentIndexChanged.connect(self._redraw)
        self._y_combo.currentIndexChanged.connect(self._redraw)
        self._clear()

    def set_plot_dataset(self, dataset: EnsemblePlotDataset) -> None:
        """Populate selectors and render the default paired scatter."""
        self._dataset = dataset
        self._variation = dataset.result.variation
        self._set_variables(dataset.variables)
        self._redraw()

    def set_variation_dataset(self, dataset: VariationDataset) -> None:
        """Render scalar-only delivery/launch studies with failure accounting."""
        self._dataset = None
        self._variation = dataset
        self._set_variables(scalar_plot_variables(dataset))
        self._redraw()

    def _set_variables(self, variables: tuple[ScalarPlotVariable, ...]) -> None:
        """Populate both axis selectors from one canonical variable list."""
        for combo in (self._x_combo, self._y_combo):
            combo.blockSignals(True)
            combo.clear()
            for variable in variables:
                combo.addItem(_axis_label(variable), variable.key)
            combo.blockSignals(False)
        self._select_default(self._x_combo, "input:")
        self._select_default(self._y_combo, "output:carry_m")

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
                    label=_cohort_label(cohort),
                    color=_COHORT_COLORS[cohort],
                    edgecolors="none",
                )
        axes.set_xlabel(_axis_label(scatter.x_variable))
        axes.set_ylabel(_axis_label(scatter.y_variable))
        axes.set_title("Variation Effects Across Typed Trial Outcomes")
        if axes.collections:
            axes.legend(loc="best", fontsize=8)
        self._availability.setText(
            _availability_text(scatter.cohort_summaries.values())
        )
        self._canvas.draw_idle()

    def _redraw_scalar(self, dataset: VariationDataset) -> None:
        """Render finite paired rows from a scalar-only variation dataset."""
        variables = scalar_plot_variables(dataset)
        by_key = {variable.key: variable for variable in variables}
        x_variable = by_key[str(self._x_combo.currentData())]
        y_variable = by_key[str(self._y_combo.currentData())]
        x_values = _dataset_values(dataset, x_variable)
        y_values = _dataset_values(dataset, y_variable)
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        axes = self._canvas.axes
        axes.clear()
        self._canvas.apply_theme()
        axes.scatter(
            x_values[finite],
            y_values[finite],
            s=20,
            alpha=0.72,
            color="#2f8bd6",
            edgecolors="none",
            label="Evaluated",
        )
        axes.set_xlabel(_axis_label(x_variable))
        axes.set_ylabel(_axis_label(y_variable))
        axes.set_title("Variation Effects Across Evaluated Trials")
        if axes.collections:
            axes.legend(loc="best", fontsize=8)
        unavailable = int(dataset.plan.n_runs - np.count_nonzero(finite))
        failed = int(dataset.plan.n_runs - dataset.n_success)
        self._availability.setText(
            f"Evaluated: {np.count_nonzero(finite)}/{dataset.plan.n_runs} plotted"
            f" · {unavailable} paired values unavailable · {failed} failures. "
            "This scalar evaluator has no geometric no-impact cohort."
        )
        self._canvas.draw_idle()


class ArcOverlayView(QWidget):
    """Rotatable all-trial 3-D swing arcs with a median reference trace."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: EnsemblePlotDataset | None = None
        self._point_combo = QComboBox()
        self._point_combo.setToolTip(
            "Choose which modeled point to overlay across every variation trial."
        )
        self._status = QLabel("Run a trace-capable swing variation study.")
        self._status.setWordWrap(True)
        self._canvas = _PlotCanvas(projection="3d")
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Modeled Point"))
        controls.addWidget(self._point_combo, stretch=1)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._status)
        layout.addWidget(self._canvas, stretch=1)
        self._point_combo.currentIndexChanged.connect(self._redraw)
        self._clear()

    def set_plot_dataset(self, dataset: EnsemblePlotDataset) -> None:
        """Populate modeled points and render every valid trial arc."""
        self._dataset = dataset
        self._point_combo.blockSignals(True)
        self._point_combo.clear()
        for point_id in dataset.result.traces.point_ids:
            self._point_combo.addItem(_point_label(point_id), point_id)
        self._point_combo.blockSignals(False)
        clubhead = self._point_combo.findData("swing.clubhead.reference")
        self._point_combo.setCurrentIndex(max(clubhead, 0))
        self._redraw()

    def _clear(self) -> None:
        self._canvas.axes.clear()
        self._canvas.apply_theme()
        self._canvas.axes.set_title("All-Trial Swing Arc Overlay")
        self._set_axes()
        self._canvas.draw_idle()

    def _redraw(self, *_args: object) -> None:
        if self._dataset is None or self._point_combo.currentIndex() < 0:
            return
        overlay = self._dataset.arc_overlay(str(self._point_combo.currentData()))
        axes = self._canvas.axes
        axes.clear()
        self._canvas.apply_theme()
        for positions, valid, cohort in zip(
            overlay.positions_m,
            overlay.sample_valid,
            overlay.cohorts,
            strict=True,
        ):
            if np.any(valid):
                axes.plot(
                    positions[valid, 0],
                    positions[valid, 2],
                    positions[valid, 1],
                    color=_COHORT_COLORS[cohort],
                    linewidth=0.8,
                    alpha=0.34,
                )
        reference = overlay.reference_positions_m
        axes.plot(
            reference[:, 0],
            reference[:, 2],
            reference[:, 1],
            color="#f2f4f8",
            linewidth=2.2,
            label="Median Reference",
        )
        axes.set_title(f"All Trials — {_point_label(overlay.point_id)}")
        self._set_axes()
        _equal_3d_axes(axes, overlay)
        valid_trials = sum(bool(np.any(row)) for row in overlay.sample_valid)
        self._status.setText(
            f"{valid_trials}/{len(overlay.cohorts)} trials shown; "
            f"{overlay.rendered_vertex_count:,}/{overlay.raw_vertex_count:,} vertices. "
            f"Frame: {overlay.coordinate_frame}. Drag to rotate; scroll to zoom."
        )
        axes.legend(loc="best", fontsize=8)
        self._canvas.draw_idle()

    def _set_axes(self) -> None:
        """Apply the app-frame labels while plotting y-up as visual z."""
        self._canvas.axes.set_xlabel("Target, x [m]")
        self._canvas.axes.set_ylabel("Right, z [m]")
        self._canvas.axes.set_zlabel("Up, y [m]")  # type: ignore[attr-defined]


def _axis_label(variable: ScalarPlotVariable) -> str:
    """Return a compact, unit-bearing plot label."""
    return f"{variable.label} [{variable.unit}]" if variable.unit else variable.label


def _cohort_label(cohort: TrialEvaluationStatus) -> str:
    """Return the concise UI label for a typed trial cohort."""
    return {
        TrialEvaluationStatus.EVALUATED_HIT: "Hit",
        TrialEvaluationStatus.EVALUATED_NO_IMPACT: "No Impact",
        TrialEvaluationStatus.NUMERICAL_FAILURE: "Numerical Failure",
    }[cohort]


def _availability_text(summaries: Iterable[CohortAvailability]) -> str:
    """Describe plotted/unavailable counts without selection bias."""
    return " · ".join(
        f"{_cohort_label(summary.cohort)}: {summary.plotted}/{summary.total} plotted"
        + (f", {summary.unavailable} unavailable" if summary.unavailable else "")
        for summary in summaries
    )


def _point_label(point_id: str) -> str:
    """Convert a stable spatial point ID into a title-case label."""
    return point_id.rsplit(".", 1)[-1].replace("_", " ").title()


def _dataset_values(
    dataset: VariationDataset,
    variable: ScalarPlotVariable,
) -> np.ndarray:
    """Return one all-row scalar column without silently dropping failures."""
    source, name = variable.key.split(":", 1)
    if source == "input":
        return np.asarray(dataset.inputs[:, dataset.input_names.index(name)])
    return np.asarray(dataset.outputs[:, dataset.output_names.index(name)])


def _equal_3d_axes(axes, overlay: ArcOverlayData) -> None:  # type: ignore[no-untyped-def]
    """Set one physical scale across all three spatial axes."""
    finite = overlay.positions_m[np.isfinite(overlay.positions_m).all(axis=-1)]
    if finite.size == 0:
        return
    plot_xyz = finite[:, [0, 2, 1]]
    low = np.min(plot_xyz, axis=0)
    high = np.max(plot_xyz, axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 1e-6)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)
    axes.set_box_aspect((1.0, 1.0, 1.0))


__all__ = ["ArcOverlayView", "DatasetScatterView"]
