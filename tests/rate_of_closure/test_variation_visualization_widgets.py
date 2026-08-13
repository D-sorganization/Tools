"""Headless widget tests for universal variation visualizations."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")
from PyQt6.QtWidgets import QLabel  # noqa: E402

from rate_of_closure.club import get_club  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.simulation import SimulationConfig  # noqa: E402
from rate_of_closure.ui.pyqt6.variation_distribution_matrix import (  # noqa: E402
    DistributionMatrixView,
)
from rate_of_closure.ui.pyqt6.variation_results import LandingCanvas  # noqa: E402
from rate_of_closure.ui.pyqt6.variation_visualizations import (  # noqa: E402
    ArcOverlayView,
    DatasetScatterView,
)
from rate_of_closure.variation.plot_data import (  # noqa: E402
    build_ensemble_plot_dataset,
)
from rate_of_closure.variation.simulation_adapter import (  # noqa: E402
    build_simulation_ensemble_request,
    run_simulation_ensemble,
)
from shared.python.swing_sim.variation import (  # noqa: E402
    CATEGORY_SWING,
    ELLIPSOID_VOLUME,
    NoiseSpec,
    VariationDataset,
    VariationPlan,
    dispersion_ellipse,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_YAW = f"{CATEGORY_SWING}.yaw_deg"


def _plot_dataset(n_runs: int = 3):  # type: ignore[no-untyped-def]
    plan = VariationPlan(
        mode="swing",
        noise=(NoiseSpec(_YAW, distribution="uniform", scale=0.2),),
        n_runs=n_runs,
        seed=8,
    )
    base = SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=30.0),
        club=get_club("Sand Wedge"),
        source_kind="double_pendulum",
        swing_duration_s=0.05,
    )
    result = run_simulation_ensemble(build_simulation_ensemble_request(plan, base))
    return build_ensemble_plot_dataset(result)


def test_scatter_view_exposes_inputs_impact_and_shot_axes(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    plot_dataset = _plot_dataset()
    view = DatasetScatterView()
    qtbot.addWidget(view)

    view.set_plot_dataset(plot_dataset)

    keys = {view._x_combo.itemData(index) for index in range(view._x_combo.count())}
    assert f"input:{_YAW}" in keys
    assert "output:clubhead_speed_mps" in keys
    assert "output:carry_m" in keys
    assert view._canvas.axes.collections
    assert "Hit" in view._availability.text()
    png_path = tmp_path / "scatter.png"
    svg_path = tmp_path / "scatter.svg"
    csv_path = tmp_path / "scatter.csv"
    definition_path = tmp_path / "scatter.plot.json"
    view._exports.write_png(png_path)
    view._exports.write_svg(svg_path)
    view._exports.write_csv(csv_path)
    view._exports.write_definition(definition_path)
    assert png_path.stat().st_size > 1000
    assert "<svg" in svg_path.read_text(encoding="utf-8")
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4
    assert rows[0].startswith("trial_index,outcome,input:")
    assert view._table.rowCount() == 3
    assert view._table.accessibleName() == "Selected scatter trial data"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    assert definition["schema_version"] == 2


def test_landing_canvas_counts_only_paired_finite_coordinates(qtbot) -> None:  # type: ignore[no-untyped-def]
    """The plot title must count exactly the rows that reach the canvas."""
    plan = VariationPlan(
        mode="swing",
        noise=(NoiseSpec(_YAW, distribution="uniform", scale=0.2),),
        n_runs=4,
        seed=9,
    )
    dataset = VariationDataset(
        plan=plan,
        input_names=(_YAW,),
        inputs=np.zeros((4, 1)),
        output_names=("carry_m", "lateral_m"),
        outputs=np.array(
            (
                (100.0, 10.0),
                (1000.0, np.nan),
                (np.nan, 100.0),
                (300.0, 30.0),
            )
        ),
        success=np.ones(4, dtype=bool),
    )
    view = LandingCanvas()
    qtbot.addWidget(view)

    view.set_dataset(dataset, dispersion_ellipse(dataset))

    assert "2 landings / 4 trials" in view._axes.get_title()
    plotted = view._axes.collections[0].get_offsets()
    assert plotted.shape == (2, 2)
    assert plotted.tolist() == [[10.0, 100.0], [30.0, 300.0]]


def test_arc_overlay_draws_every_valid_trial_and_reference(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    plot_dataset = _plot_dataset()
    view = ArcOverlayView()
    qtbot.addWidget(view)

    view.set_plot_dataset(plot_dataset)

    overlay = plot_dataset.arc_overlay(view._point_combo.currentData())
    valid_trials = sum(row.any() for row in overlay.sample_valid)
    assert len(view._canvas.axes.lines) >= valid_trials + 1
    assert "3/3 trials" in view._status.text()
    assert "quiet samples" in view._status.text()
    assert view._canvas.axes.get_xlabel() == "Target, x [m]"
    assert view._variability_canvas.axes.lines
    assert view._variability_canvas.axes.get_ylabel() == "RMS Position Radius [mm]"
    definition_path = tmp_path / "arc.plot.json"
    view._exports.write_definition(definition_path)
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    assert definition["coordinate_frame"] == "app_frame:x_target,y_up,z_right"
    assert definition["dispersion_metric"] == "rms-radius"
    assert definition["dispersion_unit"] == "m"
    assert definition["quiet_threshold"] == pytest.approx(0.005)
    assert definition["confidence_level"] is None
    assert definition["min_quiet_duration_s"] == pytest.approx(0.0)
    assert definition["camera_yaw_deg"] is not None
    timeline_svg = tmp_path / "variability.svg"
    timeline_definition = tmp_path / "variability.plot.json"
    view._variability_exports.write_svg(timeline_svg)
    view._variability_exports.write_definition(timeline_definition)
    assert "<svg" in timeline_svg.read_text(encoding="utf-8")
    assert (
        json.loads(timeline_definition.read_text(encoding="utf-8"))["plot_type"]
        == "geometric_variability"
    )

    assert not view._confidence.isEnabled()
    view._metric_combo.setCurrentIndex(view._metric_combo.findData(ELLIPSOID_VOLUME))
    assert view._confidence.isEnabled()
    assert "mm³" in view._quiet_threshold.suffix()
    assert "Gaussian position-content region" in view._status.text()
    assert "not a confidence region for the mean" in view._status.text()
    assert "Sparse 2σ principal-axis glyphs" in view._status.text()

    view._filters._source.setCurrentIndex(1)
    view._filters._band.setCurrentIndex(1)
    view._filters._phase.setValue(50)
    selected_count = view._filters.trial_indices(plot_dataset).size
    assert f"{selected_count}/3 trials shown" in view._status.text()
    assert "phase 0–50%" in view._status.text()
    filtered_path = tmp_path / "filtered-arc.plot.json"
    view._exports.write_definition(filtered_path)
    filtered_definition = json.loads(filtered_path.read_text(encoding="utf-8"))
    assert filtered_definition["phase_end_fraction"] == pytest.approx(0.5)
    assert filtered_definition["perturbation_source_key"] == _YAW
    assert filtered_definition["perturbation_band"] == "Lower Third"


def test_trial_selection_links_scatter_matrix_and_arc_views(qtbot) -> None:  # type: ignore[no-untyped-def]
    plot_dataset = _plot_dataset()
    scatter = DatasetScatterView()
    arcs = ArcOverlayView()
    matrix = DistributionMatrixView()
    qtbot.addWidget(scatter)
    qtbot.addWidget(arcs)
    qtbot.addWidget(matrix)
    scatter.selectionChanged.connect(arcs.set_selected_trial)
    scatter.selectionChanged.connect(matrix.set_selected_trial)
    arcs.selectionChanged.connect(scatter.set_selected_trial)
    arcs.selectionChanged.connect(matrix.set_selected_trial)
    matrix.selectionChanged.connect(scatter.set_selected_trial)
    matrix.selectionChanged.connect(arcs.set_selected_trial)
    scatter.set_plot_dataset(plot_dataset)
    arcs.set_plot_dataset(plot_dataset)
    matrix.set_plot_dataset(plot_dataset)

    scatter._trial_combo.setCurrentIndex(scatter._trial_combo.findData(1))

    assert arcs._trial_combo.currentData() == 1
    assert matrix._table.currentRow() == 1
    assert max(line.get_linewidth() for line in arcs._canvas.axes.lines) >= 2.8
    assert any(
        collection.get_linewidths().max() >= 1.8
        for collection in scatter._canvas.axes.collections
    )

    matrix._table.cellClicked.emit(0, 0)
    assert scatter._trial_combo.currentData() == 0
    assert arcs._trial_combo.currentData() == 0


def test_replacing_result_clears_and_bounds_linked_trial_selection(qtbot) -> None:  # type: ignore[no-untyped-def]
    """A smaller rerun must not retain an impossible trial identity."""
    larger = _plot_dataset(3)
    smaller = _plot_dataset(1)
    scatter = DatasetScatterView()
    arcs = ArcOverlayView()
    matrix = DistributionMatrixView()
    for view in (scatter, arcs, matrix):
        qtbot.addWidget(view)
        view.set_plot_dataset(larger)
        view.set_selected_trial(2)

    for view in (scatter, arcs, matrix):
        view.set_plot_dataset(smaller)
        assert view._selected_trial is None

    assert scatter._trial_combo.currentData() is None
    assert arcs._trial_combo.currentData() is None
    assert matrix._table.currentRow() == -1
    for view in (scatter, arcs, matrix):
        with pytest.raises(ValueError, match="trial_index"):
            view.set_selected_trial(1)


def test_dispersion_controls_have_accessible_names_and_label_buddies(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = ArcOverlayView()
    qtbot.addWidget(view)
    expected = {
        "Dispersion Metric": (view._metric_combo, "Dispersion metric"),
        "Confidence": (view._confidence, "Dispersion confidence percent"),
        "Quiet Threshold": (view._quiet_threshold, "Quiet-zone metric threshold"),
        "Min Duration": (view._min_duration, "Minimum quiet duration seconds"),
        "Min Samples": (view._min_samples, "Minimum quiet samples"),
    }

    labels = {label.text(): label for label in view.findChildren(QLabel)}
    for text, (control, accessible_name) in expected.items():
        assert control.accessibleName() == accessible_name
        assert labels[text].buddy() is control


def test_distribution_matrix_draws_and_exports_selected_raw_rows(
    qtbot, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    view = DistributionMatrixView()
    qtbot.addWidget(view)

    view.set_plot_dataset(_plot_dataset())

    assert len(view._figure.axes) == 16
    assert any(axis.patches for axis in view._figure.axes)
    assert any(axis.collections for axis in view._figure.axes)
    assert "canonical exports retain every miss/failure row" in view._status.text()
    assert view._table.rowCount() == 3
    assert view._table.accessibleName() == "Selected scatter matrix trial data"
    csv_path = tmp_path / "matrix.csv"
    definition_path = tmp_path / "matrix.plot.json"
    view._exports.write_csv(csv_path)
    view._exports.write_definition(definition_path)
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4
    assert rows[0].startswith("trial_index,outcome,")
    assert "evaluated_hit" in rows[1]
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    assert definition["plot_type"] == "distribution_matrix"
    assert len(definition["variable_keys"]) == 4
