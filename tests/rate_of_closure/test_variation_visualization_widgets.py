"""Headless widget tests for universal variation visualizations."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.club import get_club  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.simulation import SimulationConfig  # noqa: E402
from rate_of_closure.ui.pyqt6.variation_distribution_matrix import (  # noqa: E402
    DistributionMatrixView,
)
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
    NoiseSpec,
    VariationPlan,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_YAW = f"{CATEGORY_SWING}.yaw_deg"


def _plot_dataset():  # type: ignore[no-untyped-def]
    plan = VariationPlan(
        mode="swing",
        noise=(NoiseSpec(_YAW, distribution="uniform", scale=0.2),),
        n_runs=3,
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
    definition_path = tmp_path / "scatter.plot.json"
    view._exports.write_png(png_path)
    view._exports.write_svg(svg_path)
    view._exports.write_definition(definition_path)
    assert png_path.stat().st_size > 1000
    assert "<svg" in svg_path.read_text(encoding="utf-8")
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    assert definition["schema_version"] == 1


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
    assert definition["quiet_threshold_m"] == pytest.approx(0.005)
    assert definition["camera_yaw_deg"] is not None


def test_trial_selection_links_scatter_and_arc_views(qtbot) -> None:  # type: ignore[no-untyped-def]
    plot_dataset = _plot_dataset()
    scatter = DatasetScatterView()
    arcs = ArcOverlayView()
    qtbot.addWidget(scatter)
    qtbot.addWidget(arcs)
    scatter.selectionChanged.connect(arcs.set_selected_trial)
    arcs.selectionChanged.connect(scatter.set_selected_trial)
    scatter.set_plot_dataset(plot_dataset)
    arcs.set_plot_dataset(plot_dataset)

    scatter._trial_combo.setCurrentIndex(scatter._trial_combo.findData(1))

    assert arcs._trial_combo.currentData() == 1
    assert max(line.get_linewidth() for line in arcs._canvas.axes.lines) >= 2.8
    assert any(
        collection.get_linewidths().max() >= 1.8
        for collection in scatter._canvas.axes.collections
    )


def test_distribution_matrix_draws_histograms_and_paired_scatter(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = DistributionMatrixView()
    qtbot.addWidget(view)

    view.set_plot_dataset(_plot_dataset())

    assert len(view._figure.axes) == 16
    assert any(axis.patches for axis in view._figure.axes)
    assert any(axis.collections for axis in view._figure.axes)
    assert "canonical exports retain every miss/failure row" in view._status.text()
