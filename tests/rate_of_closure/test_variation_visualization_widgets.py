"""Headless widget tests for universal variation visualizations."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.club import get_club  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.simulation import SimulationConfig  # noqa: E402
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


def test_scatter_view_exposes_inputs_impact_and_shot_axes(qtbot) -> None:  # type: ignore[no-untyped-def]
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


def test_arc_overlay_draws_every_valid_trial_and_reference(qtbot) -> None:  # type: ignore[no-untyped-def]
    plot_dataset = _plot_dataset()
    view = ArcOverlayView()
    qtbot.addWidget(view)

    view.set_plot_dataset(plot_dataset)

    overlay = plot_dataset.arc_overlay(view._point_combo.currentData())
    valid_trials = sum(row.any() for row in overlay.sample_valid)
    assert len(view._canvas.axes.lines) == valid_trials + 1
    assert "3/3 trials" in view._status.text()
    assert view._canvas.axes.get_xlabel() == "Target, x [m]"
