"""Focused UI tests for the wind-strategy workflow panel."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import replace

import pytest

from rate_of_closure.ui.pyqt6.flight_explorer_tab import FlightExplorerTab
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow
from rate_of_closure.ui.pyqt6.wind_strategy_launch import (
    WindStrategyLaunchContext,
    WindStrategySettings,
    build_strategy_request,
    scalar_ensemble_csv,
)
from rate_of_closure.ui.pyqt6.wind_strategy_panel import WindStrategyPanel
from rate_of_closure.variation.wind_strategy_plot_adapter import (
    build_wind_strategy_plot_dataset,
)
from shared.python.swing_sim.flight import (
    LaunchConditions,
    StrategyAnalysisConfig,
    analyze_wind_strategies,
)
from shared.python.swing_sim.solver import (
    SpatialTarget,
    SurfaceCircleTolerance,
    TargetPoint,
)


def _context() -> WindStrategyLaunchContext:
    return WindStrategyLaunchContext(
        LaunchConditions.from_imperial(145.0, 12.0, 2600.0),
        SpatialTarget(
            label="Test green",
            kind="landing_area",
            point=TargetPoint(210.0, 0.0, 4.0),
            tolerance=SurfaceCircleTolerance(8.0),
            elevation_source="course_surface",
            ground_source="test.flat",
        ),
        "waterloo_penner",
    )


def _settings() -> WindStrategySettings:
    return WindStrategySettings(
        trials=2,
        true_speed_mps=4.0,
        true_from_bearing_deg=90.0,
        speed_bias_mps=-0.5,
        speed_std_mps=0.6,
        bearing_bias_deg=2.0,
        bearing_std_deg=3.0,
        correlation=0.25,
        aim_gain_deg_per_mps=0.1,
        seed=4199,
    )


def test_request_uses_current_launch_and_canonical_landing_target() -> None:
    context = _context()

    request = build_strategy_request(context, _settings())

    assert request.strategies[0].launch is context.launch
    assert request.target.forward_m == 210.0
    assert request.target.right_m == 4.0
    assert request.analysis.target_radius_m == 8.0
    assert request.uncertainty.estimate_error.correlation == 0.25


def test_generic_csv_preserves_all_rows_variables_and_nulls() -> None:
    request = build_strategy_request(_context(), _settings())
    request = replace(
        request,
        analysis=StrategyAnalysisConfig(
            model_name=request.analysis.model_name,
            max_time_s=0.01,
            failure_cost=request.analysis.failure_cost,
            target_radius_m=request.analysis.target_radius_m,
        ),
    )
    dataset = build_wind_strategy_plot_dataset(
        request, analyze_wind_strategies(request)
    )
    first = dataset.rows[0]
    values = {**first.values, "information_cost_delta": -1.25}
    attributes = {
        **(first.attributes or {}),
        "equals_formula": "=SUM(1,2)",
        "plus_formula": "+cmd",
        "minus_formula": "-cmd",
        "at_formula": "@cmd",
    }
    dataset = replace(
        dataset,
        rows=(replace(first, values=values, attributes=attributes), *dataset.rows[1:]),
    )

    rows = list(csv.DictReader(io.StringIO(scalar_ensemble_csv(dataset))))

    assert len(rows) == len(dataset.rows)
    assert set(variable.key for variable in dataset.variables).issubset(rows[0])
    assert {"row_id", "trial_index", "series_id", "cohort"}.issubset(rows[0])
    assert "attribute:actual_failure_reason" in rows[0]
    assert rows[0]["actual_landing_forward_m"] == ""
    assert rows[0]["attribute:actual_failure_reason"] == "ground not reached"
    assert rows[0]["attribute:equals_formula"] == "'=SUM(1,2)"
    assert rows[0]["attribute:plus_formula"] == "'+cmd"
    assert rows[0]["attribute:minus_formula"] == "'-cmd"
    assert rows[0]["attribute:at_formula"] == "'@cmd"
    assert rows[0]["information_cost_delta"] == "-1.25"

    first_key, second_key = (item.key for item in dataset.variables[:2])
    variables = (
        replace(dataset.variables[0], key="=formula_key"),
        replace(dataset.variables[1], key="'=formula_key"),
        *dataset.variables[2:],
    )
    unsafe_rows = tuple(
        replace(
            row,
            values={
                "=formula_key": row.values[first_key],
                "'=formula_key": row.values[second_key],
                **{
                    key: value
                    for key, value in row.values.items()
                    if key not in (first_key, second_key)
                },
            },
            attributes={**(row.attributes or {}), "=attribute_key": "safe"},
        )
        for row in dataset.rows
    )
    unsafe_csv = scalar_ensemble_csv(
        replace(dataset, variables=variables, rows=unsafe_rows)
    )
    headers = next(csv.reader(io.StringIO(unsafe_csv)))
    assert "'=formula_key" in headers
    assert "''=formula_key" in headers
    assert len(headers) == len(set(headers))
    assert "attribute:=attribute_key" in headers


def test_panel_exposes_controls_status_summary_scatter_and_export(qtbot) -> None:  # type: ignore[no-untyped-def]
    panel = WindStrategyPanel(_context)
    qtbot.addWidget(panel)
    panel.apply_result(
        build_strategy_request(_context(), _settings()),
        analyze_wind_strategies(build_strategy_request(_context(), _settings())),
    )

    assert panel._run.accessibleName() == "Run Wind Strategy Analysis"
    assert panel._input_grid.columnCount() == 4
    assert panel._input_grid.rowCount() == 5
    assert [panel._workspace.tabText(index) for index in range(2)] == [
        "Setup",
        "Results",
    ]
    assert panel._workspace.currentIndex() == 1
    assert panel._status.text().startswith("Completed")
    assert panel._summary.rowCount() == 1
    assert panel._x_axis.count() > 20
    assert panel._y_axis.count() == panel._x_axis.count()
    assert panel._export.isEnabled()
    basis = panel._basis.text()
    assert panel._basis.accessibleName() == "Wind Strategy Calculation Basis"
    for expected in (
        "model waterloo_penner",
        "2 paired trials",
        "seed 4199",
        "target +210.000 m forward, +4.000 m right",
        "hold radius 8.000 m",
        "maximum time 10 s",
        "time step 0.01 s",
        "failure cost 100",
        "CVaR alpha 0.9",
        "Current Launch +0.100 deg/(m/s)",
    ):
        assert expected in basis


def test_panel_scatter_has_zoom_pan_autofit_and_legend_controls(qtbot) -> None:  # type: ignore[no-untyped-def]
    panel = WindStrategyPanel(_context)
    qtbot.addWidget(panel)
    request = build_strategy_request(_context(), _settings())
    panel.apply_result(request, analyze_wind_strategies(request))

    assert panel._plot.toolbar().actions()
    assert panel._plot.zoom_percent() == 100
    panel._plot.zoom_in()
    assert panel._plot.zoom_percent() == 125
    panel._plot.auto_fit()
    assert panel._plot.zoom_percent() == 100
    panel._plot.set_legend_placement("hidden")
    assert panel._plot.legend_placement() == "hidden"
    legends = [axis.get_legend() for axis in panel._plot.figure().axes]
    assert all(legend is None or not legend.get_visible() for legend in legends)
    toolbar = panel._plot.toolbar()
    toolbar.push_current()
    assert toolbar._nav_stack._elements
    panel._y_axis.setCurrentIndex(0)
    assert toolbar._nav_stack._elements == []


def test_panel_seed_round_trips_full_uint32_without_coercion(qtbot) -> None:  # type: ignore[no-untyped-def]
    panel = WindStrategyPanel(_context)
    qtbot.addWidget(panel)

    panel._seed.setText("4294967295")
    assert panel.settings().seed == 4_294_967_295

    panel._seed.setText("4294967296")
    with pytest.raises(ValueError, match="uint32"):
        panel.settings()


def test_completed_snapshot_is_invalidated_when_consumed_factors_change(qtbot) -> None:  # type: ignore[no-untyped-def]
    current = [_context()]
    panel = WindStrategyPanel(lambda: current[0])
    qtbot.addWidget(panel)
    request = build_strategy_request(current[0], _settings())
    panel.apply_result(request, analyze_wind_strategies(request))

    panel._spins["true_speed"].setValue(6.0)
    assert panel._dataset is None
    assert not panel._export.isEnabled()
    assert panel._summary.rowCount() == 0
    assert panel._basis.text() == "Calculation basis: no current result."
    assert panel._status.text().startswith("Wind inputs changed")
    invalidated_status = panel._status.text()
    panel._on_progress(1, 2)
    panel._on_failed("queued stale failure")
    assert panel._status.text() == invalidated_status

    panel.apply_result(request, analyze_wind_strategies(request))
    current[0] = replace(current[0], model_name="penner")
    qtbot.waitUntil(lambda: panel._dataset is None, timeout=1_000)
    assert panel._status.text().startswith("Launch, target, or flight model changed")


def test_panel_run_completes_asynchronously_and_restores_controls(qtbot) -> None:  # type: ignore[no-untyped-def]
    panel = WindStrategyPanel(_context)
    qtbot.addWidget(panel)
    panel._trials.setValue(1)

    panel._run.click()
    assert not panel._run.isEnabled()
    qtbot.waitUntil(panel._run.isEnabled, timeout=10_000)

    assert panel._run.isEnabled()
    assert panel._status.text().startswith("Completed 1 outcome")
    assert panel._summary.rowCount() == 1
    assert panel._worker is None
    panel.stop()


def test_panel_stop_cancels_and_joins_running_worker(qtbot) -> None:  # type: ignore[no-untyped-def]
    panel = WindStrategyPanel(_context)
    qtbot.addWidget(panel)

    class StubWorker:
        def __init__(self) -> None:
            self.cancelled = False
            self.wait_calls: list[int | None] = []

        def cancel(self) -> None:
            self.cancelled = True

        def wait(self, timeout: int | None = None) -> bool:
            self.wait_calls.append(timeout)
            return timeout is None

    worker = StubWorker()
    panel._worker = worker

    panel.stop()

    assert worker.cancelled
    assert worker.wait_calls == [10_000, None]
    assert panel._worker is None


def test_flight_explorer_supplies_live_launch_target_and_model(qtbot) -> None:  # type: ignore[no-untyped-def]
    explorer = FlightExplorerTab()
    qtbot.addWidget(explorer)
    explorer._direct_spins["launch_angle_deg"].setValue(18.0)
    explorer._spatial_target_panel.coordinate_edit("x").setText("205")

    context = explorer._wind_strategy_context()

    assert context.launch.launch_angle == pytest.approx(math.radians(18.0))
    assert context.target.point.app_coordinates_m[0] == 205.0
    assert context.model_name == explorer._model_combo.currentText()
    assert explorer._wind_strategy_panel in explorer.findChildren(WindStrategyPanel)
    request = build_strategy_request(context, _settings())
    explorer._wind_strategy_panel.apply_result(
        request, analyze_wind_strategies(request)
    )
    explorer._direct_spins["launch_angle_deg"].setValue(19.0)
    qtbot.waitUntil(
        lambda: explorer._wind_strategy_panel._dataset is None,
        timeout=1_000,
    )
    assert explorer._wind_strategy_panel._status.text().startswith("Launch, target")
    explorer.stop()


def test_main_window_close_explicitly_stops_flight_explorer(qtbot, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    window = RateOfClosureMainWindow()
    qtbot.addWidget(window)
    stopped: list[bool] = []
    monkeypatch.setattr(
        window._flight_explorer_tab, "stop", lambda: stopped.append(True)
    )

    window.close()

    assert stopped == [True]
