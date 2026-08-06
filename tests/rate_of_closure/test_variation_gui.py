"""PyQt6 GUI smoke tests for the Variation tab (epic #4120, V3).

Headless-safe. Covers: tab construction and tooltips on every new
input, noise-row add/remove and mode switching, plan build/load round
trips, a full worker-thread study populating all result views,
cancellation, and dataset export from the tab.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.club import get_club  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.simulation import SimulationConfig  # noqa: E402
from rate_of_closure.ui.pyqt6.variation_tab import VariationTab  # noqa: E402
from rate_of_closure.ui.pyqt6.variation_worker import VariationWorker  # noqa: E402
from shared.python.swing_sim.variation import (  # noqa: E402
    CATEGORY_DELIVERY,
    CATEGORY_LAUNCH,
    CATEGORY_SWING,
    NoiseSpec,
    VariationPlan,
    keys_for_mode,
)
from shared.python.swing_sim.variation.dataset_io import read_json  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_BALL = f"{CATEGORY_LAUNCH}.ball_speed_mph"


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = VariationTab()
    qtbot.addWidget(widget)
    yield widget
    widget.stop()


def _fast_launch_plan(n_runs: int = 12) -> VariationPlan:
    return VariationPlan(
        mode="launch",
        noise=(NoiseSpec(_BALL, scale=1.0),),
        n_runs=n_runs,
        seed=1,
    )


class TestConstruction:
    def test_every_input_carries_hover_guidance(self, tab: VariationTab) -> None:
        widgets = [
            tab._mode_combo,
            tab._base_combo,
            tab._flight_combo,
            tab._runs_spin,
            tab._seed_spin,
            tab._sens_check,
            tab._forgiveness_check,
            tab._chip_target_yd,
            tab._run_button,
            tab._cancel_button,
            tab._export_csv,
            tab._export_json,
        ]
        for row in tab._rows:
            widgets.extend([row.variable, row.distribution, row.scale, row.clip])
        for widget in widgets:
            assert widget.toolTip(), f"missing tooltip on {widget}"

    def test_variable_picker_covers_the_mode_registry(self, tab: VariationTab) -> None:
        row = tab._rows[0]
        keys = tuple(row.variable.itemData(i) for i in range(row.variable.count()))
        assert keys == keys_for_mode("delivery")

    def test_mode_switch_repopulates_rows(self, tab: VariationTab) -> None:
        tab._mode_combo.setCurrentIndex(2)  # launch
        row = tab._rows[0]
        keys = tuple(row.variable.itemData(i) for i in range(row.variable.count()))
        assert keys == keys_for_mode("launch")

    def test_add_and_remove_rows_keeps_at_least_one(self, tab: VariationTab) -> None:
        tab._add_row()
        assert len(tab._rows) == 2
        tab._remove_row(tab._rows[1])
        assert len(tab._rows) == 1
        tab._remove_row(tab._rows[0])
        assert len(tab._rows) == 1  # refused, status message instead


class TestPlanRoundTrip:
    def test_build_plan_reflects_the_editors(self, tab: VariationTab) -> None:
        tab._runs_spin.setValue(33)
        tab._seed_spin.setValue(7)
        row = tab._rows[0]
        index = row.variable.findData(f"{CATEGORY_DELIVERY}.face_angle_deg")
        row.variable.setCurrentIndex(index)
        row.distribution.setCurrentText("uniform")
        row.scale.setValue(2.5)
        plan = tab.build_plan()
        assert plan.mode == "delivery"
        assert plan.n_runs == 33 and plan.seed == 7
        spec = plan.noise[0]
        assert spec.variable_key == f"{CATEGORY_DELIVERY}.face_angle_deg"
        assert spec.distribution == "uniform" and spec.scale == 2.5

    def test_load_plan_round_trips_including_base_and_truncation(
        self, tab: VariationTab
    ) -> None:
        plan = VariationPlan(
            mode="launch",
            base_variables={_BALL: 155.0},
            noise=(
                NoiseSpec(_BALL, scale=2.0, lower=140.0, upper=170.0),
                NoiseSpec(
                    f"{CATEGORY_LAUNCH}.spin_rpm",
                    distribution="triangular",
                    scale=150.0,
                ),
            ),
            n_runs=44,
            seed=9,
            flight_model="waterloo_penner",
        )
        tab.load_plan(plan)
        assert tab.build_plan() == plan

    def test_explorer_scenario_base_carries_speed_and_offsets(
        self, tab: VariationTab
    ) -> None:
        tab._base_combo.setCurrentIndex(1)
        plan = tab.build_plan()
        key = f"{CATEGORY_DELIVERY}.clubhead_speed_mps"
        assert plan.base_variables[key] == pytest.approx(113.0 * 0.44704)


class TestRunAndResults:
    def test_full_study_populates_every_results_view(
        self, qtbot, tab: VariationTab
    ) -> None:  # type: ignore[no-untyped-def]
        tab.load_plan(_fast_launch_plan())
        with _wait_done(qtbot, tab):
            pass
        dataset = tab.dataset()
        assert dataset is not None and dataset.n_success == 12
        assert tab._summary_table.rowCount() == len(dataset.output_names)
        assert tab._sensitivity_table.rowCount() == 1
        assert tab._spearman_table.columnCount() == len(dataset.output_names)
        assert tab._ensemble_scatter._canvas.axes.collections
        scatter_keys = {
            tab._ensemble_scatter._y_combo.itemData(index)
            for index in range(tab._ensemble_scatter._y_combo.count())
        }
        assert "output:carry_m" in scatter_keys

    def test_cancel_before_start_reports_cancelled(
        self, qtbot, tab: VariationTab
    ) -> None:  # type: ignore[no-untyped-def]
        worker = VariationWorker(_fast_launch_plan(200), compute_sensitivity=False)
        worker.cancel()
        with qtbot.waitSignal(worker.cancelled, timeout=15_000):
            worker.start()
        worker.wait(10_000)

    def test_swing_study_populates_trace_scatter_and_arc_views(
        self, qtbot, tab: VariationTab
    ) -> None:  # type: ignore[no-untyped-def]
        yaw = f"{CATEGORY_SWING}.yaw_deg"
        tab.load_plan(
            VariationPlan(
                mode="swing",
                noise=(NoiseSpec(yaw, distribution="uniform", scale=0.2),),
                n_runs=3,
                seed=2,
            )
        )
        tab.set_simulation_config(
            SimulationConfig(
                scenario=ImpactScenario(clubhead_speed_mph=30.0),
                club=get_club("Sand Wedge"),
                source_kind="double_pendulum",
                swing_duration_s=0.05,
            )
        )

        with _wait_done(qtbot, tab):
            pass

        assert tab.ensemble_result() is not None
        assert tab._sensitivity_table.rowCount() == 1
        assert tab._ensemble_scatter._canvas.axes.collections
        assert len(tab._arc_overlay._canvas.axes.lines) >= 4
        assert tab._arc_overlay._variability_canvas.axes.lines
        assert "3/3 trials" in tab._arc_overlay._status.text()
        assert "Hits 3 · No Impact 0 · Failures 0 · Landings 3" in (
            tab._landing._axes.get_title()
        )
        assert tab._export_trace_csv.isEnabled()
        assert tab._export_ensemble_json.isEnabled()
        assert tab._forgiveness_view.summary() is not None
        assert tab._forgiveness_view._scatter._canvas.axes.collections
        assert "illustrative" in tab._forgiveness_view.scope_text().lower()
        assert tab._export_forgiveness_csv.isEnabled()
        assert tab._export_forgiveness_json.isEnabled()

    def test_export_json_round_trips_from_the_tab(
        self, qtbot, tab: VariationTab, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        tab.load_plan(_fast_launch_plan())
        with _wait_done(qtbot, tab):
            pass
        target = tmp_path / "study.json"
        monkeypatch.setattr(
            "rate_of_closure.ui.pyqt6.variation_tab.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: (str(target), "JSON (*.json)")),
        )
        tab._on_export_json()
        loaded = read_json(target)
        assert loaded.plan == tab.dataset().plan
        assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == 1


class _wait_done:
    """Context manager: click Run and wait for the worker to finish."""

    def __init__(self, qtbot, tab: VariationTab) -> None:  # type: ignore[no-untyped-def]
        self._qtbot = qtbot
        self._tab = tab

    def __enter__(self) -> None:
        tab = self._tab
        tab._on_run()
        assert tab._worker is not None
        with self._qtbot.waitSignal(tab._worker.finished, timeout=60_000):
            pass

    def __exit__(self, *exc: object) -> None:
        return None
