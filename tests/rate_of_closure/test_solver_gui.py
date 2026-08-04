"""PyQt6 GUI smoke tests for the Solver panel (epic #4103, #4109/#4110).

Headless-safe. Covers: spec tables matching the solver package's
variable/goal names, sourced hover guidance on every new input, a full
worker-thread solve populating the results view, clean cancellation
(pre-set and mid-run), the apply round-trip landing solved variables in
the simulation session, and DbC validation errors surfacing as friendly
status messages instead of tracebacks.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.model import MPH_PER_MPS, ImpactScenario  # noqa: E402
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab  # noqa: E402
from rate_of_closure.ui.pyqt6.solver_panel import SolverPanel  # noqa: E402
from rate_of_closure.ui.pyqt6.solver_specs import (  # noqa: E402
    GOAL_SPECS,
    VARIABLE_SPECS,
)
from rate_of_closure.ui.pyqt6.solver_worker import SolverWorker  # noqa: E402
from shared.python.swing_sim.solver.goals import (  # noqa: E402
    DELIVERY_VARIABLE_DEFAULTS,
    GOAL_QUANTITIES,
    SWING_VARIABLE_DEFAULTS,
    ImpactGoal,
    VariablePartition,
)
from shared.python.swing_sim.solver.solve import SolverResult  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _easy_goal_and_partition() -> tuple[ImpactGoal, VariablePartition]:
    """The pinned easy case: hit 150 mph ball speed with clubhead speed."""
    goal = ImpactGoal.of(ball_speed_mph=150.0)
    partition = VariablePartition(free={"clubhead_speed_mps": (30.0, 70.0)})
    return goal, partition


def _configure_easy_case(panel: SolverPanel) -> None:
    """Drive the editors to the pinned easy case (defaults already close)."""
    for name, row in panel._goal_rows.items():
        row.enabled.setChecked(name == "ball_speed_mph")
    panel._goal_rows["ball_speed_mph"].target.setValue(150.0)
    for name, row in panel._var_rows.items():
        row.optimize.setChecked(name == "clubhead_speed_mps")
        row.fix.setChecked(name != "clubhead_speed_mps")
    panel._var_rows["clubhead_speed_mps"].low.setValue(30.0)
    panel._var_rows["clubhead_speed_mps"].high.setValue(70.0)
    panel._starts_spin.setValue(2)


@pytest.fixture
def panel(qtbot):  # type: ignore[no-untyped-def]
    widget = SolverPanel()
    qtbot.addWidget(widget)
    yield widget
    widget.stop()


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = SimulationTab()
    qtbot.addWidget(widget)
    widget.set_scenario(ImpactScenario(clubhead_speed_mph=113.0))
    yield widget
    widget.stop()


class TestSpecs:
    def test_goal_specs_cover_the_solver_quantities_exactly(self) -> None:
        assert tuple(spec.name for spec in GOAL_SPECS) == GOAL_QUANTITIES

    def test_variable_specs_cover_both_modes_exactly(self) -> None:
        names = {spec.name for spec in VARIABLE_SPECS}
        assert names == set(DELIVERY_VARIABLE_DEFAULTS) | set(SWING_VARIABLE_DEFAULTS)
        swing_only = {spec.name for spec in VARIABLE_SPECS if spec.swing_only}
        assert swing_only == set(SWING_VARIABLE_DEFAULTS)

    def test_every_input_carries_sourced_guidance(self, panel) -> None:  # type: ignore[no-untyped-def]
        widgets = [panel._swing_check, panel._starts_spin]
        for row in panel._goal_rows.values():
            widgets += [row.enabled, row.target, row.weight]
        for row in panel._var_rows.values():
            widgets += [row.optimize, row.low, row.high, row.fixed_value]
        for widget in widgets:
            assert "Suggested range" in widget.toolTip(), widget
            assert "Source:" in widget.toolTip(), widget


class TestEditors:
    def test_mode_toggle_swaps_derived_for_swing_rows(self, panel) -> None:  # type: ignore[no-untyped-def]
        panel._swing_check.setChecked(True)
        partition = panel.build_partition()
        assert partition.use_swing_source
        names = set(partition.free) | set(partition.fixed)
        assert "swing_side_tilt_deg" in names
        assert "clubhead_speed_mps" not in names
        panel._swing_check.setChecked(False)
        partition = panel.build_partition()
        names = set(partition.free) | set(partition.fixed)
        assert "clubhead_speed_mps" in names
        assert "swing_side_tilt_deg" not in names

    def test_no_goals_checked_surfaces_a_friendly_message(self, panel) -> None:  # type: ignore[no-untyped-def]
        for row in panel._goal_rows.values():
            row.enabled.setChecked(False)
        panel._on_run()
        assert panel._status.text().startswith("Cannot solve:")
        assert panel._run_button.isEnabled()

    def test_inverted_bounds_surface_a_friendly_message(self, panel) -> None:  # type: ignore[no-untyped-def]
        _configure_easy_case(panel)
        row = panel._var_rows["clubhead_speed_mps"]
        row.low.setValue(60.0)
        row.high.setValue(40.0)
        panel._on_run()
        assert panel._status.text().startswith("Cannot solve:")
        assert panel._run_button.isEnabled()


class TestWorker:
    def test_worker_completes_and_reports_the_pinned_solution(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        goal, partition = _easy_goal_and_partition()
        worker = SolverWorker(goal, partition, n_starts=2)
        with qtbot.waitSignal(worker.succeeded, timeout=60000) as blocker:
            worker.start()
        worker.wait(10_000)
        result = blocker.args[0]
        assert result.converged
        # Pinned solution (matches the web parity test): ~45.82 m/s.
        assert result.variables["clubhead_speed_mps"] == pytest.approx(45.825, abs=0.05)
        assert result.achieved["ball_speed_mph"] == pytest.approx(150.0, abs=0.1)

    def test_preset_cancel_event_cancels_cleanly(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        goal, partition = _easy_goal_and_partition()
        worker = SolverWorker(goal, partition, n_starts=4)
        worker.cancel()
        with qtbot.waitSignal(worker.cancelled, timeout=30000):
            worker.start()
        worker.wait(10_000)
        assert worker.cancel_event.is_set()

    def test_midrun_cancel_finishes_without_error(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        goal = ImpactGoal.of(carry_m=230.0)  # flight per eval: slower run
        partition = VariablePartition(free={"clubhead_speed_mps": (30.0, 70.0)})
        worker = SolverWorker(goal, partition, n_starts=8)
        outcomes: list[str] = []
        worker.succeeded.connect(lambda _r: outcomes.append("succeeded"))
        worker.cancelled.connect(lambda: outcomes.append("cancelled"))
        worker.failed.connect(lambda _m: outcomes.append("failed"))
        with qtbot.waitSignal(worker.finished, timeout=120000):
            worker.start()
            worker.cancel()
        worker.wait(10_000)
        assert worker.cancel_event.is_set()
        assert outcomes in (["succeeded"], ["cancelled"])


class TestPanelRun:
    def test_panel_run_populates_results_and_enables_apply(self, panel, qtbot) -> None:  # type: ignore[no-untyped-def]
        _configure_easy_case(panel)
        assert not panel._apply_button.isEnabled()
        panel._on_run()
        assert panel._worker is not None
        with qtbot.waitSignal(panel._worker.succeeded, timeout=60000):
            pass
        panel._worker.wait(10_000)
        qtbot.waitUntil(panel._apply_button.isEnabled, timeout=10000)
        assert panel._table.rowCount() == 1
        assert panel._table.item(0, 0).text() == "Ball Speed"
        assert "Solved variables" in panel._summary.text()
        assert panel._starts_tree.topLevelItemCount() == 2
        assert "converged" in panel._status.text()


class TestApply:
    def test_delivery_apply_round_trip_lands_in_the_session(self, tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        panel = tab.solver_panel()
        _configure_easy_case(panel)
        panel._var_rows["impact_offset_toe_mm"].fixed_value.setValue(4.0)
        panel._on_run()
        with qtbot.waitSignal(panel._worker.succeeded, timeout=60000):
            pass
        panel._worker.wait(10_000)
        qtbot.waitUntil(panel._apply_button.isEnabled, timeout=10000)
        result = panel.result()
        with qtbot.waitSignal(tab.runCompleted, timeout=30000):
            panel._apply_button.click()
        config = tab.config()
        assert config.source_kind == "manual"
        assert config.scenario.clubhead_speed_mph == pytest.approx(
            result.variables["clubhead_speed_mps"] * MPH_PER_MPS
        )
        assert config.scenario.impact_offset_toe_mm == pytest.approx(4.0)
        assert tab.last_run() is not None

    def test_swing_apply_selects_pendulum_and_drives_tilts(self, tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        variables = {
            "face_angle_deg": 0.0,
            "dynamic_loft_deg": 10.5,
            "lie_deg": 0.0,
            "impact_offset_toe_mm": 2.0,
            "impact_offset_high_mm": -1.0,
            "swing_yaw_deg": 5.0,
            "swing_side_tilt_deg": -40.0,
            "swing_forward_tilt_deg": -3.0,
            "swing_impact_time_offset_s": 0.01,
            "swing_damping_shoulder": 0.4,
            "swing_damping_wrist": 0.25,
        }
        result = SolverResult(
            variables=variables,
            free_names=("swing_side_tilt_deg",),
            x=np.array([-40.0]),
            achieved={},
            per_goal_errors={},
            residual_norm=0.0,
            cost=0.0,
            converged=True,
            n_evals=1,
            elapsed_s=0.0,
            starts=(),
        )
        with qtbot.waitSignal(tab.runCompleted, timeout=60000):
            run = tab.apply_solver_solution(result, True)
        assert run is not None
        assert tab.source_kind() == "double_pendulum"
        assert tab.plane().yaw_deg == pytest.approx(5.0)
        assert tab.plane().side_tilt_deg == pytest.approx(-40.0)
        config = tab.config()
        assert config.scenario.impact_offset_toe_mm == pytest.approx(2.0)
        # The solved impact-time offset shifted tau off the auto instant.
        assert config.impact_time_s is not None
