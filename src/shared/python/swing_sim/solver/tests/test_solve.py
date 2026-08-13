"""Physics/behaviour tests for the robust multi-start solve driver."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.solver import (
    CancelledError,
    ImpactGoal,
    ProgressReport,
    VariablePartition,
    achieved_quantities,
    solve,
)

pytestmark = pytest.mark.physics


def _recovery_setup() -> tuple[VariablePartition, dict[str, float], ImpactGoal]:
    """Known variables -> outputs -> goal targeting those outputs."""
    partition = VariablePartition(
        free={
            "clubhead_speed_mps": (30.0, 55.0),
            "dynamic_loft_deg": (5.0, 20.0),
            "face_angle_deg": (-8.0, 8.0),
        }
    )
    truth = partition.assemble(np.array([46.0, 13.5, 1.5]))
    probe = ImpactGoal.of(ball_speed_mph=1.0)
    achieved = achieved_quantities(truth, partition, probe)
    goal = ImpactGoal.of(
        ball_speed_mph=achieved["ball_speed_mph"],
        launch_angle_deg=achieved["launch_angle_deg"],
        launch_azimuth_deg=achieved["launch_azimuth_deg"],
    )
    return partition, truth, goal


class TestExactRecovery:
    def test_recovers_generating_variables_from_cold_start(self) -> None:
        partition, truth, goal = _recovery_setup()
        result = solve(goal, partition, n_starts=4, seed=1)
        assert result.converged
        assert result.residual_norm < 1e-6
        assert result.variables["clubhead_speed_mps"] == pytest.approx(
            truth["clubhead_speed_mps"], abs=1e-2
        )
        assert result.variables["dynamic_loft_deg"] == pytest.approx(
            truth["dynamic_loft_deg"], abs=1e-2
        )
        assert result.variables["face_angle_deg"] == pytest.approx(
            truth["face_angle_deg"], abs=1e-2
        )
        for error in result.per_goal_errors.values():
            assert abs(error) < 1e-4

    def test_result_diagnostics_are_complete(self) -> None:
        partition, _, goal = _recovery_setup()
        result = solve(goal, partition, n_starts=3, seed=2)
        assert result.free_names == partition.free_names
        assert result.x.shape == (3,)
        assert len(result.starts) == 3
        assert {s.seed for s in result.starts} == {0, 1, 2}
        assert result.n_evals == sum(s.n_evals for s in result.starts)
        assert result.n_evals > 0
        assert result.elapsed_s >= 0.0
        assert result.cost == pytest.approx(
            0.5 * result.residual_norm**2, rel=1e-6, abs=1e-9
        )


class TestUnderdeterminedAndConflicting:
    def test_underdetermined_returns_valid_flat_residual_solution(self) -> None:
        """Two free variables, one goal: any exact solution is acceptable."""
        partition = VariablePartition(
            free={
                "clubhead_speed_mps": (30.0, 55.0),
                "dynamic_loft_deg": (5.0, 20.0),
            }
        )
        goal = ImpactGoal.of(ball_speed_mph=150.0)
        result = solve(goal, partition, n_starts=3, seed=3)
        assert result.residual_norm < 1e-6
        lo, hi = partition.bounds_arrays()
        assert np.all(result.x >= lo) and np.all(result.x <= hi)
        assert result.achieved["ball_speed_mph"] == pytest.approx(150.0, abs=1e-3)

    def test_conflicting_goals_report_honest_nonzero_residual(self) -> None:
        """Face controls both azimuth and face-angle goals set in conflict."""
        partition = VariablePartition(
            free={"face_angle_deg": (-8.0, 8.0)},
            fixed={"clubhead_speed_mps": 45.0, "dynamic_loft_deg": 12.0},
        )
        goal = ImpactGoal.of(face_angle_deg=5.0, launch_azimuth_deg=-5.0)
        result = solve(goal, partition, n_starts=3, seed=4)
        assert result.residual_norm > 1.0  # genuinely unattainable
        errors = result.per_goal_errors
        # Best compromise: both goals miss in opposite directions.
        assert errors["face_angle_deg"] < 0.0
        assert errors["launch_azimuth_deg"] > 0.0
        # The compromise face sits strictly between the two pulls.
        assert -5.0 < result.variables["face_angle_deg"] < 5.0


class TestBoundsAndValidation:
    def test_solution_respects_bounds_under_unreachable_goal(self) -> None:
        partition = VariablePartition(
            free={"clubhead_speed_mps": (30.0, 40.0)},
            fixed={"dynamic_loft_deg": 12.0},
        )
        goal = ImpactGoal.of(ball_speed_mph=250.0)
        result = solve(goal, partition, n_starts=2, seed=5)
        assert result.x[0] == pytest.approx(40.0, abs=1e-6)
        assert result.residual_norm > 0.0

    def test_empty_free_set_raises(self) -> None:
        partition = VariablePartition(free={}, fixed={"face_angle_deg": 0.0})
        with pytest.raises(ContractViolationError):
            solve(ImpactGoal.of(ball_speed_mph=150.0), partition)

    def test_goal_variable_cannot_be_free_and_fixed(self) -> None:
        with pytest.raises(ContractViolationError):
            VariablePartition(
                free={"dynamic_loft_deg": (5.0, 20.0)},
                fixed={"dynamic_loft_deg": 12.0},
            )

    def test_bad_x0_rejected(self) -> None:
        partition = VariablePartition(free={"face_angle_deg": (-5.0, 5.0)})
        goal = ImpactGoal.of(face_angle_deg=1.0)
        with pytest.raises(ContractViolationError):
            solve(goal, partition, x0=np.array([1.0, 2.0]))


class TestCancellationAndProgress:
    def test_pre_set_cancel_event_raises(self) -> None:
        partition = VariablePartition(free={"face_angle_deg": (-5.0, 5.0)})
        goal = ImpactGoal.of(face_angle_deg=1.0)
        event = threading.Event()
        event.set()
        with pytest.raises(CancelledError):
            solve(goal, partition, cancel_event=event)

    def test_cancel_during_solve_is_honoured(self) -> None:
        partition = VariablePartition(
            free={
                "clubhead_speed_mps": (30.0, 55.0),
                "dynamic_loft_deg": (5.0, 20.0),
            }
        )
        goal = ImpactGoal.of(ball_speed_mph=150.0, launch_angle_deg=13.0)
        event = threading.Event()
        calls = {"n": 0}

        def cancelling_cb(report: ProgressReport) -> None:
            calls["n"] += 1
            event.set()

        try:
            result = solve(
                goal,
                partition,
                n_starts=6,
                progress_cb=cancelling_cb,
                cancel_event=event,
                n_workers=1,
            )
        except CancelledError:
            return  # cancelled before any start completed: valid outcome
        # Otherwise at least one start was cut short by the event.
        assert calls["n"] >= 1
        assert any(s.cancelled for s in result.starts)

    def test_progress_reports_have_movement_optimizer_shape(self) -> None:
        partition, _, goal = _recovery_setup()
        reports: list[ProgressReport] = []
        solve(goal, partition, n_starts=3, seed=6, progress_cb=reports.append)
        assert reports, "expected at least one progress report"
        report = reports[-1]
        assert report.iteration > 0
        assert report.best_cost <= report.cost or report.best_cost >= 0.0
        assert report.elapsed_s >= 0.0
        assert isinstance(report.cost_history, list)
        assert isinstance(report.is_stalled, bool)
