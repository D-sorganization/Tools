"""Contract test pinning the public API surface of swing_sim.solver.

Downstream consumers (UI integration wave, web backend, UpstreamDrift)
import from the subpackage facade only; this test fails loudly when the
surface changes so removals are always deliberate.
"""

from __future__ import annotations

import dataclasses

import pytest

import shared.python.swing_sim.solver as solver

EXPECTED_PUBLIC_API = {
    "DELIVERY_VARIABLE_DEFAULTS",
    "GOAL_QUANTITIES",
    "SWING_DERIVED_VARIABLES",
    "SWING_VARIABLE_DEFAULTS",
    "CancelledError",
    "EvaluationConfig",
    "GoalTerm",
    "ImpactGoal",
    "ProgressCallback",
    "ProgressReport",
    "SolverResult",
    "StartSummary",
    "VariablePartition",
    "achieved_quantities",
    "detect_stall",
    "evaluate_candidate",
    "residuals",
    "solve",
}

pytestmark = pytest.mark.contract


def test_public_api_is_pinned() -> None:
    assert set(solver.__all__) == EXPECTED_PUBLIC_API


def test_facade_exports_resolve() -> None:
    for name in EXPECTED_PUBLIC_API:
        assert getattr(solver, name) is not None


def test_goal_quantities_are_pinned() -> None:
    assert solver.GOAL_QUANTITIES == (
        "club_path_deg",
        "face_angle_deg",
        "attack_angle_deg",
        "dynamic_loft_deg",
        "ball_speed_mph",
        "launch_angle_deg",
        "launch_azimuth_deg",
        "spin_rpm",
        "spin_axis_deg",
        "carry_m",
    )


def test_variable_registries_are_pinned() -> None:
    assert set(solver.DELIVERY_VARIABLE_DEFAULTS) == {
        "clubhead_speed_mps",
        "club_path_deg",
        "face_angle_deg",
        "attack_angle_deg",
        "dynamic_loft_deg",
        "lie_deg",
        "impact_offset_toe_mm",
        "impact_offset_high_mm",
    }
    assert set(solver.SWING_VARIABLE_DEFAULTS) == {
        "swing_yaw_deg",
        "swing_side_tilt_deg",
        "swing_forward_tilt_deg",
        "swing_impact_time_offset_s",
        "swing_damping_shoulder",
        "swing_damping_wrist",
    }
    assert solver.SWING_DERIVED_VARIABLES == (
        "clubhead_speed_mps",
        "club_path_deg",
        "attack_angle_deg",
    )


def test_progress_report_matches_movement_optimizer_shape() -> None:
    """Field-for-field copy of movement_optimizer's ProgressReport."""
    names = [f.name for f in dataclasses.fields(solver.ProgressReport)]
    assert names == [
        "iteration",
        "cost",
        "best_cost",
        "improvement_pct",
        "elapsed_s",
        "cost_history",
        "is_stalled",
        "stall_reason",
    ]


def test_solver_result_diagnostic_fields_are_pinned() -> None:
    names = {f.name for f in dataclasses.fields(solver.SolverResult)}
    assert names == {
        "variables",
        "free_names",
        "x",
        "achieved",
        "per_goal_errors",
        "residual_norm",
        "cost",
        "converged",
        "n_evals",
        "elapsed_s",
        "starts",
    }


def test_parent_facade_untouched() -> None:
    """The parent swing_sim facade must not re-export solver names yet."""
    import shared.python.swing_sim as swing_sim

    assert "solve" not in getattr(swing_sim, "__all__", ())
