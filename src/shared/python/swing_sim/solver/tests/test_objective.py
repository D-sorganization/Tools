"""Unit/physics tests for the pure candidate-evaluation objective."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.solver import (
    EvaluationConfig,
    ImpactGoal,
    VariablePartition,
    achieved_quantities,
    evaluate_candidate,
    residuals,
)

FAST_SWING = EvaluationConfig(swing_duration_s=0.8, swing_dt_s=5e-3)


def _delivery_partition() -> VariablePartition:
    return VariablePartition(
        free={
            "clubhead_speed_mps": (30.0, 55.0),
            "dynamic_loft_deg": (5.0, 20.0),
        }
    )


@pytest.mark.unit
class TestEvaluateCandidate:
    def test_zero_residuals_at_generating_variables(self) -> None:
        """Goal built from a candidate's own outputs scores ~zero."""
        partition = _delivery_partition()
        variables = partition.assemble(np.array([44.0, 12.0]))
        achieved = achieved_quantities(
            variables,
            partition,
            ImpactGoal.of(ball_speed_mph=1.0, spin_rpm=1.0),
        )
        goal = ImpactGoal.of(
            ball_speed_mph=achieved["ball_speed_mph"],
            spin_rpm=achieved["spin_rpm"],
        )
        res = evaluate_candidate(variables, partition, goal)
        assert res.shape == (2,)
        np.testing.assert_allclose(res, 0.0, atol=1e-9)

    def test_pure_function_deterministic_and_non_mutating(self) -> None:
        partition = _delivery_partition()
        variables = partition.assemble(np.array([44.0, 12.0]))
        snapshot = dict(variables)
        goal = ImpactGoal.of(ball_speed_mph=150.0, launch_angle_deg=13.0)
        first = evaluate_candidate(variables, partition, goal)
        second = evaluate_candidate(variables, partition, goal)
        np.testing.assert_array_equal(first, second)
        assert variables == snapshot

    def test_residual_ordering_matches_goal_items(self) -> None:
        partition = _delivery_partition()
        goal = ImpactGoal.of(spin_rpm=99999.0, club_path_deg=0.0)
        res = residuals(np.array([44.0, 12.0]), partition, goal)
        # Canonical order: club_path_deg (exact -> 0) before spin_rpm.
        assert res[0] == pytest.approx(0.0, abs=1e-12)
        assert res[1] < 0.0

    def test_weight_and_scale_applied(self) -> None:
        partition = _delivery_partition()
        variables = partition.assemble(np.array([44.0, 12.0]))
        achieved = achieved_quantities(
            variables, partition, ImpactGoal.of(ball_speed_mph=1.0)
        )
        target = achieved["ball_speed_mph"] - 3.0  # 3 mph high, scale 1 mph
        base = evaluate_candidate(
            variables, partition, ImpactGoal.of(ball_speed_mph=target)
        )
        double = evaluate_candidate(
            variables, partition, ImpactGoal.of(ball_speed_mph=(target, 2.0))
        )
        assert base[0] == pytest.approx(3.0)
        assert double[0] == pytest.approx(6.0)

    def test_delivery_only_goal_skips_impact(self) -> None:
        partition = _delivery_partition()
        variables = partition.assemble(np.array([44.0, 12.0]))
        achieved = achieved_quantities(
            variables, partition, ImpactGoal.of(dynamic_loft_deg=10.0)
        )
        assert "ball_speed_mph" not in achieved
        assert achieved["dynamic_loft_deg"] == 12.0


@pytest.mark.physics
class TestPhysicsSignatures:
    def test_more_speed_more_ball_speed(self) -> None:
        partition = _delivery_partition()
        goal = ImpactGoal.of(ball_speed_mph=1.0)
        slow = achieved_quantities(
            partition.assemble(np.array([38.0, 12.0])), partition, goal
        )
        fast = achieved_quantities(
            partition.assemble(np.array([50.0, 12.0])), partition, goal
        )
        assert fast["ball_speed_mph"] > slow["ball_speed_mph"]

    def test_open_face_launches_right_with_fade_axis(self) -> None:
        partition = VariablePartition(
            free={"face_angle_deg": (-10.0, 10.0)},
            fixed={"clubhead_speed_mps": 45.0, "dynamic_loft_deg": 12.0},
        )
        goal = ImpactGoal.of(launch_azimuth_deg=0.0, spin_axis_deg=0.0)
        open_face = achieved_quantities(
            partition.assemble(np.array([4.0])), partition, goal
        )
        square = achieved_quantities(
            partition.assemble(np.array([0.0])), partition, goal
        )
        assert open_face["launch_azimuth_deg"] > square["launch_azimuth_deg"]
        # Face open to a straight path tilts spin to the fade side (+).
        assert open_face["spin_axis_deg"] > square["spin_axis_deg"]

    def test_more_loft_more_launch_and_spin(self) -> None:
        partition = _delivery_partition()
        goal = ImpactGoal.of(launch_angle_deg=0.0, spin_rpm=0.0)
        low = achieved_quantities(
            partition.assemble(np.array([44.0, 8.0])), partition, goal
        )
        high = achieved_quantities(
            partition.assemble(np.array([44.0, 16.0])), partition, goal
        )
        assert high["launch_angle_deg"] > low["launch_angle_deg"]
        assert high["spin_rpm"] > low["spin_rpm"]

    def test_carry_reported_when_goal_needs_flight(self) -> None:
        partition = _delivery_partition()
        achieved = achieved_quantities(
            partition.assemble(np.array([44.0, 12.0])),
            partition,
            ImpactGoal.of(carry_m=200.0),
        )
        assert 20.0 < achieved["carry_m"] < 400.0

    def test_swing_mode_derives_target_directed_delivery(self) -> None:
        partition = VariablePartition(
            free={"swing_impact_time_offset_s": (-0.05, 0.05)},
            fixed={"dynamic_loft_deg": 12.0},
            use_swing_source=True,
        )
        goal = ImpactGoal.of(club_path_deg=0.0, attack_angle_deg=0.0)
        achieved = achieved_quantities(
            partition.assemble(np.array([0.0])), partition, goal, FAST_SWING
        )
        # Gravity-driven pendulum: modest speed, roughly target-line path.
        assert abs(achieved["club_path_deg"]) < 10.0
        assert abs(achieved["attack_angle_deg"]) < 45.0

    def test_swing_yaw_tilts_club_path(self) -> None:
        base_part = VariablePartition(
            free={"swing_yaw_deg": (-15.0, 15.0)},
            fixed={"dynamic_loft_deg": 12.0},
            use_swing_source=True,
        )
        goal = ImpactGoal.of(club_path_deg=0.0)
        yawed = achieved_quantities(
            base_part.assemble(np.array([8.0])), base_part, goal, FAST_SWING
        )
        square = achieved_quantities(
            base_part.assemble(np.array([0.0])), base_part, goal, FAST_SWING
        )
        assert yawed["club_path_deg"] != pytest.approx(square["club_path_deg"], abs=1.0)
