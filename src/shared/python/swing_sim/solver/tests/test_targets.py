"""Target-region geometry + solver integration tests (#4125 H7b)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.solver import (
    ImpactGoal,
    TargetRegion,
    VariablePartition,
    evaluate_candidate,
    solve,
)
from shared.python.swing_sim.solver.targets import CENTERING_WEIGHT

pytestmark = pytest.mark.physics


class TestGreenSignedDistance:
    def test_inside_boundary_outside(self) -> None:
        green = TargetRegion(kind="green", distance_m=200.0, radius_m=10.0)
        assert green.signed_distance(200.0, 0.0) == pytest.approx(-10.0)
        assert green.signed_distance(210.0, 0.0) == pytest.approx(0.0)
        assert green.signed_distance(215.0, 0.0) == pytest.approx(5.0)
        # Exact Euclidean off-axis: 6-8-10 triangle to the center.
        assert green.signed_distance(206.0, 8.0) == pytest.approx(0.0)
        assert green.contains(204.0, 4.0)
        assert not green.contains(212.0, 0.0)

    def test_lateral_offset_moves_the_circle(self) -> None:
        green = TargetRegion(
            kind="green", distance_m=150.0, radius_m=5.0, lateral_m=20.0
        )
        assert green.contains(150.0, 20.0)
        assert not green.contains(150.0, 0.0)
        assert green.signed_distance(150.0, 0.0) == pytest.approx(15.0)


class TestFairwaySignedDistance:
    def test_inside_boundary_outside(self) -> None:
        fw = TargetRegion(
            kind="fairway",
            distance_m=230.0,
            band_half_length_m=20.0,
            half_width_m=15.0,
        )
        assert fw.signed_distance(230.0, 0.0) == pytest.approx(-15.0)
        assert fw.signed_distance(230.0, 15.0) == pytest.approx(0.0)
        assert fw.signed_distance(250.0, 0.0) == pytest.approx(0.0)
        # Outside past the corner: exact Euclidean corner distance.
        assert fw.signed_distance(253.0, 19.0) == pytest.approx(5.0)
        # Outside along one axis only.
        assert fw.signed_distance(230.0, 25.0) == pytest.approx(10.0)
        assert fw.signed_distance(260.0, 5.0) == pytest.approx(10.0)

    def test_interior_signed_distance_is_nearest_edge(self) -> None:
        fw = TargetRegion(
            kind="fairway",
            distance_m=100.0,
            band_half_length_m=30.0,
            half_width_m=10.0,
        )
        # 4 m from the side edge, 20 m from the band edge -> -4.
        assert fw.signed_distance(110.0, 6.0) == pytest.approx(-4.0)


class TestResidual:
    def test_zero_outside_distance_inside_with_centering(self) -> None:
        green = TargetRegion(kind="green", distance_m=200.0, radius_m=10.0)
        assert green.residual_m(200.0, 0.0) == pytest.approx(0.0)
        # Inside off-center: only the centering term contributes.
        assert green.residual_m(205.0, 0.0) == pytest.approx(CENTERING_WEIGHT * 5.0)
        # Outside: distance-outside dominates + centering.
        expected = 10.0 + CENTERING_WEIGHT * 20.0
        assert green.residual_m(220.0, 0.0) == pytest.approx(expected)

    def test_validation_rejects_bad_geometry(self) -> None:
        with pytest.raises((ContractViolationError, ValueError)):
            TargetRegion(kind="green", distance_m=-5.0)
        with pytest.raises((ContractViolationError, ValueError)):
            TargetRegion(kind="green", distance_m=100.0, radius_m=0.0)
        with pytest.raises((ContractViolationError, ValueError)):
            TargetRegion(kind="lake", distance_m=100.0)  # type: ignore[arg-type]
        with pytest.raises((ContractViolationError, ValueError)):
            TargetRegion(kind="green", distance_m=math.nan)


class TestGoalIntegration:
    def test_region_only_goal_is_valid_and_needs_flight(self) -> None:
        green = TargetRegion(kind="green", distance_m=180.0, radius_m=12.0)
        goal = ImpactGoal.of(target_region=green)
        assert goal.needs_flight and goal.needs_launch
        assert goal.items() == ()

    def test_no_goal_at_all_is_rejected(self) -> None:
        with pytest.raises((ContractViolationError, ValueError)):
            ImpactGoal()

    def test_region_appends_one_residual(self) -> None:
        partition = VariablePartition(free={"clubhead_speed_mps": (30.0, 55.0)})
        variables = partition.assemble(np.array([45.0]))
        green = TargetRegion(kind="green", distance_m=180.0, radius_m=12.0)
        base = ImpactGoal.of(ball_speed_mph=150.0)
        with_region = ImpactGoal.of(ball_speed_mph=150.0, target_region=green)
        r_base = evaluate_candidate(variables, partition, base)
        r_region = evaluate_candidate(variables, partition, with_region)
        assert r_region.shape[0] == r_base.shape[0] + 1
        assert r_region[0] == pytest.approx(r_base[0])
        assert r_region[-1] >= 0.0


class TestOptimizeToTarget:
    def test_reaches_a_reachable_green_from_a_cold_start(self) -> None:
        """The optimizer lands a generously sized green with free speed
        and loft — the H7b acceptance case."""
        green = TargetRegion(kind="green", distance_m=170.0, radius_m=15.0)
        goal = ImpactGoal.of(target_region=green)
        partition = VariablePartition(
            free={
                "clubhead_speed_mps": (30.0, 55.0),
                "dynamic_loft_deg": (8.0, 18.0),
            }
        )
        result = solve(goal, partition, n_starts=3, seed=7)
        assert "target_region_m" in result.per_goal_errors
        # The achieved landing point holds the green (small tolerance
        # for the centering term trading against nothing else).
        assert result.achieved["target_distance_m"] < 1.0
        assert green.signed_distance(
            result.achieved["carry_m"], result.achieved["landing_lateral_m"]
        ) == pytest.approx(result.achieved["target_distance_m"])
