"""Unit tests for goal and variable-partition types (DbC validation)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.solver import (
    DELIVERY_VARIABLE_DEFAULTS,
    GOAL_QUANTITIES,
    SWING_VARIABLE_DEFAULTS,
    GoalTerm,
    ImpactGoal,
    VariablePartition,
)

pytestmark = pytest.mark.unit


class TestGoalTerm:
    def test_defaults_weight_to_one(self) -> None:
        assert GoalTerm(10.0).weight == 1.0

    @pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_weight(self, weight: float) -> None:
        with pytest.raises(ContractViolationError):
            GoalTerm(10.0, weight)

    def test_rejects_non_finite_target(self) -> None:
        with pytest.raises(ContractViolationError):
            GoalTerm(float("nan"))


class TestImpactGoal:
    def test_requires_at_least_one_target(self) -> None:
        with pytest.raises(ContractViolationError):
            ImpactGoal()

    def test_of_accepts_floats_tuples_and_terms(self) -> None:
        goal = ImpactGoal.of(
            ball_speed_mph=150.0,
            spin_rpm=(2600.0, 0.5),
            carry_m=GoalTerm(230.0, 2.0),
        )
        items = dict(goal.items())
        assert items["ball_speed_mph"] == GoalTerm(150.0)
        assert items["spin_rpm"] == GoalTerm(2600.0, 0.5)
        assert items["carry_m"] == GoalTerm(230.0, 2.0)

    def test_from_mapping_accepts_dynamic_ui_targets(self) -> None:
        targets: dict[str, float | tuple[float, float]] = {
            "ball_speed_mph": 150.0,
            "spin_rpm": (2600.0, 0.5),
        }

        goal = ImpactGoal.from_mapping(targets)

        assert dict(goal.items()) == {
            "ball_speed_mph": GoalTerm(150.0),
            "spin_rpm": GoalTerm(2600.0, 0.5),
        }

    def test_of_rejects_unknown_quantity(self) -> None:
        with pytest.raises(ContractViolationError):
            ImpactGoal.of(smash_factor=1.5)

    def test_rejects_raw_float_field(self) -> None:
        with pytest.raises(ContractViolationError):
            ImpactGoal(ball_speed_mph=150.0)  # type: ignore[arg-type]

    def test_items_follow_canonical_order(self) -> None:
        goal = ImpactGoal.of(carry_m=200.0, club_path_deg=2.0, spin_rpm=2500.0)
        names = [name for name, _ in goal.items()]
        assert names == sorted(names, key=GOAL_QUANTITIES.index)

    def test_needs_flags(self) -> None:
        assert not ImpactGoal.of(club_path_deg=1.0).needs_launch
        assert ImpactGoal.of(ball_speed_mph=140.0).needs_launch
        assert not ImpactGoal.of(ball_speed_mph=140.0).needs_flight
        assert ImpactGoal.of(carry_m=200.0).needs_flight
        assert ImpactGoal.of(carry_m=200.0).needs_launch


class TestVariablePartition:
    def test_free_and_fixed_must_be_disjoint(self) -> None:
        with pytest.raises(ContractViolationError):
            VariablePartition(
                free={"face_angle_deg": (-5.0, 5.0)},
                fixed={"face_angle_deg": 1.0},
            )

    def test_unknown_variable_rejected(self) -> None:
        with pytest.raises(ContractViolationError):
            VariablePartition(free={"smash_factor": (1.0, 2.0)})

    def test_swing_names_rejected_in_delivery_mode(self) -> None:
        with pytest.raises(ContractViolationError):
            VariablePartition(free={"swing_yaw_deg": (-10.0, 10.0)})

    def test_derived_delivery_names_rejected_in_swing_mode(self) -> None:
        with pytest.raises(ContractViolationError):
            VariablePartition(
                free={"clubhead_speed_mps": (30.0, 50.0)},
                use_swing_source=True,
            )

    def test_swing_mode_accepts_swing_and_face_variables(self) -> None:
        partition = VariablePartition(
            free={"swing_yaw_deg": (-10.0, 10.0)},
            fixed={"dynamic_loft_deg": 12.0},
            use_swing_source=True,
        )
        assert partition.free_names == ("swing_yaw_deg",)

    @pytest.mark.parametrize(
        "bounds", [(1.0, 1.0), (2.0, 1.0), (float("nan"), 1.0), (0.0, float("inf"))]
    )
    def test_bad_bounds_rejected(self, bounds: tuple[float, float]) -> None:
        with pytest.raises(ContractViolationError):
            VariablePartition(free={"face_angle_deg": bounds})

    def test_non_finite_fixed_rejected(self) -> None:
        with pytest.raises(ContractViolationError):
            VariablePartition(
                free={"face_angle_deg": (-5.0, 5.0)},
                fixed={"lie_deg": float("nan")},
            )

    def test_empty_free_set_rejected_at_bounds(self) -> None:
        partition = VariablePartition(free={}, fixed={"face_angle_deg": 1.0})
        with pytest.raises(ContractViolationError):
            partition.bounds_arrays()

    def test_assemble_fills_defaults_fixed_and_free(self) -> None:
        partition = VariablePartition(
            free={"clubhead_speed_mps": (30.0, 50.0)},
            fixed={"face_angle_deg": 2.0},
        )
        variables = partition.assemble(np.array([41.0]))
        assert variables["clubhead_speed_mps"] == 41.0
        assert variables["face_angle_deg"] == 2.0
        assert variables["lie_deg"] == DELIVERY_VARIABLE_DEFAULTS["lie_deg"]

    def test_assemble_swing_mode_swaps_variable_sets(self) -> None:
        partition = VariablePartition(
            free={"swing_impact_time_offset_s": (-0.05, 0.05)},
            use_swing_source=True,
        )
        variables = partition.assemble(np.array([0.01]))
        assert "clubhead_speed_mps" not in variables
        assert variables["swing_impact_time_offset_s"] == 0.01
        assert variables["swing_yaw_deg"] == SWING_VARIABLE_DEFAULTS["swing_yaw_deg"]

    def test_assemble_rejects_wrong_shape_and_non_finite(self) -> None:
        partition = VariablePartition(free={"face_angle_deg": (-5.0, 5.0)})
        with pytest.raises(ContractViolationError):
            partition.assemble(np.array([1.0, 2.0]))
        with pytest.raises(ContractViolationError):
            partition.assemble(np.array([float("nan")]))
