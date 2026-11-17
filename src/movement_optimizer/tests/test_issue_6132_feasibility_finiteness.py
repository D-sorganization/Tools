# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Issue #6132: non-finite or malformed data must not pass the feasibility check.

Before the fix, ``_check_solution_feasibility`` accepted a ``-inf`` cost and an
all-NaN joint trajectory (comparisons against NaN are False, so it counted
zero joint-limit violations). Invalid numbers are reported separately from
physical bound violations: the violation count stays physical-only.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from movement_optimizer.models import BodyModel, make_squat_config
from movement_optimizer.trajectory import (
    OptimizationResult,
    OptimizationStatus,
    TrajectoryOptimizer,
    populate_result_metadata,
)
from movement_optimizer.trajectory.optimizer_packaging import (
    find_invalid_trajectory_data,
)

N_EVAL = 20
N_DOF = 3


@pytest.fixture(scope="module")
def squat_optimizer() -> TrajectoryOptimizer:
    body = BodyModel(75.0, 1.75)
    dyn, q_start, q_end, q_bounds = make_squat_config(body, 60.0)
    return TrajectoryOptimizer(
        body,
        dyn,
        "squat",
        60.0,
        q_start,
        q_end,
        q_bounds,
        duration=2.0,
        n_waypoints=6,
        n_eval=N_EVAL,
        n_starts=1,
    )


def _valid_com(opt: TrajectoryOptimizer, n: int = N_EVAL) -> np.ndarray:
    return np.full(n, opt.inner_center)


def _res(cost: float) -> SimpleNamespace:
    return SimpleNamespace(fun=cost, success=True, message="", nit=1)


@pytest.mark.parametrize("cost", [float("-inf"), float("inf"), float("nan")])
def test_non_finite_cost_is_infeasible(squat_optimizer, cost: float) -> None:
    q = np.zeros((N_EVAL, N_DOF))
    success, n_viol = squat_optimizer._check_solution_feasibility(
        _res(cost), q, _valid_com(squat_optimizer)
    )
    assert success is False
    assert n_viol == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_joint_samples_are_infeasible(squat_optimizer, bad: float) -> None:
    q = np.zeros((N_EVAL, N_DOF))
    q[N_EVAL // 2, 1] = bad
    success, _ = squat_optimizer._check_solution_feasibility(
        _res(1.0), q, _valid_com(squat_optimizer)
    )
    assert success is False


def test_all_nan_joints_infeasible_but_not_counted_as_violations(
    squat_optimizer,
) -> None:
    q = np.full((N_EVAL, N_DOF), np.nan)
    success, n_viol = squat_optimizer._check_solution_feasibility(
        _res(1.0), q, _valid_com(squat_optimizer)
    )
    assert success is False
    assert n_viol == 0


def test_empty_trajectory_is_infeasible(squat_optimizer) -> None:
    success, _ = squat_optimizer._check_solution_feasibility(
        _res(1.0), np.zeros((0, N_DOF)), np.zeros(0)
    )
    assert success is False


@pytest.mark.parametrize(
    ("q_shape", "com_len"),
    [((N_EVAL, N_DOF - 1), N_EVAL), ((N_EVAL, N_DOF), N_EVAL - 1), ((N_EVAL,), N_EVAL)],
)
def test_shape_mismatch_is_infeasible(squat_optimizer, q_shape, com_len) -> None:
    success, _ = squat_optimizer._check_solution_feasibility(
        _res(1.0), np.zeros(q_shape), _valid_com(squat_optimizer, com_len)
    )
    assert success is False


def test_valid_data_reports_no_problem() -> None:
    q = np.zeros((N_EVAL, N_DOF))
    assert find_invalid_trajectory_data(q, np.zeros(N_EVAL), N_DOF) is None


@pytest.mark.parametrize(
    ("q", "com_x", "fragment"),
    [
        (np.zeros((0, N_DOF)), np.zeros(0), "empty"),
        (np.zeros((N_EVAL, 2)), np.zeros(N_EVAL), "shape"),
        (np.zeros((N_EVAL, N_DOF)), np.zeros((N_EVAL, 1)), "shape"),
        (np.full((N_EVAL, N_DOF), np.nan), np.zeros(N_EVAL), "non-finite"),
        (np.zeros((N_EVAL, N_DOF)), np.full(N_EVAL, -np.inf), "non-finite"),
    ],
)
def test_invalid_data_is_described(q, com_x, fragment: str) -> None:
    reason = find_invalid_trajectory_data(q, com_x, N_DOF)
    assert reason is not None
    assert fragment in reason


def test_non_finite_com_rejected_even_when_com_constraint_exempt() -> None:
    # Bench press skips the COM bound check, but NaN COM data is still invalid.
    com_x = np.full(N_EVAL, np.nan)
    assert find_invalid_trajectory_data(np.zeros((N_EVAL, N_DOF)), com_x, N_DOF)


def _result(q: np.ndarray, cost: float) -> OptimizationResult:
    n = q.shape[0]
    empty = np.zeros((n, N_DOF))
    return OptimizationResult(
        t=np.zeros(n),
        q=q,
        qd=empty,
        qdd=empty,
        torques=empty,
        power=empty,
        com=np.zeros((n, 2)),
        bar=np.zeros((n, 2)),
        success=False,
        cost=cost,
        com_horizontal_range_cm=0.0,
        status=OptimizationStatus.INFEASIBLE,
    )


def test_negative_infinite_cost_reports_numerical_failure() -> None:
    result = _result(np.zeros((N_EVAL, N_DOF)), float("-inf"))
    populate_result_metadata(result, _res(float("-inf")))
    assert result.status is OptimizationStatus.NUMERICAL_FAILURE


def test_non_finite_trajectory_reports_numerical_failure() -> None:
    result = _result(np.full((N_EVAL, N_DOF), np.nan), 1.0)
    populate_result_metadata(result, _res(1.0))
    assert result.status is OptimizationStatus.NUMERICAL_FAILURE
