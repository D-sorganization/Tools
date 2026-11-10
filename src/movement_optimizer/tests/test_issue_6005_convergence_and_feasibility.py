# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for issue #6005: SciPy convergence status and feasibility-first selection."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from movement_optimizer.persistence import _dict_to_result, load_solution, save_solution
from movement_optimizer.trajectory import OptimizationResult, OptimizationStatus
from movement_optimizer.trajectory.optimizer_packaging import (
    _derive_optimization_status,
)
from movement_optimizer.trajectory.optimizer_parallel import select_best_result

from .conftest import make_test_result


def test_select_best_result_prioritizes_feasibility_over_cost() -> None:
    """Acceptance criterion: feasible start is returned over lower-cost infeasible start."""
    inf, feas = SimpleNamespace(fun=1.0, feasible=False), SimpleNamespace(fun=2.0, feasible=True)
    best, total_evals = select_best_result([(inf, 50), (feas, 60)])
    assert best is feas and total_evals == 110


def test_optimization_status_enum_values() -> None:
    """Ensure all required OptimizationStatus enum values exist."""
    for name in ("CONVERGED", "ITERATION_LIMIT", "STALLED", "INFEASIBLE", "NUMERICAL_FAILURE"):
        assert getattr(OptimizationStatus, name).value == name


def test_deprecation_warning_when_success_and_status_not_converged() -> None:
    """Ensure DeprecationWarning fires only when success=True and status is not CONVERGED."""
    base = make_test_result()

    def make_res(success: bool, status: OptimizationStatus) -> OptimizationResult:
        kw = {
            k: getattr(base, k) for k in ("t", "q", "qd", "qdd", "torques", "power", "com", "bar")
        }
        return OptimizationResult(
            **kw, cost=10.0, com_horizontal_range_cm=2.0, success=success, status=status
        )

    with pytest.deprecated_call(match="success=True but status is not CONVERGED"):
        make_res(True, OptimizationStatus.ITERATION_LIMIT)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        make_res(True, OptimizationStatus.CONVERGED)
        make_res(False, OptimizationStatus.ITERATION_LIMIT)


def test_derive_optimization_status() -> None:
    """Ensure status mapping correctly handles numerical failures, feasibility, and SciPy flags."""
    cases = [
        (
            SimpleNamespace(fun=float("nan"), success=True),
            False,
            False,
            OptimizationStatus.NUMERICAL_FAILURE,
        ),
        (SimpleNamespace(fun=5.0, success=True), True, False, OptimizationStatus.INFEASIBLE),
        (
            SimpleNamespace(fun=5.0, success=False, status=9, message="Iteration limit reached"),
            True,
            True,
            OptimizationStatus.ITERATION_LIMIT,
        ),
        (
            SimpleNamespace(
                fun=5.0, success=False, status=2, message="Maximum iterations exceeded"
            ),
            True,
            True,
            OptimizationStatus.ITERATION_LIMIT,
        ),
        (
            SimpleNamespace(
                fun=5.0, success=False, status=8, message="Positive directional derivative"
            ),
            True,
            True,
            OptimizationStatus.STALLED,
        ),
        (
            SimpleNamespace(fun=5.0, success=True, status=0, message="Optimization terminated"),
            True,
            True,
            OptimizationStatus.CONVERGED,
        ),
    ]
    for res, cost_finite, is_feasible, expected in cases:
        assert (
            _derive_optimization_status(res, cost_finite=cost_finite, is_feasible=is_feasible)
            == expected
        )


def test_persistence_roundtrip_new_fields_and_schema_v1_compatibility(tmp_path: Any) -> None:
    """Ensure new fields round-trip through JSON persistence and schema-v1 state loads cleanly."""
    base = make_test_result()
    kw = {k: getattr(base, k) for k in ("t", "q", "qd", "qdd", "torques", "power", "com", "bar")}
    res = OptimizationResult(
        **kw,
        success=True,
        cost=25.0,
        com_horizontal_range_cm=1.5,
        status=OptimizationStatus.CONVERGED,
        solver_message="Optimization terminated successfully",
        n_iterations=42,
        best_start_seed=3,
        n_starts_feasible=5,
        backend="SLSQP",
    )
    path = tmp_path / "solution_v2.json"
    save_solution(str(path), res, {"body_mass": 80.0, "height": 1.80}, "squat", 60.0)
    loaded_data = load_solution(str(path))
    meta = loaded_data["metadata"]

    for k, v in [
        ("status", "CONVERGED"),
        ("solver_message", "Optimization terminated successfully"),
        ("n_iterations", 42),
        ("best_start_seed", 3),
        ("n_starts_feasible", 5),
        ("backend", "SLSQP"),
    ]:
        assert meta[k] == v

    reconstructed = _dict_to_result(loaded_data)
    assert reconstructed.status == OptimizationStatus.CONVERGED
    assert reconstructed.solver_message == "Optimization terminated successfully"
    assert reconstructed.n_iterations == 42
    assert reconstructed.best_start_seed == 3
    assert reconstructed.n_starts_feasible == 5
    assert reconstructed.backend == "SLSQP"

    # Schema-v1 backward compatibility (omits new metadata fields)
    v1_dict = {
        "arrays": loaded_data["arrays"],
        "metadata": {
            "success": True,
            "cost": 30.0,
            "com_horizontal_range_cm": 2.0,
            "elapsed_s": 1.0,
            "n_evals": 10,
        },
    }
    v1_res = _dict_to_result(v1_dict)
    assert v1_res.status == OptimizationStatus.CONVERGED
    assert v1_res.solver_message == ""
    assert v1_res.n_iterations == 0
    assert v1_res.best_start_seed is None
    assert v1_res.n_starts_feasible == 0
    assert v1_res.backend == "SLSQP"


def test_max_iter_patched_to_1_produces_iteration_limit(
    squat_optimizer: Any, monkeypatch: Any
) -> None:
    """Acceptance criterion: With MAX_ITER_PER_START patched to 1, status is ITERATION_LIMIT."""
    import movement_optimizer.trajectory.optimizer as opt_mod
    import movement_optimizer.trajectory.optimizer_diagnostics as diag_mod
    import movement_optimizer.trajectory.tuning as tuning

    for m in (tuning, opt_mod, diag_mod):
        monkeypatch.setattr(m, "MAX_ITER_PER_START", 1)
    opt, _, _, _, _ = squat_optimizer
    assert opt.optimize().status == OptimizationStatus.ITERATION_LIMIT
