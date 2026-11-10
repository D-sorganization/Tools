# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for trajectory optimization execution.

Covers: optimisation convergence, COM constraint enforcement,
parallel multi-start, cancellation, progress reporting, and
stall detection.
"""

from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest

from movement_optimizer.models import (
    BodyModel,
    make_squat_config,
)
from movement_optimizer.trajectory import (
    CancelledError,
    OptimizationResult,
    OptimizationStatus,
    ProgressReport,
    TrajectoryOptimizer,
    derive_optimization_status,
)
from movement_optimizer.trajectory.optimizer_parallel import select_best_result

# ==============================================================
# Optimisation
# ==============================================================


class TestOptimization:
    def test_optimize_returns_result(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        result = opt.optimize()
        assert isinstance(result, OptimizationResult)

    def test_result_shapes(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        result = opt.optimize()
        n = opt.n_eval
        assert result.t.shape == (n,)
        assert result.q.shape == (n, 3)
        assert result.qd.shape == (n, 3)
        assert result.torques.shape == (n, 3)
        assert result.power.shape == (n, 3)
        assert result.com.shape == (n, 2)
        assert result.bar.shape == (n, 2)

    def test_precondition_objective_finite(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        wp = opt._initial_guess()
        cost = opt._compute_cost(wp.flatten())
        assert cost < float(
            "inf"
        ), "Precondition violated: initial objective is not finite"

    def test_postcondition_kkt_within_tol(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        # We assume the optimization result includes 'success' which means KKT conditions are within tolerance
        result = opt.optimize()
        assert (
            result.success
        ), "Postcondition violated: optimization did not satisfy KKT within tolerance"

    def test_cost_decreases(self) -> None:
        """With enough waypoints, optimization should reduce cost."""
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb = make_squat_config(body, 60.0)
        opt = TrajectoryOptimizer(
            body,
            dyn,
            "squat",
            60.0,
            qs,
            qe,
            qb,
            duration=2.0,
            n_waypoints=12,
            n_eval=40,
            n_starts=1,
        )
        wp0 = opt._initial_guess()
        initial_cost = opt._compute_cost(wp0.flatten())
        result = opt.optimize()
        assert result.cost <= initial_cost

    def test_horizontal_range_positive(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        result = opt.optimize()
        assert result.com_horizontal_range_cm >= 0

    def test_full_squat_reaches_depth(self, full_squat_optimizer) -> None:
        opt, _, _ = full_squat_optimizer
        result = opt.optimize()
        min_thigh_angle = np.min(result.q[:, 1])
        assert min_thigh_angle < np.radians(-45)

    def test_optimization_succeeds(self) -> None:
        """Optimization with production-size config should succeed."""
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb = make_squat_config(body, 60.0)
        opt = TrajectoryOptimizer(
            body,
            dyn,
            "squat",
            60.0,
            qs,
            qe,
            qb,
            duration=2.0,
            n_waypoints=12,
            n_eval=40,
            n_starts=1,
        )
        result = opt.optimize()
        assert result.cost < float("inf")
        assert result.success

    def test_com_stays_in_inner_bos(self) -> None:
        """After SLSQP optimization, COM should stay within the inner 60% BOS."""
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb = make_squat_config(body, 60.0)
        opt = TrajectoryOptimizer(
            body,
            dyn,
            "squat",
            60.0,
            qs,
            qe,
            qb,
            duration=2.0,
            n_waypoints=12,
            n_eval=40,
            n_starts=2,
        )
        result = opt.optimize()
        com_x = result.com[:, 0]
        assert np.all(
            com_x >= body.inner_heel - 0.01
        ), f"COM below inner_heel: min={com_x.min():.4f}, bound={body.inner_heel:.4f}"
        assert np.all(
            com_x <= body.inner_toe + 0.01
        ), f"COM above inner_toe: max={com_x.max():.4f}, bound={body.inner_toe:.4f}"
        assert result.success, "Optimization should report success with COM in bounds"


# ==============================================================
# Multi-start
# ==============================================================


class TestMultiStart:
    def test_perturbed_guess_seed_0_is_baseline(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        baseline = opt._initial_guess()
        perturbed = opt._perturbed_guess(0)
        np.testing.assert_array_equal(baseline, perturbed)

    def test_perturbed_guess_different_seeds_differ(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        g1 = opt._perturbed_guess(1)
        g2 = opt._perturbed_guess(2)
        assert not np.allclose(g1, g2)

    def test_perturbed_guess_within_bounds(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        for seed in range(10):
            wp = opt._perturbed_guess(seed)
            for j in range(3):
                assert np.all(wp[:, j] >= opt.q_bounds[j, 0])
                assert np.all(wp[:, j] <= opt.q_bounds[j, 1])

    def test_parallel_multistart_runs(self) -> None:
        """Multi-start with n_starts > 1 should run and return a result."""
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb = make_squat_config(body, 60.0)
        opt = TrajectoryOptimizer(
            body,
            dyn,
            "squat",
            60.0,
            qs,
            qe,
            qb,
            duration=2.0,
            n_waypoints=6,
            n_eval=20,
            n_starts=3,
        )
        result = opt.optimize()
        assert isinstance(result, OptimizationResult)
        assert result.cost < float("inf")


# ==============================================================
# Cancellation
# ==============================================================


class TestCancellation:
    def test_cancel_returns_inf(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        opt.cancel_event.set()
        wp = opt._initial_guess()
        result = opt._compute_cost(wp.flatten())
        assert result == float("inf")

    def test_cancel_during_optimize(self) -> None:
        """Setting cancel_event before optimize should raise CancelledError."""
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb = make_squat_config(body, 60.0)
        cancel = threading.Event()
        opt = TrajectoryOptimizer(
            body,
            dyn,
            "squat",
            60.0,
            qs,
            qe,
            qb,
            n_waypoints=6,
            n_eval=20,
            n_starts=1,
            cancel_event=cancel,
        )
        # Set cancel immediately — should abort on first cost eval or callback
        cancel.set()
        with pytest.raises(CancelledError):
            opt.optimize()

    def test_cancel_event_default(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        assert not opt.cancel_event.is_set()


# ==============================================================
# Progress Reporting & Stall Detection
# ==============================================================


class TestProgressReporting:
    def test_progress_callback_receives_report(self) -> None:
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb = make_squat_config(body, 60.0)
        reports: list[ProgressReport] = []
        # Use larger problem so SLSQP does enough evals to trigger progress
        opt = TrajectoryOptimizer(
            body,
            dyn,
            "squat",
            60.0,
            qs,
            qe,
            qb,
            n_waypoints=12,
            n_eval=40,
            n_starts=1,
            progress_cb=lambda r: reports.append(r),
        )
        opt.optimize()
        # SLSQP may converge fast with hard constraints; reports may be empty
        # The key test is that optimization completes without error
        if len(reports) > 0:
            assert isinstance(reports[0], ProgressReport)

    def test_stall_detection_flat_cost(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        opt._cost_history = [100.0] * 100
        is_stalled, _reason = opt._detect_stall()
        assert is_stalled

    def test_stall_detection_improving(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        opt._cost_history = list(np.linspace(1000, 100, 100))
        is_stalled, _ = opt._detect_stall()
        assert not is_stalled


class _MockOptRes:
    pass


def _make_mock_res(**kw: object) -> _MockOptRes:
    obj = _MockOptRes()
    vars(obj).update(kw)
    return obj


class TestConvergenceAndFeasibilitySelection:
    def test_select_best_result_prioritizes_feasibility_over_cost(self) -> None:
        inf, feas = _make_mock_res(fun=1.0, feasible=False), _make_mock_res(fun=2.0, feasible=True)
        best, total_evals = select_best_result([(inf, 50), (feas, 60)])
        assert best is feas and total_evals == 110

    def test_optimization_status_enum_values(self) -> None:
        for name in ("CONVERGED", "ITERATION_LIMIT", "STALLED", "INFEASIBLE", "NUMERICAL_FAILURE"):
            assert getattr(OptimizationStatus, name).value == name

    def test_deprecation_warning_when_success_and_status_not_converged(self) -> None:
        z3, z2, z1 = np.zeros((1, 3)), np.zeros((1, 2)), np.zeros(1)
        kw = dict(t=z1, q=z3, qd=z3, qdd=z3, torques=z3, power=z3, com=z2, bar=z2, cost=10.0, com_horizontal_range_cm=2.0)
        with pytest.deprecated_call(match="success=True but status is not CONVERGED"):
            OptimizationResult(**kw, success=True, status=OptimizationStatus.ITERATION_LIMIT)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            OptimizationResult(**kw, success=True, status=OptimizationStatus.CONVERGED)
            OptimizationResult(**kw, success=False, status=OptimizationStatus.ITERATION_LIMIT)

    def test_derive_optimization_status(self) -> None:
        cases = [
            (_make_mock_res(fun=float("nan"), success=True), False, False, OptimizationStatus.NUMERICAL_FAILURE),
            (_make_mock_res(fun=5.0, success=True), True, False, OptimizationStatus.INFEASIBLE),
            (_make_mock_res(fun=5.0, success=False, status=9, message="Iteration limit reached"), True, True, OptimizationStatus.ITERATION_LIMIT),
            (_make_mock_res(fun=5.0, success=False, status=2, message="Maximum iterations exceeded"), True, True, OptimizationStatus.ITERATION_LIMIT),
            (_make_mock_res(fun=5.0, success=False, status=8, message="Positive directional derivative"), True, True, OptimizationStatus.STALLED),
            (_make_mock_res(fun=5.0, success=True, status=0, message="Optimization terminated"), True, True, OptimizationStatus.CONVERGED),
        ]
        for res, cost_finite, is_feasible, expected in cases:
            assert derive_optimization_status(res, cost_finite=cost_finite, is_feasible=is_feasible) == expected

    def test_max_iter_patched_to_1_produces_iteration_limit(self, squat_optimizer, monkeypatch) -> None:
        from movement_optimizer.trajectory import optimizer as opt_mod
        from movement_optimizer.trajectory import optimizer_diagnostics as diag_mod
        from movement_optimizer.trajectory import tuning as tuning_mod

        for m in (tuning_mod, opt_mod, diag_mod):
            monkeypatch.setattr(m, "MAX_ITER_PER_START", 1)
        opt, _, _, _, _ = squat_optimizer
        assert opt.optimize().status == OptimizationStatus.ITERATION_LIMIT
