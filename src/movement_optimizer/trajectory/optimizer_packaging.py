# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Result-packaging helpers for the trajectory optimiser.

These free functions handle the post-solve steps: expanding the solver's
flat solution vector into physical trajectories, checking feasibility, and
assembling the final :class:`OptimizationResult` dataclass.

Keeping them separate from the main optimiser class makes them independently
testable and keeps the class focused on orchestration.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..backend import PhysicsBackend
from ..constants import COM_FEASIBILITY_TOL_M, JOINT_FEASIBILITY_TOL_RAD
from .result import OptimizationResult, OptimizationStatus


class _OptimizeResult(Protocol):
    """Structural protocol for scipy.optimize.OptimizeResult.

    Scipy's OptimizeResult is a dict-like object; the stubs expose it as
    ``object`` in some contexts.  Declaring a minimal Protocol avoids
    ``type: ignore[attr-defined]`` at every ``res.fun`` and ``res.x``
    access without importing scipy in type-check-only paths.
    """

    @property
    def fun(self) -> float:
        """Objective function value at the solution."""
        ...

    @property
    def x(self) -> NDArray:
        """Solution array."""
        ...


class _SplineCallable(Protocol):
    """Protocol for build_splines callable: x -> CubicSpline-like."""

    def __call__(self, x: NDArray) -> object:
        """Build splines from flat waypoint vector."""
        ...


class _TrajectoryCallable(Protocol):
    """Protocol for eval_trajectory callable: splines -> (q, qd, qdd, qddd)."""

    def __call__(self, splines: object) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Evaluate trajectory from splines."""
        ...


logger = logging.getLogger(__name__)

__all__ = [
    "build_result_object",
    "check_com_feasibility",
    "count_joint_limit_violations",
    "evaluate_solution",
]


def evaluate_solution(
    res: _OptimizeResult,
    dynamics: PhysicsBackend,
    exercise_type: str,
    bar_mass: float,
    build_splines_fn: _SplineCallable,
    eval_trajectory_fn: _TrajectoryCallable,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Expand solver output into physical trajectories and COM / bar paths.

    Parameters:
        res: scipy OptimizeResult (must have .x attribute).
        dynamics: physics backend providing inverse_dynamics_batch, com_x_batch,
            com_position, and bar_position.
        exercise_type: exercise identifier string.
        bar_mass: barbell mass in kg.
        build_splines_fn: callable(x) -> CubicSpline.
        eval_trajectory_fn: callable(splines) -> (q, qd, qdd, qddd).

    Returns:
        Tuple of (q, qd, qdd, torques, power, com_traj, bar_traj, com_x),
        each as float64 NDArray.
    """
    splines = build_splines_fn(res.x)
    q, qd, qdd, _ = eval_trajectory_fn(splines)

    torques = dynamics.inverse_dynamics_batch(q, qd, qdd)
    power = torques * qd

    n_pts = q.shape[0]
    com_x = dynamics.com_x_batch(q, exercise_type, bar_mass)
    com_traj = np.empty((n_pts, 2))
    bar_traj = np.empty((n_pts, 2))
    for n in range(n_pts):
        com_full = dynamics.com_position(q[n], exercise_type, bar_mass)
        com_traj[n, 0] = com_x[n]
        com_traj[n, 1] = com_full[1]
        bar_traj[n] = dynamics.bar_position(q[n], exercise_type)

    return q, qd, qdd, torques, power, com_traj, bar_traj, com_x


def check_com_feasibility(
    cost_finite: bool,
    com_x: NDArray[np.float64],
    exercise_type: str,
    inner_heel: float,
    inner_toe: float,
) -> bool:
    """Return True if the COM trajectory satisfies inner-BOS bounds.

    Bench press is exempt (lifter is lying down, COM constraint is inactive).
    Emits a warning when cost is finite but COM is out of bounds.

    Preconditions:
        com_x.ndim == 1
    """
    if exercise_type == "bench_press":
        return True
    in_bounds = bool(
        np.all(com_x >= inner_heel - COM_FEASIBILITY_TOL_M)
        and np.all(com_x <= inner_toe + COM_FEASIBILITY_TOL_M)
    )
    if cost_finite and not in_bounds:
        logger.warning(
            "Solution found but COM violated inner BOS: min=%.4f max=%.4f (bounds: [%.4f, %.4f])",
            com_x.min(),
            com_x.max(),
            inner_heel,
            inner_toe,
        )
    return in_bounds


def count_joint_limit_violations(
    q: NDArray[np.float64],
    q_bounds: NDArray[np.float64] | None,
) -> int:
    """Count (and warn about) trajectory points that violate joint limits.

    Spline overshoot between control points may push q outside q_bounds.
    A small tolerance (:data:`JOINT_FEASIBILITY_TOL_RAD`) absorbs benign
    numerical overshoot so only material violations are counted.
    Returns 0 when q_bounds is None.

    Preconditions:
        q.ndim == 2 and q.shape[1] == 3
        q_bounds is None or q_bounds.shape == (3, 2)
    """
    if q_bounds is None:
        return 0
    lower = q_bounds[:, 0] - JOINT_FEASIBILITY_TOL_RAD
    upper = q_bounds[:, 1] + JOINT_FEASIBILITY_TOL_RAD
    n_violations = int(np.sum((q < lower) | (q > upper)))
    if n_violations > 0:
        logger.warning(
            "Trajectory has %d point(s) violating joint limits "
            "(spline overshoot between control points).",
            n_violations,
        )
    return n_violations


def build_result_object(
    *,
    t_eval: NDArray[np.float64],
    res: _OptimizeResult,
    q: NDArray[np.float64],
    qd: NDArray[np.float64],
    qdd: NDArray[np.float64],
    torques: NDArray[np.float64],
    power: NDArray[np.float64],
    com_traj: NDArray[np.float64],
    bar_traj: NDArray[np.float64],
    com_x: NDArray[np.float64],
    success: bool,
    n_joint_limit_violations: int,
    elapsed: float,
    n_evals: int,
    clubhead_poses: NDArray[np.float64] | None = None,
    clubhead_twists: NDArray[np.float64] | None = None,
    status: OptimizationStatus = OptimizationStatus.CONVERGED,
    solver_message: str = "",
    n_iterations: int = 0,
    best_start_seed: int | None = None,
    n_starts_feasible: int = 0,
    backend: str = "SLSQP",
) -> OptimizationResult:
    """Assemble the final OptimizationResult dataclass from solver outputs.

    Preconditions:
        t_eval.ndim == 1 and len(t_eval) == q.shape[0]
        elapsed >= 0
    """
    cost_val = float(res.fun)
    com_h_range = float((np.max(com_x) - np.min(com_x)) * 100.0)
    return OptimizationResult(
        t=t_eval,
        q=q,
        qd=qd,
        qdd=qdd,
        torques=torques,
        power=power,
        com=com_traj,
        bar=bar_traj,
        success=success,
        cost=cost_val,
        com_horizontal_range_cm=com_h_range,
        elapsed_s=elapsed,
        n_evals=n_evals,
        n_joint_limit_violations=n_joint_limit_violations,
        clubhead_poses=clubhead_poses,
        clubhead_twists=clubhead_twists,
        status=status,
        solver_message=solver_message,
        n_iterations=n_iterations,
        best_start_seed=best_start_seed,
        n_starts_feasible=n_starts_feasible,
        backend=backend,
    )


def derive_optimization_status(
    res: Any,
    cost_finite: bool,
    is_feasible: bool,
) -> OptimizationStatus:
    """Derive OptimizationStatus from SciPy result flags and physical feasibility."""
    if not cost_finite:
        return OptimizationStatus.NUMERICAL_FAILURE
    if not is_feasible:
        return OptimizationStatus.INFEASIBLE

    scipy_success = bool(getattr(res, "success", True))
    if not scipy_success:
        status_code = getattr(res, "status", None)
        msg = str(getattr(res, "message", "")).lower()
        if (
            status_code == 9
            or "iteration limit" in msg
            or ("iteration" in msg and ("maximum" in msg or "exceed" in msg))
        ):
            return OptimizationStatus.ITERATION_LIMIT
        return OptimizationStatus.STALLED

    return OptimizationStatus.CONVERGED


_derive_optimization_status = derive_optimization_status


def package_optimization_result(
    opt: Any,
    res: Any,
    elapsed: float = 0.0,
    n_evals: int = 0,
    best_start_seed: int | None = None,
    n_starts_feasible: int = 0,
) -> OptimizationResult:
    """Evaluate trajectories, assess feasibility, and build the result object."""
    q, qd, qdd, torques, power, com_traj, bar_traj, com_x = evaluate_solution(
        res,
        opt.dynamics,
        opt.exercise_type,
        opt.bar_mass,
        opt.build_splines,
        opt.eval_trajectory,
    )
    cost_val = float(res.fun)
    cost_finite = cost_val < float("inf") and not np.isnan(cost_val)
    success, n_viol = opt._check_solution_feasibility(res, q, com_x)
    status = derive_optimization_status(res, cost_finite, success)
    solver_msg = str(getattr(res, "message", ""))
    n_iters = int(getattr(res, "nit", 0))

    clubhead_poses: NDArray[np.float64] | None = None
    clubhead_twists: NDArray[np.float64] | None = None
    if hasattr(opt.dynamics, "clubhead_pose"):
        clubhead_poses = np.empty((len(opt.t_eval), 4, 4), dtype=np.float64)
        for n in range(len(opt.t_eval)):
            clubhead_poses[n] = opt.dynamics.clubhead_pose(q[n])
    if hasattr(opt.dynamics, "clubhead_twist"):
        clubhead_twists = np.empty((len(opt.t_eval), 6), dtype=np.float64)
        for n in range(len(opt.t_eval)):
            clubhead_twists[n] = opt.dynamics.clubhead_twist(q[n], qd[n])
    return build_result_object(
        t_eval=opt.t_eval,
        res=res,
        q=q,
        qd=qd,
        qdd=qdd,
        torques=torques,
        power=power,
        com_traj=com_traj,
        bar_traj=bar_traj,
        com_x=com_x,
        success=success,
        n_joint_limit_violations=n_viol,
        elapsed=elapsed,
        n_evals=n_evals or getattr(opt._progress, "iteration_count", 0),
        clubhead_poses=clubhead_poses,
        clubhead_twists=clubhead_twists,
        status=status,
        solver_message=solver_msg,
        n_iterations=n_iters,
        best_start_seed=best_start_seed,
        n_starts_feasible=n_starts_feasible,
        backend="SLSQP",
    )
