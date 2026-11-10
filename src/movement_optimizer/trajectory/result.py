# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Data structures for optimisation results."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class OptimizationStatus(str, Enum):
    """Categorical status of an optimisation solve."""

    CONVERGED = "CONVERGED"
    ITERATION_LIMIT = "ITERATION_LIMIT"
    STALLED = "STALLED"
    INFEASIBLE = "INFEASIBLE"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


def derive_optimization_status(
    res: Any, cost_finite: bool, is_feasible: bool
) -> OptimizationStatus:
    """Derive OptimizationStatus from SciPy result flags and physical feasibility."""
    if not cost_finite:
        return OptimizationStatus.NUMERICAL_FAILURE
    if not is_feasible:
        return OptimizationStatus.INFEASIBLE
    if not getattr(res, "success", True):
        msg = str(getattr(res, "message", "")).lower()
        if getattr(res, "status", None) == 9 or "iteration limit" in msg or ("iteration" in msg and ("maximum" in msg or "exceed" in msg)):
            return OptimizationStatus.ITERATION_LIMIT
        return OptimizationStatus.STALLED
    return OptimizationStatus.CONVERGED


def populate_result_metadata(
    result: OptimizationResult, res: Any, best_start_seed: int | None = None, n_starts_feasible: int = 0
) -> OptimizationResult:
    """Attach status, solver message, iterations, and seed provenance to result."""
    cost_val = float(getattr(res, "fun", result.cost))
    cost_finite = cost_val < float("inf") and not np.isnan(cost_val)
    result.status = derive_optimization_status(res, cost_finite, result.success)
    result.solver_message = str(getattr(res, "message", ""))
    result.n_iterations = int(getattr(res, "nit", 0))
    result.best_start_seed = best_start_seed
    result.n_starts_feasible = n_starts_feasible
    result.backend = "SLSQP"
    return result


def extract_clubhead_kinematics(
    dynamics: Any, t_eval: NDArray, q: NDArray, qd: NDArray
) -> tuple[NDArray | None, NDArray | None]:
    """Extract clubhead poses and twists if supported by dynamics."""
    n = len(t_eval)
    poses = np.array([dynamics.clubhead_pose(q[i]) for i in range(n)], dtype=np.float64) if hasattr(dynamics, "clubhead_pose") else None
    twists = np.array([dynamics.clubhead_twist(q[i], qd[i]) for i in range(n)], dtype=np.float64) if hasattr(dynamics, "clubhead_twist") else None
    return poses, twists


@dataclass
class OptimizationResult:
    """Container for optimiser output.

    Attributes:
        t: Time grid, shape ``(N,)`` in seconds.
        q: Joint angles, shape ``(N, n_dof)`` in radians.
        qd: Joint velocities, shape ``(N, n_dof)`` in rad/s.
        qdd: Joint accelerations, shape ``(N, n_dof)`` in rad/s^2.
        torques: Joint torques, shape ``(N, n_dof)`` in N*m.
        power: Per-joint mechanical power, shape ``(N, n_dof)`` in W.
        com: Whole-body COM trajectory ``(x, y)``, shape ``(N, 2)`` in m.
        bar: Barbell trajectory ``(x, y)``, shape ``(N, 2)`` in m.
        success: True when the optimiser converged and all hard
            constraints (COM in inner BOS, joint limits) are satisfied.
        cost: Final scalar cost (lower is better).
        com_horizontal_range_cm: Peak-to-peak horizontal COM excursion.
        elapsed_s: Wall-clock seconds spent in :meth:`optimize`.
        n_evals: Total cost evaluations across all starts.
        n_joint_limit_violations: Number of trajectory samples whose
            joint angles fall outside ``q_bounds``.
        status: Categorical status of the optimization solve.
        solver_message: Status or termination message from the solver.
        n_iterations: Number of solver iterations completed.
        best_start_seed: Multi-start seed index that yielded this result.
        n_starts_feasible: Total number of starts that yielded a feasible solution.
        backend: Solver backend identifier string (default: "SLSQP").
    """

    t: NDArray
    q: NDArray
    qd: NDArray
    qdd: NDArray
    torques: NDArray
    power: NDArray
    com: NDArray
    bar: NDArray
    success: bool
    cost: float
    com_horizontal_range_cm: float
    elapsed_s: float = 0.0
    n_evals: int = 0
    n_joint_limit_violations: int = 0
    clubhead_poses: NDArray | None = None
    clubhead_twists: NDArray | None = None
    frame_convention: str | None = None
    joint_positions_m: NDArray | None = None
    joint_ids: tuple[str, ...] | None = None
    status: OptimizationStatus = OptimizationStatus.CONVERGED
    solver_message: str = ""
    n_iterations: int = 0
    best_start_seed: int | None = None
    n_starts_feasible: int = 0
    backend: str = "SLSQP"

    def __post_init__(self) -> None:
        status_val: object = self.status
        if not isinstance(status_val, OptimizationStatus):
            object.__setattr__(self, "status", OptimizationStatus(str(status_val)))
        if self.success and self.status is not OptimizationStatus.CONVERGED:
            warnings.warn(
                "OptimizationResult has success=True but status is not CONVERGED. "
                "In a future release, success will strictly require "
                "status == OptimizationStatus.CONVERGED.",
                DeprecationWarning,
                stacklevel=2,
            )

    def torque_joint_count(self) -> int:
        """Return the number of torque columns in the trajectory."""
        return int(self.torques.shape[1])

    def has_com_samples(self) -> bool:
        """Return ``True`` when the COM trajectory has at least one sample."""
        return bool(self.com.shape[0])


@dataclass
class ProgressReport:
    """Snapshot of optimiser state emitted to a progress callback.

    Attributes:
        iteration: Cost evaluation counter.
        cost: Current cost value.
        best_cost: Best cost seen so far.
        improvement_pct: Percent improvement over the previous report.
        elapsed_s: Seconds since :meth:`optimize` started.
        cost_history: Running list of all observed cost values.
        is_stalled: True when the heuristic stall detector triggered.
        stall_reason: Human-readable reason when ``is_stalled``.
    """

    iteration: int
    cost: float
    best_cost: float
    improvement_pct: float
    elapsed_s: float
    cost_history: list[float] = field(default_factory=list)
    is_stalled: bool = False
    stall_reason: str = ""


class CancelledError(Exception):
    """Raised when the user cancels optimisation via ``cancel_event``."""
