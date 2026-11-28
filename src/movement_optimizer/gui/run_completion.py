# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""User-facing completion notices for one optimization run.

The GUI asks this module what to say. It does not re-derive solver status
or COM bounds itself. One classifier feeds the stall line, the status bar,
and the balance label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import BodyModel
from ..trajectory.optimizer_packaging import check_com_feasibility
from ..trajectory.result import OptimizationResult, OptimizationStatus

COM_BOS_STALL = (
    "\u26a0 COM went outside the inner 60% BOS zone. "
    "Try increasing smoothness or adjusting body parameters."
)
ITERATION_LIMIT_STALL = (
    "\u26a0 Iteration limit reached before convergence. "
    "Constraints are satisfied, but the trajectory is not fully converged."
)
NUMERICAL_STALL = "\u26a0 Numerical failure. The solver returned a non-finite result."
JOINT_LIMIT_STALL = "\u26a0 Joint limits violated. COM stayed inside the inner base of support."
INFEASIBLE_STALL = "\u26a0 Optimization did not produce a feasible trajectory."


@dataclass(frozen=True)
class RunCompletion:
    """Messages for one finished run.

    ``stall_message`` is ``None`` when the sidebar stall should be cleared.
    ``balance_label`` is ``None`` when the exercise has no COM constraint.
    """

    status_message: str
    stall_message: str | None
    balance_label: str | None


def com_constraint_active(exercise_type: str) -> bool:
    """Return True when the exercise enforces an inner-BOS COM constraint.

    Bench press is supine, so the COM constraint is inactive.
    """
    return exercise_type != "bench_press"


def com_balance_violated(result: OptimizationResult, exercise_type: str, body: BodyModel) -> bool:
    """Return True only when an active COM constraint is actually broken.

    Preconditions:
        ``body.inner_heel`` and ``body.inner_toe`` are the inner-BOS bounds
        used for this run.
    Postconditions:
        Numerical failures and inactive COM constraints return False even
        when COM samples are non-finite or outside the foot.
    """
    if not com_constraint_active(exercise_type):
        return False
    if result.status is OptimizationStatus.NUMERICAL_FAILURE:
        return False
    com = np.asarray(result.com)
    if com.ndim != 2 or com.shape[0] == 0 or com.shape[1] < 1:
        return False
    com_x = np.asarray(com[:, 0], dtype=float)
    if not bool(np.all(np.isfinite(com_x))):
        return False
    return not check_com_feasibility(
        True,
        com_x,
        exercise_type,
        float(body.inner_heel),
        float(body.inner_toe),
    )


def balance_word(result: OptimizationResult, exercise_type: str, body: BodyModel) -> str:
    """Return the sidebar balance word for an exercise that shows one.

    Preconditions:
        The caller only asks when a balance line is displayed.
    """
    label = _balance_label(exercise_type, _completion_kind(result, exercise_type, body))
    if label is None:
        return "BALANCED"
    return label


def describe_run_completion(
    result: OptimizationResult,
    *,
    exercise_name: str,
    exercise_type: str,
    body: BodyModel,
    elapsed_label: str,
) -> RunCompletion:
    """Classify one finished run.

    Preconditions:
        ``exercise_name`` and ``elapsed_label`` are non-empty.
    Postconditions:
        A COM BOS stall is used only when the COM constraint is active and
        the COM samples leave the inner zone. A feasible iteration-limit
        run names that limit and does not claim convergence.
    """
    if not exercise_name or not elapsed_label:
        raise ValueError("exercise_name and elapsed_label are required")
    kind = _completion_kind(result, exercise_type, body)
    stall, status = _messages(kind, exercise_name, elapsed_label)
    return RunCompletion(
        status_message=status,
        stall_message=stall,
        balance_label=_balance_label(exercise_type, kind),
    )


def _completion_kind(result: OptimizationResult, exercise_type: str, body: BodyModel) -> str:
    """Pick one completion class. Earlier classes outrank later ones."""
    if result.status is OptimizationStatus.NUMERICAL_FAILURE:
        return "numerical"
    if result.status is OptimizationStatus.ITERATION_LIMIT and bool(result.is_feasible):
        return "iteration_limit"
    if com_balance_violated(result, exercise_type, body):
        return "com_bos"
    if not result.is_feasible and result.n_joint_limit_violations > 0:
        return "joint_limits"
    if not result.is_feasible:
        return "infeasible"
    return "complete"


def _balance_label(exercise_type: str, kind: str) -> str | None:
    if not com_constraint_active(exercise_type):
        return None
    if kind == "com_bos":
        return "OUT OF BOUNDS"
    if kind == "numerical":
        return "UNAVAILABLE"
    return "BALANCED"


def _messages(kind: str, name: str, elapsed: str) -> tuple[str | None, str]:
    if kind == "complete":
        return None, f"{name} optimization complete in {elapsed}!"
    if kind == "iteration_limit":
        return (
            ITERATION_LIMIT_STALL,
            f"{name} done in {elapsed} -- WARNING: iteration limit (not converged)",
        )
    if kind == "com_bos":
        return (
            COM_BOS_STALL,
            f"{name} done in {elapsed} -- WARNING: COM balance violated",
        )
    if kind == "numerical":
        return (
            NUMERICAL_STALL,
            f"{name} done in {elapsed} -- WARNING: numerical failure",
        )
    if kind == "joint_limits":
        return (
            JOINT_LIMIT_STALL,
            f"{name} done in {elapsed} -- WARNING: joint limits violated",
        )
    return (
        INFEASIBLE_STALL,
        f"{name} done in {elapsed} -- WARNING: infeasible trajectory",
    )
