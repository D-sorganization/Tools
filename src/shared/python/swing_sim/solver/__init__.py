"""Impact-parameter solver subpackage of ``swing_sim`` (epic #4103, #4109).

Goal-driven robust optimization over golf delivery/swing variables:
declare target launch-monitor numbers (:class:`ImpactGoal`), choose which
variables the optimizer controls (:class:`VariablePartition`), and
:func:`solve` finds the bounded least-squares best fit with parallel
multi-start (Latin hypercube), progress reporting, and cancellation.

Scaffolding modeled on UpstreamDrift's
``src/shared/python/movement_optimizer`` (pure cost module, multi-start
parallel driver, ProgressReport/cancel_event plumbing, named tuning
constants), with golf-impact semantics replacing the barbell/balance
costs. The inner evaluation is the pure function
:func:`evaluate_candidate` — the documented seam a later Rust port
replaces behind a facade.

Self-facaded: downstream code imports from
``shared.python.swing_sim.solver`` only. The parent
``swing_sim/__init__.py`` facade is wired during epic integration; do not
add solver exports there from this subpackage.
"""

from __future__ import annotations

from .goals import (
    DELIVERY_VARIABLE_DEFAULTS,
    GOAL_QUANTITIES,
    SWING_DERIVED_VARIABLES,
    SWING_VARIABLE_DEFAULTS,
    GoalTerm,
    ImpactGoal,
    VariablePartition,
)
from .objective import (
    EvaluationConfig,
    achieved_quantities,
    evaluate_candidate,
    residuals,
)
from .solve import (
    CancelledError,
    ProgressCallback,
    ProgressReport,
    SolverResult,
    StartSummary,
    detect_stall,
    solve,
)

__all__ = [
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
]
