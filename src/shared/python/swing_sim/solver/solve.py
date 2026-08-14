"""Robust multi-start driver for the impact-parameter solver (#4103, #4109).

Scaffolding modeled on UpstreamDrift's ``movement_optimizer``:

- :class:`ProgressReport` / :class:`CancelledError` copy the shapes of
  ``movement_optimizer/trajectory/result.py`` so UIs can share plumbing;
- the multi-start dispatch/collect/select structure mirrors
  ``trajectory/optimizer_parallel.py`` (``concurrent.futures`` pool,
  drain loop honouring the cancel event, best-of selection);
- the progress tracker mirrors ``trajectory/optimizer_progress.py``
  (thread-safe recording, periodic emission, stall heuristic).

Golf-impact semantics replace the barbell/balance machinery: each start is
a bounded ``scipy.optimize.least_squares`` (trf) run over the free
variables of a :class:`~shared.python.swing_sim.solver.goals.VariablePartition`,
scoring :func:`~shared.python.swing_sim.solver.objective.residuals`
against an :class:`~shared.python.swing_sim.solver.goals.ImpactGoal`.
Starts are a Latin-hypercube sample of the bounds (start 0 is the caller's
``x0`` or the bounds midpoint).
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from shared.python.contracts import require

from .goals import ImpactGoal, VariablePartition
from .objective import EvaluationConfig, achieved_quantities, residuals
from .tuning import (
    DEFAULT_FTOL,
    DEFAULT_GTOL,
    DEFAULT_MAX_NFEV_PER_START,
    DEFAULT_N_STARTS,
    DEFAULT_XTOL,
    PROGRESS_EMIT_EVERY,
    STALL_THRESHOLD,
    STALL_WINDOW,
)


class CancelledError(Exception):
    """Raised when the caller cancels solving via ``cancel_event``.

    Same shape as ``movement_optimizer.trajectory.result.CancelledError``.
    """


@dataclass
class ProgressReport:
    """Snapshot of solver state emitted to a progress callback.

    Field-for-field copy of
    ``movement_optimizer.trajectory.result.ProgressReport`` so existing
    progress UIs plug in unchanged; ``cost`` is half the squared residual
    norm (the scipy least-squares cost).
    """

    iteration: int
    cost: float
    best_cost: float
    improvement_pct: float
    elapsed_s: float
    cost_history: list[float] = field(default_factory=list)
    is_stalled: bool = False
    stall_reason: str = ""


ProgressCallback = Callable[[ProgressReport], None]


def detect_stall(history: list[float]) -> tuple[bool, str]:
    """Stall heuristic ported from ``optimizer_progress.detect_stall``."""
    if len(history) < STALL_WINDOW:
        return False, ""
    old_cost, new_cost = history[-STALL_WINDOW], history[-1]
    if old_cost == 0:
        return False, ""
    if abs(old_cost - new_cost) / abs(old_cost) < STALL_THRESHOLD:
        return True, (
            f"Cost changed < {STALL_THRESHOLD * 100:.2f}% over last "
            f"{STALL_WINDOW} evals ({old_cost:.3g} -> {new_cost:.3g})"
        )
    return False, ""


class _ProgressTracker:
    """Thread-safe eval recording + periodic emission (movement_optimizer)."""

    def __init__(self, progress_cb: ProgressCallback | None) -> None:
        self._cb = progress_cb
        self._lock = threading.Lock()
        self._iter = 0
        self._best = float("inf")
        self._history: list[float] = []
        self._start = time.monotonic()

    def record(self, cost: float) -> None:
        with self._lock:
            self._iter += 1
            self._history.append(cost)
            self._best = min(self._best, cost)
            if self._cb is not None and self._iter % PROGRESS_EMIT_EVERY == 0:
                self._emit(cost)

    def _emit(self, cost: float) -> None:
        if len(self._history) >= 2 * PROGRESS_EMIT_EVERY:
            prev = self._history[-2 * PROGRESS_EMIT_EVERY]
            improvement = (prev - cost) / abs(prev) * 100 if prev != 0 else 0.0
        else:
            improvement = 0.0
        is_stalled, reason = detect_stall(self._history)
        assert self._cb is not None
        self._cb(
            ProgressReport(
                iteration=self._iter,
                cost=cost,
                best_cost=self._best,
                improvement_pct=improvement,
                elapsed_s=time.monotonic() - self._start,
                cost_history=self._history[-200:],
                is_stalled=is_stalled,
                stall_reason=reason,
            )
        )


@dataclass(frozen=True)
class StartSummary:
    """Diagnostics for one multi-start seed."""

    seed: int
    x0: np.ndarray
    x: np.ndarray | None
    cost: float
    n_evals: int
    converged: bool
    message: str
    cancelled: bool = False


@dataclass(frozen=True)
class SolverResult:
    """Best-of-all-starts solution plus diagnostics.

    Attributes:
        variables: Full solved variable mapping (free solution + fixed +
            defaults), launch-monitor units per :mod:`.goals`.
        free_names: Free-variable ordering matching ``x``.
        x: Solved free-variable vector.
        achieved: Achieved quantities at the solution (delivery-level
            always; launch/flight-level when the goal required them).
        per_goal_errors: ``quantity -> achieved - target`` at the solution.
        residual_norm: 2-norm of the weighted residual vector.
        cost: Scipy least-squares cost (``0.5 * residual_norm**2``).
        converged: True when the best start terminated successfully.
        n_evals: Total residual evaluations across all starts.
        elapsed_s: Wall-clock seconds spent in :func:`solve`.
        starts: Per-start summaries (diagnostics for all seeds).
    """

    variables: dict[str, float]
    free_names: tuple[str, ...]
    x: np.ndarray
    achieved: dict[str, float]
    per_goal_errors: dict[str, float]
    residual_norm: float
    cost: float
    converged: bool
    n_evals: int
    elapsed_s: float
    starts: tuple[StartSummary, ...]


class _StartCancelled(Exception):
    """Internal: unwinds a scipy start when the cancel event fires."""


def _build_starts(
    lo: np.ndarray,
    hi: np.ndarray,
    n_starts: int,
    seed: int,
    x0: np.ndarray | None,
) -> list[np.ndarray]:
    """Start 0 = ``x0`` or bounds midpoint; rest = Latin hypercube."""
    first = (
        np.clip(np.asarray(x0, dtype=float), lo, hi)
        if x0 is not None
        else 0.5 * (lo + hi)
    )
    starts = [first]
    if n_starts > 1:
        sampler = qmc.LatinHypercube(d=lo.size, seed=seed)
        unit = sampler.random(n=n_starts - 1)
        starts.extend(lo + unit_row * (hi - lo) for unit_row in unit)
    return starts


def _run_single_start(
    seed_idx: int,
    x_start: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    partition: VariablePartition,
    goal: ImpactGoal,
    config: EvaluationConfig | None,
    tracker: _ProgressTracker,
    cancel_event: threading.Event,
    max_nfev: int,
) -> StartSummary:
    """One bounded trf run; cancellation unwinds via :class:`_StartCancelled`."""
    n_evals = 0

    def fun(x: np.ndarray) -> np.ndarray:
        nonlocal n_evals
        if cancel_event.is_set():
            raise _StartCancelled
        res: np.ndarray = residuals(x, partition, goal, config)
        n_evals += 1
        tracker.record(0.5 * float(res @ res))
        return res

    try:
        res = least_squares(
            fun,
            x_start,
            bounds=(lo, hi),
            method="trf",
            xtol=DEFAULT_XTOL,
            ftol=DEFAULT_FTOL,
            gtol=DEFAULT_GTOL,
            max_nfev=max_nfev,
        )
    except _StartCancelled:
        return StartSummary(
            seed=seed_idx,
            x0=x_start,
            x=None,
            cost=float("inf"),
            n_evals=n_evals,
            converged=False,
            message="cancelled",
            cancelled=True,
        )
    return StartSummary(
        seed=seed_idx,
        x0=x_start,
        x=np.asarray(res.x, dtype=float),
        cost=float(res.cost),
        n_evals=n_evals,
        converged=bool(res.success),
        message=str(res.message),
    )


def _collect(
    pending: set[Future[StartSummary]],
    cancel_event: threading.Event,
) -> list[StartSummary]:
    """Drain futures, honouring cancellation (optimizer_parallel shape)."""
    summaries: list[StartSummary] = []
    while pending:
        if cancel_event.is_set():
            for f in pending:
                f.cancel()
            # Workers observe the event via their residual wrapper; any
            # still-running start returns a cancelled summary.
            done, _ = wait(pending, timeout=None)
            summaries.extend(f.result() for f in done if not f.cancelled())
            break
        done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
        summaries.extend(f.result() for f in done)
    return summaries


def solve(
    goal: ImpactGoal,
    partition: VariablePartition,
    config: EvaluationConfig | None = None,
    n_starts: int = DEFAULT_N_STARTS,
    seed: int = 0,
    x0: np.ndarray | None = None,
    max_nfev_per_start: int = DEFAULT_MAX_NFEV_PER_START,
    n_workers: int | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> SolverResult:
    """Solve for the variable values that best achieve the goal.

    Args:
        goal: Targeted quantities with weights.
        partition: Free variables (with bounds) vs fixed values.
        config: Optional evaluation knobs (flight model, swing grid).
        n_starts: Multi-start count (>= 1); start 0 is ``x0`` or the
            bounds midpoint, the rest a seeded Latin-hypercube sample.
        seed: RNG seed for the Latin-hypercube starts.
        x0: Optional initial free-variable vector for start 0.
        max_nfev_per_start: Residual-evaluation cap per start.
        n_workers: Thread-pool size (default ``min(n_starts, 4)``).
        progress_cb: Optional callback receiving :class:`ProgressReport`.
        cancel_event: Optional event; setting it aborts pending and
            in-flight starts.

    Returns:
        :class:`SolverResult` for the lowest-cost completed start.

    Raises:
        CancelledError: If cancelled before any start completed.
        ValueError / ContractViolationError: On invalid inputs (empty free
            set is rejected by ``partition.bounds_arrays``).
    """
    require(n_starts >= 1, "n_starts must be >= 1", n_starts)
    require(max_nfev_per_start >= 1, "max_nfev_per_start must be >= 1", None)
    require(seed >= 0, "seed must be >= 0", seed)
    lo, hi = partition.bounds_arrays()
    if x0 is not None:
        arr = np.asarray(x0, dtype=float)
        require(arr.shape == lo.shape, "x0 must match the free set", arr.shape)
        require(bool(np.all(np.isfinite(arr))), "x0 must be finite", arr)

    event = cancel_event or threading.Event()
    if event.is_set():
        raise CancelledError("Solve cancelled before start")

    t0 = time.monotonic()
    tracker = _ProgressTracker(progress_cb)
    starts = _build_starts(lo, hi, n_starts, seed, x0)
    workers = n_workers if n_workers is not None else min(n_starts, 4)
    require(workers >= 1, "n_workers must be >= 1", workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(
                _run_single_start,
                i,
                x_start,
                lo,
                hi,
                partition,
                goal,
                config,
                tracker,
                event,
                max_nfev_per_start,
            )
            for i, x_start in enumerate(starts)
        }
        summaries = _collect(pending, event)

    summaries.sort(key=lambda s: s.seed)
    completed = [s for s in summaries if s.x is not None]
    if not completed:
        raise CancelledError("All solver starts were cancelled")

    best = min(completed, key=lambda s: s.cost)
    assert best.x is not None
    solution = partition.assemble(best.x)
    achieved = achieved_quantities(solution, partition, goal, config)
    per_goal = {name: achieved[name] - term.target for name, term in goal.items()}
    if goal.target_region is not None:
        # Region "error" (#4125 H7b): signed distance to the region
        # boundary at the achieved landing point (<= 0 means holding).
        per_goal["target_region_m"] = achieved["target_distance_m"]
    res_vec = residuals(best.x, partition, goal, config)
    residual_norm = float(np.linalg.norm(res_vec))
    require(
        math.isclose(0.5 * residual_norm**2, best.cost, rel_tol=1e-6, abs_tol=1e-9),
        "postcondition: re-evaluated cost must match the best start "
        "(objective must be deterministic)",
        (0.5 * residual_norm**2, best.cost),
    )

    return SolverResult(
        variables=solution,
        free_names=partition.free_names,
        x=best.x,
        achieved=achieved,
        per_goal_errors=per_goal,
        residual_norm=residual_norm,
        cost=best.cost,
        converged=best.converged,
        n_evals=sum(s.n_evals for s in summaries),
        elapsed_s=time.monotonic() - t0,
        starts=tuple(summaries),
    )


__all__ = [
    "CancelledError",
    "ProgressCallback",
    "ProgressReport",
    "SolverResult",
    "StartSummary",
    "detect_stall",
    "solve",
]
