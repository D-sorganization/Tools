"""Slew-limited direct-collocation optimizer for a golf downswing.

Solves one downswing optimal control problem per objective under identical
conditions, so the only difference between the resulting swings is what was being
maximized.

Two settings here are load-bearing rather than cosmetic, and both have a
regression test:

* **Non-dimensional decision vector** — see
  :mod:`double_pendulum_golf.swing_objectives.collocation`.
* **Tight solver tolerance** — at SciPy's default ``ftol`` the solver reports
  success as soon as it finds a feasible trajectory and returns the initial guess
  essentially unchanged.

Every result carries ``max_defect`` and a derived ``feasible`` flag, so a
trajectory that does not obey the equations of motion can never be mistaken for
an optimum on the strength of the solver's own success flag.

Closes #4769.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from double_pendulum_golf.swing_objectives.collocation import CollocationTranscription
from double_pendulum_golf.swing_objectives.downswing_config import DownswingConfig
from double_pendulum_golf.swing_objectives.objectives import (
    SwingObjective,
    get_objective,
)
from double_pendulum_golf.swing_objectives.signals import (
    SwingSignals,
    build_swing_signals,
)

logger = logging.getLogger(__name__)

__all__ = ["DownswingOptimizer", "DownswingResult", "DownswingConfig"]


FloatArray = npt.NDArray[np.float64]

#: Largest dynamics defect a trajectory may carry and still be called feasible.
FEASIBILITY_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class DownswingResult:
    """Outcome of optimizing a downswing for one objective.

    Attributes:
        objective: The objective that was maximized.
        objective_value: Achieved value in the objective's own units.
        states: ``(N, 4)`` optimal state trajectory.
        torques: ``(N, 2)`` optimal joint torques in N·m.
        signals: Per-sample signals, so any objective can rescore this swing.
        success: Whether the NLP solver reported convergence.
        message: Solver status message.
        iterations: Iterations used.
        max_defect: Largest absolute dynamics collocation defect.
        max_slew_violation: Largest torque slew-rate overshoot, zero when legal.
    """

    objective: SwingObjective
    objective_value: float
    states: FloatArray
    torques: FloatArray
    signals: SwingSignals
    success: bool
    message: str
    iterations: int
    max_defect: float
    max_slew_violation: float
    feasibility_tolerance: float = field(default=FEASIBILITY_TOLERANCE)

    @property
    def feasible(self) -> bool:
        """Whether the trajectory obeys the dynamics to a usable tolerance.

        Derived from the measured defect, never from the solver's success flag.
        """
        return bool(self.max_defect < self.feasibility_tolerance)


class DownswingOptimizer:
    """Optimizes a downswing for a chosen objective under fixed conditions.

    Args:
        config: The immutable downswing configuration shared by every objective.
    """

    def __init__(self, config: DownswingConfig) -> None:
        """Initialize the optimizer and its collocation transcription."""
        self._config = config
        self._transcription = CollocationTranscription(config)

    def initial_guess(self) -> tuple[FloatArray, FloatArray]:
        """Return the starting trajectory the solver is seeded with."""
        return self._transcription.initial_guess()

    def solve(self, objective: str | SwingObjective) -> DownswingResult:
        """Optimize the downswing for a single objective.

        Args:
            objective: Objective key such as ``"coriolis"``, or an instance.

        Returns:
            The optimization result, including the analyzed swing.

        Raises:
            KeyError: If the objective key is not registered.

        Pre: the configuration validated at construction.
        Post: the result reports its own dynamics defect and slew violation.
        """
        resolved = get_objective(objective)
        scales = self._transcription.variable_scales
        guess = self._transcription.pack(*self.initial_guess())

        logger.info(
            "Optimizing downswing for objective '%s' over %d nodes",
            resolved.key,
            self._config.node_count,
        )
        solution = minimize(
            fun=self._make_cost(resolved, scales),
            x0=guess / scales,
            method="SLSQP",
            bounds=self._transcription.scaled_bounds(),
            constraints=self._make_constraints(scales),
            options={
                "maxiter": self._config.max_iterations,
                "ftol": self._config.tolerance,
                "disp": False,
            },
        )
        return self._build_result(resolved, solution, scales)

    # --- Problem assembly -----------------------------------------------------

    def _make_cost(self, objective: SwingObjective, scales: FloatArray) -> Any:
        """Build the scaled, negated objective plus its effort regularizer."""
        config = self._config
        transcription = self._transcription
        limits = config.torque_limit_vector

        def cost(scaled_decision: FloatArray) -> float:
            states, torques = transcription.unpack(scaled_decision * scales)
            signals = build_swing_signals(config.time_grid, states, torques, config.params)
            effort = float(np.mean((torques / limits) ** 2))
            return -objective.evaluate(signals) / objective.scale + (
                config.effort_weight * effort
            )

        return cost

    def _make_constraints(self, scales: FloatArray) -> list[dict[str, Any]]:
        """Build the equality and inequality constraint set."""
        transcription = self._transcription
        defect_scale = transcription.defect_normalizer
        slew_scale = transcription.slew_normalizer

        def equalities(scaled_decision: FloatArray) -> FloatArray:
            states, torques = transcription.unpack(scaled_decision * scales)
            return np.concatenate(
                [
                    transcription.boundary_residuals(states),
                    transcription.defects(states, torques) / defect_scale,
                ]
            )

        constraints: list[dict[str, Any]] = [{"type": "eq", "fun": equalities}]
        if not self._config.limit_torque_rate:
            return constraints

        def inequalities(scaled_decision: FloatArray) -> FloatArray:
            _, torques = transcription.unpack(scaled_decision * scales)
            return transcription.slew_margins(torques) / slew_scale

        constraints.append({"type": "ineq", "fun": inequalities})
        return constraints

    def _build_result(
        self, objective: SwingObjective, solution: Any, scales: FloatArray
    ) -> DownswingResult:
        """Assemble the reported result from a raw solver solution."""
        config = self._config
        states, torques = self._transcription.unpack(solution.x * scales)
        signals = build_swing_signals(config.time_grid, states, torques, config.params)

        max_defect = float(np.max(np.abs(self._transcription.defects(states, torques))))
        margins = self._transcription.slew_margins(torques)
        max_slew_violation = float(max(0.0, -np.min(margins)))
        value = objective.evaluate(signals)

        logger.info(
            "Objective '%s': value=%.6g %s, success=%s, defect=%.2e",
            objective.key,
            value,
            objective.units,
            bool(solution.success),
            max_defect,
        )
        return DownswingResult(
            objective=objective,
            objective_value=value,
            states=states,
            torques=torques,
            signals=signals,
            success=bool(solution.success),
            message=str(solution.message),
            iterations=int(getattr(solution, "nit", -1)),
            max_defect=max_defect,
            max_slew_violation=max_slew_violation,
        )
