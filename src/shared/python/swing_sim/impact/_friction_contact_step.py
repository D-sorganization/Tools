"""Coupled backward-Euler mechanics and objective end-step return mapping."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root

from ...golf_club._grip_contracts import finite_array
from ._friction_contact_response import FrictionContactResponse, friction_response
from ._friction_trajectory_contracts import (
    FrictionTrajectoryControls,
    FrictionTrajectoryProblem,
    FrictionTrajectoryState,
)
from ._friction_transport import contact_transport
from ._normal_contact_trajectory import _shift
from ._normal_contact_work import normal_contact_work
from ._tangential_contact_work import (
    TangentialContactUpdate,
    advance_tangential_contact,
)


@dataclass
class EvaluationBudget:
    """Count actual body-response evaluations, including trial and final calls."""

    maximum: int
    used: int = 0

    def consume(self) -> None:
        if self.used >= self.maximum:
            raise ValueError("friction trajectory global evaluation budget exhausted")
        self.used += 1


@dataclass
class FrictionStep:
    """Fixed accepted history; repeated residual evaluations have no state effects."""

    problem: FrictionTrajectoryProblem
    previous: FrictionTrajectoryState
    response: FrictionContactResponse
    cell: tuple[float, float, float]
    controls: FrictionTrajectoryControls
    budget: EvaluationBudget
    evaluations: int = 0
    _jacobian_point: np.ndarray | None = field(default=None, init=False, repr=False)
    _jacobian_value: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def step_s(self) -> float:
        return self.cell[2] - self.cell[0]

    @property
    def velocity_scale(self) -> np.ndarray:
        normal = self.problem.normal
        contact = normal.contact
        scales = contact.controls.scales
        return np.array([scales.length_m] * 3 + [1.0] * 3) / scales.time_s

    def evaluate(
        self, velocities: np.ndarray
    ) -> tuple[
        FrictionTrajectoryState, FrictionContactResponse, TangentialContactUpdate
    ]:
        if self.evaluations >= self.controls.max_step_evaluations:
            raise ValueError("friction step evaluation budget exhausted")
        self.budget.consume()
        self.evaluations += 1
        mechanical = _shift(
            self.previous.mechanical, self.step_s * velocities, velocities
        )
        normal_problem = self.problem.normal
        model = normal_problem.contact_at(self.cell[2])
        contact = model.kinematics(mechanical.shaft, mechanical.ball)
        bodies = self.response.bodies
        old_contact = bodies.contact
        rotation = contact_transport(
            old_contact, contact, self.step_s, self.problem.transport
        )
        normal = normal_contact_work(model.law, -contact.gap_m, -contact.gap_rate_mps)
        direction = np.asarray(contact.normal)
        slip = np.cross(direction, np.cross(contact.relative_velocity_mps, direction))
        update = advance_tangential_contact(
            self.previous.tangential,
            self.step_s * slip,
            rotation,
            contact.normal,
            normal.force_n,
        )
        state = FrictionTrajectoryState(mechanical, update.state)
        response = friction_response(
            model, state, contact, normal_problem.ball_material_frame_id
        )
        return state, response, update

    def residual(self, scaled_velocities: np.ndarray) -> np.ndarray:
        previous = self.previous.mechanical
        shape = previous.twists.shape
        velocities = finite_array(
            scaled_velocities.reshape(shape) * self.velocity_scale,
            shape,
            "trial twist",
        )
        _, response, _ = self.evaluate(velocities)
        defect = velocities - previous.twists - self.step_s * response.rates
        return finite_array(
            defect / self.velocity_scale, shape, "scaled endpoint residual"
        ).ravel()

    def jacobian(self, scaled_velocities: np.ndarray) -> np.ndarray:
        """Difference on physical scales, including components near zero.

        MINPACK's component-relative perturbation can fall below roundoff in
        rotated configurations. A unit floor in dimensionless coordinates
        retains a representable perturbation. Every trial consumes the same
        evaluation budgets and starts from the same accepted contact history.
        """
        if (
            self._jacobian_value is not None
            and self._jacobian_point is not None
            and np.array_equal(scaled_velocities, self._jacobian_point)
        ):
            return self._jacobian_value.copy()
        baseline = self.residual(scaled_velocities)
        increments = np.sqrt(np.finfo(float).eps) * np.maximum(
            1.0, np.abs(scaled_velocities)
        )
        columns = []
        # These serial nonlinear solves cannot share mutable evaluation budgets.
        for index in range(scaled_velocities.size):
            trial = scaled_velocities.copy()
            trial[index] += increments[index]
            actual_increment = trial[index] - scaled_velocities[index]
            columns.append((self.residual(trial) - baseline) / actual_increment)
        self._jacobian_point = scaled_velocities.copy()
        self._jacobian_value = np.column_stack(columns)
        return self._jacobian_value.copy()

    def solve(
        self,
    ) -> tuple[
        FrictionTrajectoryState, FrictionContactResponse, TangentialContactUpdate, float
    ]:
        previous = self.previous.mechanical
        start = previous.twists
        guess = (start + self.step_s * self.response.rates) / self.velocity_scale
        tolerance = self.controls.scaled_residual_tolerance
        solved = root(
            self.residual,
            guess.ravel(),
            method="hybr",
            jac=self.jacobian,
            options={
                # MINPACK's recommended iterate criterion is distinct from
                # the caller's independently checked mechanical residual.
                "xtol": np.sqrt(np.finfo(float).eps),
                "maxfev": self.controls.max_step_evaluations,
            },
        )
        if not solved.success:
            raise ValueError(f"friction endpoint failed to converge: {solved.message}")
        velocities = np.asarray(solved.x).reshape(start.shape) * self.velocity_scale
        state, response, update = self.evaluate(velocities)
        defect = (
            velocities - start - self.step_s * response.rates
        ) / self.velocity_scale
        norm = float(
            np.max(np.abs(finite_array(defect, start.shape, "endpoint residual")))
        )
        if norm > tolerance:
            raise ValueError("friction endpoint residual exceeds convergence tolerance")
        return state, response, update, norm


__all__ = ()
