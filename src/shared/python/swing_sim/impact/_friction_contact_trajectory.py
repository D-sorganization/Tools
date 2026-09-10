"""First-order coupled friction dynamics with an uncorrected energy ledger."""

import math
from dataclasses import dataclass, replace

import numpy as np

from ...golf_club._validation import Vector3, require_finite_float, require_vector3
from ._friction_contact_response import FrictionContactResponse, initial_response
from ._friction_contact_step import EvaluationBudget, FrictionStep
from ._friction_trajectory_contracts import (
    FrictionConvergence,
    FrictionTermination,
    FrictionTrajectoryControls,
    FrictionTrajectoryProblem,
    FrictionTrajectoryState,
)
from ._normal_contact_trajectory import ContactWorkIntegrals
from ._tangential_contact_work import TangentialContactUpdate


@dataclass(frozen=True)
class FrictionTrajectorySample:
    """Accepted state, constitutive losses and mechanical integration error."""

    time_s: float
    state: FrictionTrajectoryState
    response: FrictionContactResponse
    work: ContactWorkIntegrals
    plastic_dissipation_j: float
    tangential_algorithmic_loss_j: float
    normal_impulse_ns: float
    tangential_impulse_ns: Vector3
    energy_balance_error_j: float
    convergence: FrictionConvergence

    @property
    def scaled_solver_residual(self) -> float:
        return self.convergence.scaled_residual


@dataclass(frozen=True)
class FrictionTrajectory:
    """Every requested endpoint or an exception; no event or physical qualification."""

    controls: FrictionTrajectoryControls
    samples: tuple[FrictionTrajectorySample, ...]
    evaluation_count: int

    @property
    def evidence_status(self) -> str:
        return "friction-trajectory-unqualified"


@dataclass(frozen=True)
class _Ledger:
    initial_energy_j: float
    work: ContactWorkIntegrals = ContactWorkIntegrals()
    plastic_j: float = 0.0
    algorithmic_j: float = 0.0
    normal_impulse_ns: float = 0.0
    tangential_impulse_ns: Vector3 = (0.0, 0.0, 0.0)

    def advance(
        self,
        response: FrictionContactResponse,
        update: TangentialContactUpdate,
        step_s: float,
    ) -> "_Ledger":
        work = ContactWorkIntegrals(
            tuple(np.asarray(self.work.values) + step_s * response.work_powers_w)
        )
        return replace(
            self,
            work=work,
            plastic_j=require_finite_float(
                self.plastic_j + update.plastic_dissipation_j, "plastic loss"
            ),
            algorithmic_j=require_finite_float(
                self.algorithmic_j + update.algorithmic_loss_j, "algorithmic loss"
            ),
            normal_impulse_ns=require_finite_float(
                self.normal_impulse_ns + step_s * response.normal.force_n,
                "normal impulse",
            ),
            tangential_impulse_ns=require_vector3(
                np.asarray(self.tangential_impulse_ns)
                + step_s * np.asarray(response.tangential_force_n),
                "tangential impulse",
            ),
        )

    def sample(
        self,
        time_s: float,
        state: FrictionTrajectoryState,
        response: FrictionContactResponse,
        convergence: FrictionConvergence,
    ) -> FrictionTrajectorySample:
        external, *outputs = self.work.values
        defect = math.fsum(
            (
                response.mechanical_normal_energy_j,
                state.tangential.elastic_energy_j,
                -self.initial_energy_j,
                -external,
                *outputs,
                self.plastic_j,
                self.algorithmic_j,
            )
        )
        return FrictionTrajectorySample(
            time_s,
            state,
            response,
            self.work,
            self.plastic_j,
            self.algorithmic_j,
            self.normal_impulse_ns,
            self.tangential_impulse_ns,
            require_finite_float(defect, "mechanical integration defect"),
            convergence,
        )


def integrate_friction_contact(
    problem: FrictionTrajectoryProblem,
    initial: FrictionTrajectoryState,
    controls: FrictionTrajectoryControls,
) -> FrictionTrajectory:
    """Solve endpoint mechanics and friction together on a preflighted SI grid.

    Backward Euler on body twists and exponential endpoint pose updates are
    first order. The constitutive return map uses endpoint material slip and
    explicit frame transport. Its history changes only after convergence.
    Numerical damping, contact opening, force peaks and spin require independent
    time/contact-mode refinement. No smoothing, gear effect or energy correction
    is applied. Prescribed histories must be deterministic and consistent.
    """
    for value, expected in (
        (problem, FrictionTrajectoryProblem),
        (initial, FrictionTrajectoryState),
        (controls, FrictionTrajectoryControls),
    ):
        if type(value) is not expected:
            raise TypeError(f"expected {expected.__name__}")
    cells = controls.time_cells()
    budget = EvaluationBudget(controls.max_evaluations)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            budget.consume()
            response = initial_response(problem, initial, controls.bounds_s[0])
            ledger = _Ledger(
                response.mechanical_normal_energy_j
                + initial.tangential.elastic_energy_j
            )
            state = initial
            initial_convergence = FrictionConvergence(FrictionTermination.INITIAL, 0.0)
            samples = [
                ledger.sample(
                    controls.bounds_s[0], state, response, initial_convergence
                )
            ]
            for cell in cells:
                solve = FrictionStep(problem, state, response, cell, controls, budget)
                result = solve.solve()
                state, response = result.state, result.response
                ledger = ledger.advance(response, result.update, cell[2] - cell[0])
                samples.append(
                    ledger.sample(cell[2], state, response, result.convergence)
                )
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("friction trajectory numerical evaluation failed") from error
    return FrictionTrajectory(controls, tuple(samples), budget.used)


__all__ = ()
