"""Private finite-duration normal contact with separate mechanical work ports.

Fixed local-chart RK4 samples contact at every stage. Contact switching,
spatial/time convergence and physical calibration remain separate obligations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ...golf_club._grip_contracts import finite_array
from ...golf_club._rkmk_step import RkmkStepModel, rkmk_step
from ...golf_club._shaft_load_history import PrescribedPointLoads
from ...golf_club._shaft_moving_contracts import MovingChainState
from ...golf_club._shaft_moving_trajectory import _chart_state
from ...golf_club._shaft_rkmk_trajectory import RkmkTrajectoryControls
from ...golf_club._shaft_se3 import exp_twist
from ...golf_club._shaft_trajectory_contracts import (
    AnchorHistory,
    MovingTrajectoryProblem,
)
from ...golf_club._validation import require_identifier
from ._normal_shaft_contact import NormalShaftContact, NormalShaftContactResponse
from ._spatial_contact_kinematics import ContactBodyState


@dataclass(frozen=True)
class NormalContactTrajectoryState:
    """Owned shaft and ball states; no ball inertia is added to the shaft."""

    shaft: MovingChainState
    ball: ContactBodyState

    def __post_init__(self) -> None:
        if not isinstance(self.shaft, MovingChainState):
            raise TypeError("shaft must be MovingChainState")
        if not isinstance(self.ball, ContactBodyState):
            raise TypeError("ball must be ContactBodyState")
        if self.shaft.observer_id != self.ball.observer_id:
            raise ValueError("shaft and ball observers must agree")

    @property
    def twists(self) -> np.ndarray:
        return np.vstack((np.asarray(self.shaft.twists), np.asarray(self.ball.twist)))


@dataclass(frozen=True)
class NormalContactTrajectoryProblem:
    """Fixed contact/shaft laws and explicit, consistent prescribed grips.

    Grip anchors and explicit additional applied loads may vary in time;
    contact loads vary with every evaluated state. Histories retain the
    existing fixed-material shaft and canonical external-power contracts.
    No missing history is interpolated or inferred from a player label.
    """

    contact: NormalShaftContact
    anchor_history: AnchorHistory
    ball_material_frame_id: str
    additional_load_history: PrescribedPointLoads | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.contact, NormalShaftContact):
            raise TypeError("contact must be NormalShaftContact")
        if not callable(self.anchor_history):
            raise TypeError("anchor history must be callable")
        if self.additional_load_history is not None and not isinstance(
            self.additional_load_history, PrescribedPointLoads
        ):
            raise TypeError("additional_load_history must be PrescribedPointLoads")
        identifier = require_identifier(
            self.ball_material_frame_id, "ball material frame"
        )
        inertia = self.contact.ball_inertia
        if identifier != inertia.material_frame_id:
            raise ValueError("ball material frame and inertia must agree")

    def evaluate(
        self, state: NormalContactTrajectoryState, time_s: float
    ) -> NormalShaftContactResponse:
        contact = self.contact_at(time_s)
        return contact.evaluate(state.shaft, state.ball, self.ball_material_frame_id)

    def contact_at(self, time_s: float) -> NormalShaftContact:
        """Resolve prescribed histories once per requested mechanical evaluation."""
        prescribed = MovingTrajectoryProblem(
            self.contact.chain,
            self.contact.controls,
            self.anchor_history,
            self.additional_load_history,
        )
        return replace(self.contact, chain=prescribed.chain_at(time_s))


def _powers(response: NormalShaftContactResponse) -> np.ndarray:
    energy = response.energy
    return np.array(
        [
            energy.external_power_w,
            energy.anchor_power_w,
            energy.grip_dissipated_power_w,
            energy.viscous_power_w,
            energy.cutoff_power_w,
        ]
    )


@dataclass(frozen=True)
class ContactWorkIntegrals:
    """SI work: external input, anchor output, grip/viscous/cutoff losses.

    Loss channels are nonnegative and disjoint. Cutoff storage removal is a
    constitutive bookkeeping term, never an acoustic energy measurement.
    """

    values: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        values = finite_array(self.values, (5,), "contact work integrals")
        if np.any(values[2:] < 0):
            raise ValueError("contact loss integrals must be nonnegative")
        object.__setattr__(self, "values", tuple(float(value) for value in values))

    def advance(
        self,
        responses: tuple[tuple[NormalShaftContactResponse, float], ...],
        step_s: float,
    ) -> ContactWorkIntegrals:
        stages = tuple((_powers(response), weight) for response, weight in responses)
        values = tuple(
            math.fsum(
                (
                    previous,
                    *(step_s * weight * power[index] for power, weight in stages),
                )
            )
            for index, previous in enumerate(self.values)
        )
        return ContactWorkIntegrals(values)


@dataclass(frozen=True)
class NormalContactTrajectorySample:
    """Owned endpoint and uncorrected mechanical energy-balance error."""

    time_s: float
    state: NormalContactTrajectoryState
    response: NormalShaftContactResponse
    work: ContactWorkIntegrals
    energy_balance_error_j: float


@dataclass(frozen=True)
class NormalContactTrajectory:
    """Complete requested grid; no event resolution or physical claim implied."""

    controls: RkmkTrajectoryControls
    samples: tuple[NormalContactTrajectorySample, ...]
    evaluation_count: int

    @property
    def evidence_status(self) -> str:
        return "time-discrete-contact-unqualified"


def _shift(
    state: NormalContactTrajectoryState,
    coordinates: np.ndarray,
    velocities: np.ndarray,
) -> NormalContactTrajectoryState:
    shaft = _chart_state(state.shaft, coordinates[:-1], velocities[:-1])
    ball = replace(
        state.ball,
        pose=np.asarray(state.ball.pose) @ exp_twist(coordinates[-1]),
        twist=velocities[-1],
    )
    return NormalContactTrajectoryState(shaft, ball)


def _rates(response: NormalShaftContactResponse) -> np.ndarray:
    return np.vstack((response.shaft.twist_rates, response.ball.twist_rate))


@dataclass(frozen=True)
class _Ledger:
    initial_energy_j: float
    work: ContactWorkIntegrals = ContactWorkIntegrals()

    def sample(
        self,
        time_s: float,
        state: NormalContactTrajectoryState,
        response: NormalShaftContactResponse,
    ) -> NormalContactTrajectorySample:
        external, *outputs = self.work.values
        energy = response.energy
        defect = math.fsum(
            (energy.total_energy_j, -self.initial_energy_j, -external, *outputs)
        )
        finite_array(defect, (), "contact trajectory energy balance")
        return NormalContactTrajectorySample(time_s, state, response, self.work, defect)


def integrate_normal_contact(
    problem: NormalContactTrajectoryProblem,
    initial: NormalContactTrajectoryState,
    controls: RkmkTrajectoryControls,
) -> NormalContactTrajectory:
    """Advance all poses/twists and disjoint work with the shared Lie RK4.

    Preflight the complete finite grid and evaluation budget before history
    access. Validate state/constitutive domains at every stage and endpoint.
    Return every requested endpoint or raise without a partial trajectory.
    Unilateral-force jumps and contact opening require independent event/time
    refinement; fixed-step success alone cannot qualify impact peaks or sound.
    """
    if not isinstance(problem, NormalContactTrajectoryProblem):
        raise TypeError("problem must be NormalContactTrajectoryProblem")
    if not isinstance(initial, NormalContactTrajectoryState):
        raise TypeError("initial must be NormalContactTrajectoryState")
    if type(controls) is not RkmkTrajectoryControls:
        raise TypeError("normal contact requires RkmkTrajectoryControls")
    cells = controls.time_cells()
    model = RkmkStepModel(_shift, problem.evaluate, lambda state: state.twists, _rates)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            response = problem.evaluate(initial, controls.bounds_s[0])
            energy = response.energy
            ledger = _Ledger(energy.total_energy_j)
            samples = [ledger.sample(controls.bounds_s[0], initial, response)]
            state = initial
            for cell in cells:
                state, response, stages = rkmk_step(model, state, response, cell)
                work = ledger.work.advance(stages, cell[2] - cell[0])
                ledger = replace(ledger, work=work)
                samples.append(ledger.sample(cell[2], state, response))
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError(
            "normal contact trajectory numerical evaluation failed"
        ) from error
    return NormalContactTrajectory(controls, tuple(samples), 4 * controls.steps + 1)


__all__ = ()
