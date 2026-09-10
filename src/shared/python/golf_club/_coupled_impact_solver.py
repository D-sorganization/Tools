"""One event-resolved transient implementation for legacy and audited results."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from ..swing_sim.impact._normal_contact_work import _normal_loss_rates
from ..swing_sim.impact.contact import KelvinVoigtContactLaw
from ._coupled_impact_state import (
    CoupledImpactAudit,
    CoupledImpactEnergyLedger,
    CoupledImpactInitialState,
)

if TYPE_CHECKING:
    from .impact_coupling import CoupledImpactConfig

_RELATIVE_TOLERANCE = 2e-9
_ABSOLUTE_TOLERANCE = 1e-11
_MAX_RATE_STEP = 0.25


class _ContactEvent:
    direction = -1.0

    def __init__(self, config: CoupledImpactConfig, *, force: bool) -> None:
        self.config = config
        self.force = force
        self.terminal = not force

    def __call__(self, time_s: float, state: NDArray[np.float64]) -> float:
        overlap = float(state[1] - state[0])
        if self.force:
            return float(
                self.config.contact_stiffness_n_m * overlap
                + self.config.contact_damping_n_s_m * float(state[4] - state[3])
            )
        return overlap


def _check_resolution(config: CoupledImpactConfig) -> None:
    """Refuse grossly under-resolved rate scales; refinement is still required."""
    minimum_mass = min(
        config.ball_mass_kg, config.head_mass_kg, config.grip.effective_mass_kg
    )
    stiffness = (
        config.contact_stiffness_n_m
        + config.shaft_stiffness_n_m
        + config.grip.stiffness_n_m
    )
    damping = (
        config.contact_damping_n_s_m
        + config.shaft_damping_n_s_m
        + config.grip.damping_n_s_m
    )
    rate = max(math.sqrt(2 * stiffness / minimum_mass), 2 * damping / minimum_mass)
    if not math.isfinite(rate) or rate * config.dt_s > _MAX_RATE_STEP:
        raise ValueError("dt_s does not resolve the configured stiffness/damping rates")


def _spring_energy(config: CoupledImpactConfig, state: NDArray[np.float64]) -> float:
    return float(
        0.5 * config.shaft_stiffness_n_m * (state[2] - state[1]) ** 2
        + 0.5 * config.grip.stiffness_n_m * state[2] ** 2
    )


def _energy(config: CoupledImpactConfig, state: NDArray[np.float64]) -> float:
    return float(
        0.5
        * (
            config.ball_mass_kg * state[3] ** 2
            + config.head_mass_kg * state[4] ** 2
            + config.grip.effective_mass_kg * state[5] ** 2
        )
        + _spring_energy(config, state)
        + 0.5 * config.contact_stiffness_n_m * max(0, state[1] - state[0]) ** 2
    )


def _initial_vector(
    config: CoupledImpactConfig, initial_state: CoupledImpactInitialState
) -> NDArray[np.float64]:
    """Construct first-touch coordinates without removing supplied preload."""
    head_x, grip_x = (
        initial_state.head_displacement_m,
        initial_state.grip_displacement_m,
    )
    head_v, grip_v = initial_state.head_velocity_mps, initial_state.grip_velocity_mps
    relative_velocity = head_v - (head_v - config.head_speed_mps)
    if not math.isclose(relative_velocity, config.head_speed_mps, rel_tol=1e-9):
        raise ValueError("initial relative velocity is not representable in this frame")
    return np.array(
        [
            head_x,
            head_x,
            grip_x,
            head_v - config.head_speed_mps,
            head_v,
            grip_v,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )


class _ImpactDerivative:
    """Shared force law and its disjoint work channels."""

    def __init__(self, config: CoupledImpactConfig) -> None:
        self.config = config
        self.law = KelvinVoigtContactLaw(
            config.contact_stiffness_n_m,
            config.contact_damping_n_s_m,
            np.finfo(float).max,
        )

    def __call__(
        self, time_s: float, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        config, law = self.config, self.law
        # Scalar force/work algebra does not need NumPy scalar dispatch at
        # every ODE stage. Convert once, preserving the float64 coordinates.
        x_b, x_h, x_g, v_b, v_h, v_g = map(float, state[:6])
        overlap, rate = x_h - x_b, v_h - v_b
        force = law.normal_force(float(overlap), float(rate))
        shaft = config.shaft_stiffness_n_m * (
            x_g - x_h
        ) + config.shaft_damping_n_s_m * (v_g - v_h)
        grip = -config.grip.stiffness_n_m * x_g - config.grip.damping_n_s_m * v_g
        viscous, cutoff = _normal_loss_rates(law, overlap, rate, force)
        return np.array(
            [
                v_b,
                v_h,
                v_g,
                force / config.ball_mass_kg,
                (-force + shaft) / config.head_mass_kg,
                (-shaft + grip) / config.grip.effective_mass_kg,
                viscous,
                cutoff,
                config.shaft_damping_n_s_m * (v_g - v_h) ** 2,
                config.grip.damping_n_s_m * v_g**2,
                force,
            ]
        )


def _audit_result(
    config: CoupledImpactConfig,
    start: NDArray[np.float64],
    states: NDArray[np.float64],
    events: tuple[NDArray[np.float64], ...],
) -> CoupledImpactAudit:
    """Assemble a separated state with independently integrated losses."""
    law = _ImpactDerivative(config).law
    end = states[:, -1]
    if not np.all(np.isfinite(end)):
        raise RuntimeError("nonfinite terminal impact state")
    ledger = CoupledImpactEnergyLedger(
        _energy(config, start),
        _energy(config, end),
        *(float(value) for value in end[6:10]),
    )
    clearance = float(events[0][0])
    release = float(events[1][0]) if len(events[1]) else clearance
    forces = [
        law.normal_force(float(s[1] - s[0]), float(s[4] - s[3])) for s in states.T
    ]
    # Include the right-hand limit at first touch (a KV dashpot force jumps).
    peak = max(max(forces), config.contact_damping_n_s_m * config.head_speed_mps)
    return CoupledImpactAudit(
        float(end[3]),
        float(end[4]),
        float(end[5]),
        clearance,
        release,
        peak,
        float(end[10]),
        float(end[1] - end[0]),
        _spring_energy(config, end),
        ledger,
    )


def _solve_coupled_impact(
    config: CoupledImpactConfig,
    *,
    initial_state: CoupledImpactInitialState | None = None,
) -> CoupledImpactAudit:
    """Integrate a validated, resolved first-touch state to geometric separation.

    DOP853 controls local error; dt_s bounds accepted steps. Postconditions:
    finite separated state and passive losses. Exhaustion raises RuntimeError.
    The v1 law has no force cap; no new cap is silently introduced here.
    """
    if initial_state is None:
        initial_state = CoupledImpactInitialState()
    if not isinstance(initial_state, CoupledImpactInitialState):
        raise TypeError("initial_state must be CoupledImpactInitialState")
    _check_resolution(config)
    start = _initial_vector(config, initial_state)
    solution = solve_ivp(
        _ImpactDerivative(config),
        (0, config.max_time_s),
        start,
        method="DOP853",
        max_step=config.dt_s,
        rtol=_RELATIVE_TOLERANCE,
        atol=_ABSOLUTE_TOLERANCE,
        events=(_ContactEvent(config, force=False), _ContactEvent(config, force=True)),
    )
    if not solution.success or len(solution.t_events[0]) != 1:
        raise RuntimeError(
            "contact separation was not reached; increase max_time_s or inspect model"
        )
    return _audit_result(config, start, solution.y, tuple(solution.t_events))


def _legacy_integrate(
    config: CoupledImpactConfig, *, coupled: bool
) -> tuple[float, ...]:
    """One solver for old results and the additive audit; no duplicated physics."""
    if not coupled:
        config = replace(config, shaft_stiffness_n_m=0.0, shaft_damping_n_s_m=0.0)
    result = _solve_coupled_impact(config)
    return (
        result.ball_velocity_mps,
        result.head_velocity_mps,
        result.grip_velocity_mps,
        result.clearance_time_s,
        result.peak_contact_force_n,
        result.stored_spring_energy_j,
    )
