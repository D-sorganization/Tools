"""Shared Gram-factor algebra for explicitly declared grip coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import Vector6, vector6


@dataclass(frozen=True)
class CoordinateImpedanceResponse:
    """Owned coordinate effort and energy/power; not a physical body wrench."""

    effort: Vector6
    inertial_energy_j: float
    elastic_energy_j: float
    dissipated_power_w: float
    input_power_w: float
    stored_energy_rate_w: float


def coordinate_impedance(
    factors: tuple[np.ndarray, np.ndarray, np.ndarray],
    state: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> CoordinateImpedanceResponse:
    """Evaluate validated six-axis factors/state without choosing pose semantics.

    Callers validate and own their factors and coordinates at the boundary.
    Squared norms retain nonnegative storage/loss even for singular factors;
    nonfinite effort, power, energy or closure is refused without repair.
    """
    mass, damping, stiffness = factors
    q, velocity, acceleration = state
    with np.errstate(over="ignore", invalid="ignore"):
        mass_velocity, mass_acceleration = mass @ velocity, mass @ acceleration
        damping_velocity = damping @ velocity
        elastic_q, elastic_velocity = stiffness @ q, stiffness @ velocity
        effort = (
            mass.T @ mass_acceleration
            + damping.T @ damping_velocity
            + stiffness.T @ elastic_q
        )
        dissipated = float(damping_velocity @ damping_velocity)
        input_power = float(effort @ velocity)
        energy_rate = float(
            mass_velocity @ mass_acceleration + elastic_q @ elastic_velocity
        )
        values = (
            float(mass_velocity @ mass_velocity / 2),
            float(elastic_q @ elastic_q / 2),
            dissipated,
            input_power,
            energy_rate,
            input_power - energy_rate - dissipated,
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("grip response must be finite")
    return CoordinateImpedanceResponse(
        vector6(effort, "coordinate effort"), *values[:5]
    )


__all__ = ()
