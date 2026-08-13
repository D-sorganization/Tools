"""Miss-safe scalar extraction for optional simulation phases."""

from __future__ import annotations

import math

import numpy as np

from rate_of_closure.simulation.session import SimulationRun


def path_deg(run: SimulationRun) -> float:
    """Return club path or NaN when delivery is absent."""
    if run.delivery is None:
        return math.nan
    velocity = run.delivery.clubhead_velocity
    return math.degrees(math.atan2(float(velocity[2]), float(velocity[0])))


def attack_angle_deg(run: SimulationRun) -> float:
    """Return attack angle or NaN when delivery is absent."""
    if run.delivery is None:
        return math.nan
    velocity = run.delivery.clubhead_velocity
    horizontal_speed = math.hypot(float(velocity[0]), float(velocity[2]))
    return math.degrees(math.atan2(float(velocity[1]), horizontal_speed))


def optional_float(value: float | None) -> float:
    """Return NaN as the plotting sentinel for an absent scalar."""
    return math.nan if value is None else float(value)


def delivery_scalar(run: SimulationRun, attribute: str) -> float:
    """Extract a delivery scalar or NaN when the run missed."""
    if run.delivery is None:
        return math.nan
    return float(getattr(run.delivery, attribute))


def delivered_speed(run: SimulationRun) -> float:
    """Extract delivered clubhead speed or NaN when the run missed."""
    if run.delivery is None:
        return math.nan
    return float(np.linalg.norm(run.delivery.clubhead_velocity))


def impact_energy(run: SimulationRun) -> float:
    """Extract transferred energy or NaN when the run missed."""
    if run.post_impact is None:
        return math.nan
    return float(run.post_impact.energy_transfer)


def launch_scalar(run: SimulationRun, key: str) -> float:
    """Extract a launch/flight metric or NaN when the run missed."""
    if run.launch is None:
        return math.nan
    return float(run.launch[key])
