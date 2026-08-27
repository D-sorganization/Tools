# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Movement-Optimizer façade for canonical pendulum force objectives.

The mechanics remain in the public ``shared.python.swing_sim`` DRY layer.  This
module only adapts movement-optimizer candidate trajectories to that contract.
"""

from __future__ import annotations

import math

import numpy as np
from shared.python.swing_sim.force_attribution import (
    DoublePendulumAttributionProvider,
    TrajectoryAttribution,
    attribute_trajectory,
    component_impulse_objective,
)
from shared.python.swing_sim.types import PendulumParameters


def analyze_pendulum_force_sources(
    parameters: PendulumParameters,
    time_s: np.ndarray,
    q: np.ndarray,
    velocity: np.ndarray,
    applied_torque_nm: np.ndarray,
    *,
    gravity_m_s2: float = 9.80665,
) -> TrajectoryAttribution:
    """Return source histories and integrals for one optimizer candidate.

    Preconditions:
        ``gravity_m_s2`` is finite and non-negative. Trajectory shape and
        finiteness contracts are enforced by the canonical attribution layer.
    """
    if not math.isfinite(gravity_m_s2) or gravity_m_s2 < 0.0:
        raise ValueError("gravity_m_s2 must be finite and non-negative")
    provider = DoublePendulumAttributionProvider(
        parameters,
        g_inplane=(0.0, -gravity_m_s2),
    )
    return attribute_trajectory(provider, time_s, q, velocity, applied_torque_nm)


def coriolis_hand_path_impulse_cost(
    parameters: PendulumParameters,
    time_s: np.ndarray,
    q: np.ndarray,
    velocity: np.ndarray,
    applied_torque_nm: np.ndarray,
    *,
    gravity_m_s2: float = 9.80665,
    absolute: bool = False,
) -> float:
    """Return the minimizer cost for maximum Coriolis hand-path impulse.

    This is a model objective, not a claim that maximizing one coordinate term
    maximizes clubhead speed, performance, or biological efficiency.
    """
    attribution = analyze_pendulum_force_sources(
        parameters,
        time_s,
        q,
        velocity,
        applied_torque_nm,
        gravity_m_s2=gravity_m_s2,
    )
    return component_impulse_objective(
        attribution,
        "coriolis",
        absolute=absolute,
    )


__all__ = [
    "analyze_pendulum_force_sources",
    "coriolis_hand_path_impulse_cost",
]
