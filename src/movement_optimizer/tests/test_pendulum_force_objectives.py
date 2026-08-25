"""Movement Optimizer integration with the canonical force-attribution core."""

from __future__ import annotations

import numpy as np
import pytest
from shared.python.swing_sim import PendulumParameters

from movement_optimizer.models.pendulum_force_objectives import (
    analyze_pendulum_force_sources,
    coriolis_hand_path_impulse_cost,
)


def test_optimizer_adapter_reports_all_declared_force_sources() -> None:
    time = np.linspace(0.0, 0.4, 9)
    q = np.column_stack((0.1 + 0.6 * time, -0.8 + 0.3 * time))
    velocity = np.tile(np.array([3.5, -4.5]), (time.size, 1))
    torque = np.tile(np.array([15.0, -3.0]), (time.size, 1))

    result = analyze_pendulum_force_sources(
        PendulumParameters.golf_default(), time, q, velocity, torque
    )

    assert set(result.components) == {
        "coriolis",
        "squared_speed",
        "velocity_residual",
        "gravity",
        "damping",
        "applied",
    }
    assert result.metrics["coriolis"].signed_tangent_impulse_n_s is not None
    assert coriolis_hand_path_impulse_cost(
        PendulumParameters.golf_default(), time, q, velocity, torque
    ) == pytest.approx(-result.metrics["coriolis"].signed_tangent_impulse_n_s)


def test_optimizer_adapter_rejects_nonphysical_gravity() -> None:
    time = np.array([0.0, 0.1])
    history = np.zeros((2, 2))
    with pytest.raises(ValueError, match="gravity_m_s2"):
        analyze_pendulum_force_sources(
            PendulumParameters.golf_default(),
            time,
            history,
            history,
            history,
            gravity_m_s2=-1.0,
        )
