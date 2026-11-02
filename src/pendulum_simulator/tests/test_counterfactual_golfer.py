"""Tests for counterfactual_golfer module.

TDD: Tests verify zero-torque counterfactual for the golfer model.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.constraint_solver import (
    project_to_constraints,
    project_velocity,
)
from double_pendulum_golf.counterfactual_golfer import (
    zero_torque_accelerations,
    zero_torque_joint_forces,
)
from double_pendulum_golf.physics_golfer import N_DOF, GolferParams


def _default_golfer_params() -> GolferParams:
    """Create minimal valid golfer params for testing."""
    return GolferParams(
        m_hub=10.0,
        L_hub=0.3,
        m_r_upper=2.0,
        L_r_upper=0.3,
        m_r_fore=1.5,
        L_r_fore=0.25,
        m_l_upper=2.0,
        L_l_upper=0.3,
        m_l_fore=1.5,
        L_l_fore=0.25,
        m_club=0.3,
        L_club=1.1,
        d_rs=0.1,
        d_ls=0.1,
        grip_right=0.1,
        grip_left=0.3,
        g=9.81,
    )


def _manifold_state(params: GolferParams, hub_angle: float = 0.1) -> np.ndarray:
    """Build a constraint-consistent state for nonsingular counterfactual checks."""
    q = project_to_constraints(np.zeros(N_DOF), params)
    q[0] = hub_angle
    qdot = project_velocity(q, np.zeros(N_DOF), params)
    return np.concatenate([q, qdot])


class TestZeroTorqueAccelerations:
    """Tests for zero-torque counterfactual accelerations."""

    def test_returns_array(self) -> None:
        params = _default_golfer_params()
        state = _manifold_state(params)
        result = zero_torque_accelerations(state, params)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (N_DOF,)

    def test_finite_output(self) -> None:
        params = _default_golfer_params()
        state = _manifold_state(params)
        result = zero_torque_accelerations(state, params)
        assert np.all(np.isfinite(result)), f"Non-finite accelerations: {result}"

    def test_invalid_state_shape_fails(self) -> None:
        params = _default_golfer_params()
        with pytest.raises(ValueError, match="State shape"):
            zero_torque_accelerations(np.zeros(8), params)


class TestZeroTorqueJointForces:
    """Tests for zero-torque counterfactual joint forces."""

    def test_returns_dict(self) -> None:
        params = _default_golfer_params()
        state = _manifold_state(params)
        result = zero_torque_joint_forces(state, params)
        assert isinstance(result, dict)
        assert result, "expected named joint force entries"
