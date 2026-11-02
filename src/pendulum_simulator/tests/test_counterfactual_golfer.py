"""Tests for counterfactual_golfer module.

TDD: Tests verify zero-torque counterfactual for the golfer model.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.counterfactual_golfer import (
    zero_torque_accelerations,
    zero_torque_joint_forces,
)
from golfer_test_fixtures import default_golfer_params, nonsingular_golfer_state


class TestZeroTorqueAccelerations:
    """Tests for zero-torque counterfactual accelerations."""

    def test_returns_array(self) -> None:
        params = default_golfer_params()
        state = nonsingular_golfer_state()
        result = zero_torque_accelerations(state, params)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (8,)

    def test_finite_output(self) -> None:
        params = default_golfer_params()
        state = nonsingular_golfer_state()
        result = zero_torque_accelerations(state, params)
        assert np.all(np.isfinite(result)), f"Non-finite accelerations: {result}"


class TestZeroTorqueJointForces:
    """Tests for zero-torque counterfactual joint forces."""

    def test_returns_dict(self) -> None:
        params = default_golfer_params()
        state = nonsingular_golfer_state()
        result = zero_torque_joint_forces(state, params)
        assert isinstance(result, dict)
        assert result, "joint-force map must not be empty for nonsingular state"
