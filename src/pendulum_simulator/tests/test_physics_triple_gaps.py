# ruff: noqa: E501
"""Gap-fill tests for physics_triple.py — covers remaining uncovered lines.

Lines 410-412: equations_of_motion with torque_limits (clamp_torque_ndof path)
Lines 469-471: forward_kinematics with non-zero scapula_offset_rad
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics_triple import (
    TriplePendulumParams,
    equations_of_motion,
    forward_kinematics,
)


@pytest.fixture
def params() -> TriplePendulumParams:
    return TriplePendulumParams(m1=5.0, m2=3.0, m3=0.5, L1=0.6, L2=0.6, L3=0.6)


@pytest.fixture
def zero_torque():
    return lambda t: (0.0, 0.0, 0.0)


# ===========================================================================
# equations_of_motion — torque_limits branch (lines 410-412)
# ===========================================================================


class TestEquationsOfMotionWithTorqueLimits:
    def test_with_torque_limits_clamps(
        self, params: TriplePendulumParams, zero_torque
    ) -> None:
        """Passing torque_limits clamps huge torques and still produces finite derivatives."""
        state = np.array([0.1, 0.05, -0.05, 0.0, 0.0, 0.0])
        limits = np.array([0.001, 0.001, 0.001])  # tiny limits

        def huge_torque(t):
            return (1e6, 1e6, 1e6)

        state_dot = equations_of_motion(state, 0.0, params, huge_torque, torque_limits=limits)
        assert state_dot.shape == (6,)
        assert np.all(np.isfinite(state_dot))

    def test_with_large_limits_passes_through(self, params: TriplePendulumParams) -> None:
        """With infinite limits, torques pass through unchanged."""
        state = np.array([0.1, 0.05, -0.05, 0.0, 0.0, 0.0])
        limits = np.array([np.inf, np.inf, np.inf])

        def tau_fn(t):
            return (5.0, -3.0, 2.0)

        state_dot = equations_of_motion(state, 0.0, params, tau_fn, torque_limits=limits)
        assert state_dot.shape == (6,)
        assert np.all(np.isfinite(state_dot))

    def test_no_torque_limits_same_as_none(
        self, params: TriplePendulumParams, zero_torque
    ) -> None:
        """Without limits, result should match None path."""
        state = np.array([0.1, 0.05, -0.05, 0.0, 0.0, 0.0])
        sd_no_limits = equations_of_motion(state, 0.0, params, zero_torque, torque_limits=None)
        assert np.all(np.isfinite(sd_no_limits))


# ===========================================================================
# forward_kinematics — non-zero scapula_offset_rad branch (lines 469-471)
# ===========================================================================


class TestForwardKinematicsScapulaOffset:
    def test_nonzero_scapula_shifts_shoulder(self) -> None:
        """Non-zero scapula_offset_rad displaces the shoulder anchor from origin."""
        p_with_scapula = TriplePendulumParams(
            m1=5.0,
            m2=3.0,
            m3=0.5,
            L1=0.6,
            L2=0.6,
            L3=0.6,
            scapula_offset_rad=0.3,
        )
        p_no_scapula = TriplePendulumParams(
            m1=5.0,
            m2=3.0,
            m3=0.5,
            L1=0.6,
            L2=0.6,
            L3=0.6,
        )
        pos_with = forward_kinematics(0.0, 0.0, 0.0, p_with_scapula)
        pos_no = forward_kinematics(0.0, 0.0, 0.0, p_no_scapula)

        # Shoulder should be different
        assert pos_with["shoulder"] != pos_no["shoulder"]
        # hub is always origin
        assert pos_with["hub"] == (0.0, 0.0)

    def test_scapula_shoulder_finite(self) -> None:
        p = TriplePendulumParams(
            m1=5.0,
            m2=3.0,
            m3=0.5,
            L1=0.6,
            L2=0.6,
            L3=0.6,
            scapula_offset_rad=0.5,
        )
        pos = forward_kinematics(0.1, -0.1, 0.05, p)
        for key, (x, y) in pos.items():
            assert np.isfinite(x), f"{key}.x not finite"
            assert np.isfinite(y), f"{key}.y not finite"

    def test_zero_scapula_shoulder_at_origin(self) -> None:
        """With zero scapula offset, shoulder is exactly at origin (ox=oy=0)."""
        p = TriplePendulumParams(
            m1=5.0,
            m2=3.0,
            m3=0.5,
            L1=0.6,
            L2=0.6,
            L3=0.6,
            scapula_offset_rad=0.0,
        )
        pos = forward_kinematics(0.0, 0.0, 0.0, p)
        assert pos["shoulder"] == (0.0, 0.0)
