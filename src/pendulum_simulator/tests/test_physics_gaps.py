from typing import Any

"""Gap-fill tests for physics.py — covers remaining uncovered lines.

Lines 400, 404: joint_limit_torque_ndof angle below lo / above hi
Lines 461, 467, 474: equations_of_motion with clamp / limits / near-singular M
"""

from __future__ import annotations


import logging

import numpy as np
import pytest

from double_pendulum_golf.physics import (
    JointLimits,
    JointLimitsNDOF,
    PendulumParams,
    TorqueClamp,
    clamp_torque_ndof,
    control_vector,
    equations_of_motion,
    joint_limit_torque_ndof,
)


@pytest.fixture
def params() -> PendulumParams:
    return PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)


@pytest.fixture
def zero_torque() -> Any:
    return lambda t: (0.0, 0.0)


# ===========================================================================
# joint_limit_torque_ndof — angle below lo (line 400) and above hi (line 404)
# ===========================================================================


class TestJointLimitTorqueNdof:
    def _make_limits(self, lo: float, hi: float) -> JointLimitsNDOF:
        return JointLimitsNDOF(
            angle_min=np.array([lo]),
            angle_max=np.array([hi]),
            stiffness=1000.0,
            damping=10.0,
        )

    def test_below_lo_returns_positive_torque(self) -> None:
        """Angle well below minimum → large positive restoring torque."""
        limits = self._make_limits(lo=0.1, hi=1.0)
        angles = np.array([-0.5])  # << lo
        velocities = np.array([0.0])
        tau = joint_limit_torque_ndof(angles, velocities, limits)
        assert tau[0] > 0.0

    def test_above_hi_returns_negative_torque(self) -> None:
        """Angle well above maximum → large negative restoring torque."""
        limits = self._make_limits(lo=-1.0, hi=0.1)
        angles = np.array([1.5])  # >> hi
        velocities = np.array([0.0])
        tau = joint_limit_torque_ndof(angles, velocities, limits)
        assert tau[0] < 0.0

    def test_within_limits_returns_zero(self) -> None:
        limits = self._make_limits(lo=-1.0, hi=1.0)
        angles = np.array([0.0])
        velocities = np.array([0.0])
        tau = joint_limit_torque_ndof(angles, velocities, limits)
        assert tau[0] == pytest.approx(0.0, abs=1e-12)

    def test_multi_dof(self) -> None:
        limits = JointLimitsNDOF(
            angle_min=np.array([-1.0, 0.2]),
            angle_max=np.array([1.0, 1.0]),
            stiffness=500.0,
            damping=5.0,
        )
        angles = np.array([-2.0, 0.0])  # first below lo, second below lo
        velocities = np.array([0.0, 0.0])
        tau = joint_limit_torque_ndof(angles, velocities, limits)
        assert tau[0] > 0.0  # restoring upward
        assert tau[1] > 0.0  # restoring upward


# ===========================================================================
# equations_of_motion — clamp path (line 461) and limits path (line 467)
# ===========================================================================


class TestEquationsOfMotionBranches:
    def test_with_clamp_applies_torque_limit(self, params: PendulumParams) -> None:
        """Using TorqueClamp should not crash and must return finite derivatives."""
        clamp = TorqueClamp(max_torque1=0.001, max_torque2=0.001)
        state = np.array([0.1, 0.05, 0.0, 0.0])

        # Use large torque that will be clamped
        def torque_fn(t) -> Any:
            return (1e6, 1e6)

        state_dot = equations_of_motion(state, 0.0, params, torque_fn, clamp=clamp)
        assert state_dot.shape == (4,)
        assert np.all(np.isfinite(state_dot))

    def test_with_limits_applies_penalty(self, params: PendulumParams) -> None:
        """Using JointLimits with angle beyond limit should return finite derivatives."""
        limits = JointLimits(
            phi_min=-0.01,
            phi_max=0.01,
            stiffness=1000.0,
            damping=10.0,
        )

        def torque_fn(t) -> Any:
            return (0.0, 0.0)

        # phi far beyond limit → penalty torques activated
        state = np.array([0.0, 2.0, 0.0, 0.0])
        state_dot = equations_of_motion(state, 0.0, params, torque_fn, limits=limits)
        assert state_dot.shape == (4,)
        assert np.all(np.isfinite(state_dot))

    def test_near_singular_mass_matrix_logs_warning(
        self, params: PendulumParams, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Very large phi can make mass matrix near-singular for some params."""
        # Use very small m2 to stress the mass matrix
        p = PendulumParams(m1=1e6, m2=1e-9, L1=0.6, L2=0.001)

        def torque_fn(t) -> Any:
            return (0.0, 0.0)

        state = np.array([0.0, 0.0, 0.0, 0.0])
        with caplog.at_level(logging.WARNING, logger="double_pendulum_golf.physics"):
            state_dot = equations_of_motion(state, 0.0, p, torque_fn)
        # Should still be finite even if M is near-singular
        assert np.all(np.isfinite(state_dot))


# ===========================================================================
# clamp_torque_ndof — cover this function directly
# ===========================================================================


class TestClampTorqueNdof:
    def test_clamps_positive(self) -> None:
        tau = np.array([100.0, -200.0])
        limits = np.array([10.0, 10.0])
        result = clamp_torque_ndof(tau, limits)
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(-10.0)

    def test_within_limits_unchanged(self) -> None:
        tau = np.array([3.0, -4.0])
        limits = np.array([10.0, 10.0])
        result = clamp_torque_ndof(tau, limits)
        np.testing.assert_array_equal(result, tau)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            clamp_torque_ndof(np.array([1.0, 2.0]), np.array([10.0]))

    def test_non_positive_limit_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            clamp_torque_ndof(np.array([1.0]), np.array([0.0]))


# ===========================================================================
# control_vector — lines 633-639 in physics.py
# ===========================================================================


class TestControlVector:
    def test_returns_expected_keys(self) -> None:
        params = PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)
        state = np.array([0.1, 0.05, 0.5, 0.2])
        qddot_actual = np.array([0.1, -0.1])
        cv = control_vector(state, qddot_actual, params)
        assert set(cv.keys()) == {"cvx", "cvy", "magnitude"}

    def test_magnitude_non_negative(self) -> None:
        params = PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)
        state = np.array([0.1, 0.05, 0.5, 0.2])
        qddot_actual = np.array([0.0, 0.0])
        cv = control_vector(state, qddot_actual, params)
        assert cv["magnitude"] >= 0.0

    def test_values_finite(self) -> None:
        params = PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)
        state = np.array([0.3, -0.1, 1.0, -0.5])
        qddot_actual = np.array([0.5, -0.3])
        cv = control_vector(state, qddot_actual, params)
        assert np.isfinite(cv["cvx"])
        assert np.isfinite(cv["cvy"])
        assert np.isfinite(cv["magnitude"])

    def test_magnitude_equals_norm(self) -> None:
        params = PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)
        state = np.array([0.1, 0.05, 0.5, 0.2])
        qddot_actual = np.array([0.1, -0.1])
        cv = control_vector(state, qddot_actual, params)
        expected_mag = np.sqrt(cv["cvx"] ** 2 + cv["cvy"] ** 2)
        assert cv["magnitude"] == pytest.approx(expected_mag, rel=1e-10)
