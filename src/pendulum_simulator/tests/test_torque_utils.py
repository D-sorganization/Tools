"""Tests for torque_utils module.

TDD: Tests cover polynomial torque generation for N-joint models.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.torque_utils import make_polynomial_torque


class TestMakePolynomialTorque:
    """Contract and behavior tests for make_polynomial_torque."""

    def test_single_joint_constant(self):
        """Constant torque: tau(t) = 5 for all t."""
        tf = make_polynomial_torque([5.0])
        assert tf(0.0) == pytest.approx((5.0,))
        assert tf(1.0) == pytest.approx((5.0,))
        assert tf(100.0) == pytest.approx((5.0,))

    def test_single_joint_linear(self):
        """Linear torque: tau(t) = 2 + 3*t."""
        tf = make_polynomial_torque([2.0, 3.0])
        assert tf(0.0) == pytest.approx((2.0,))
        assert tf(1.0) == pytest.approx((5.0,))
        assert tf(2.0) == pytest.approx((8.0,))

    def test_two_joints(self):
        """Two joints: tau1 = 1, tau2 = t."""
        tf = make_polynomial_torque([1.0], [0.0, 1.0])
        result = tf(3.0)
        assert len(result) == 2
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(3.0)

    def test_seven_joints(self):
        """Golfer model with 7 joints all constant."""
        coeffs = [[float(i)] for i in range(7)]
        tf = make_polynomial_torque(*coeffs)
        result = tf(0.0)
        assert len(result) == 7
        for i in range(7):
            assert result[i] == pytest.approx(float(i))

    def test_quadratic(self):
        """tau(t) = 1 + 2*t + 3*t^2 at t=2 → 1+4+12 = 17."""
        tf = make_polynomial_torque([1.0, 2.0, 3.0])
        assert tf(2.0) == pytest.approx((17.0,))

    def test_zero_joints_raises(self):
        """Must have at least one joint."""
        with pytest.raises((ValueError, TypeError), match="Need at least one joint"):
            make_polynomial_torque()

    def test_empty_coefficients_raises(self):
        """Each joint needs at least one coefficient."""
        with pytest.raises((ValueError, TypeError), match="Need at least one coefficient"):
            make_polynomial_torque([])

    def test_returns_tuple(self):
        """Return type is always tuple."""
        tf = make_polynomial_torque([1.0])
        result = tf(0.0)
        assert isinstance(result, tuple)

    def test_all_values_finite(self):
        """Output must be finite for finite input."""
        tf = make_polynomial_torque([1.0, 2.0], [3.0, 4.0])
        result = tf(1.5)
        assert all(np.isfinite(v) for v in result)
