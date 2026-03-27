"""Tests for dynamics_quantities module — impulse, work, power.

TDD: These tests define the expected interface and behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.dynamics_quantities import (
    angular_impulse_series,
    angular_power_at,
    angular_power_series,
    angular_work_series,
    compute_all_dynamics,
    linear_impulse_series,
    linear_power_at,
    linear_power_series,
    linear_work_series,
)

# ---------------------------------------------------------------------------
# angular_power_at
# ---------------------------------------------------------------------------


class TestAngularPowerAt:
    """Unit tests for single-timestep angular power."""

    def test_positive_torque_positive_velocity(self):
        assert angular_power_at(10.0, 2.0) == pytest.approx(20.0)

    def test_negative_torque_positive_velocity(self):
        assert angular_power_at(-5.0, 3.0) == pytest.approx(-15.0)

    def test_zero_torque(self):
        assert angular_power_at(0.0, 5.0) == 0.0

    def test_zero_velocity(self):
        assert angular_power_at(5.0, 0.0) == 0.0

    def test_nan_torque_raises(self):
        with pytest.raises((ValueError, TypeError), match="torque must be finite"):
            angular_power_at(float("nan"), 1.0)

    def test_inf_velocity_raises(self):
        with pytest.raises((ValueError, TypeError), match="omega must be finite"):
            angular_power_at(1.0, float("inf"))


# ---------------------------------------------------------------------------
# linear_power_at
# ---------------------------------------------------------------------------


class TestLinearPowerAt:
    """Unit tests for single-timestep linear power."""

    def test_aligned_force_velocity(self):
        assert linear_power_at(np.array([1.0, 0.0]), np.array([3.0, 0.0])) == pytest.approx(
            3.0
        )

    def test_orthogonal_force_velocity(self):
        assert linear_power_at(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(
            0.0
        )

    def test_2d_dot_product(self):
        assert linear_power_at(np.array([2.0, 3.0]), np.array([4.0, 5.0])) == pytest.approx(
            23.0
        )

    def test_wrong_shape_raises(self):
        with pytest.raises((ValueError, TypeError), match="force must be shape"):
            linear_power_at(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Series computations
# ---------------------------------------------------------------------------


class TestAngularPowerSeries:
    """Tests for vectorised angular power."""

    def test_constant_torque_and_velocity(self):
        tau = np.full(10, 5.0)
        omega = np.full(10, 2.0)
        result = angular_power_series(tau, omega)
        np.testing.assert_allclose(result, 10.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises((ValueError, TypeError), match="Shape mismatch"):
            angular_power_series(np.ones(5), np.ones(6))


class TestLinearPowerSeries:
    """Tests for vectorised linear power."""

    def test_constant_force_velocity(self):
        F = np.tile([1.0, 2.0], (5, 1))
        v = np.tile([3.0, 4.0], (5, 1))
        result = linear_power_series(F, v)
        np.testing.assert_allclose(result, 11.0)  # 1*3 + 2*4


# ---------------------------------------------------------------------------
# Work (cumulative integrals)
# ---------------------------------------------------------------------------


class TestAngularWork:
    """Tests for cumulative angular work."""

    def test_starts_at_zero(self):
        t = np.linspace(0, 1, 100)
        tau = np.ones(100)
        omega = np.ones(100)
        W = angular_work_series(tau, omega, t)
        assert W[0] == pytest.approx(0.0)

    def test_constant_power_linear_work(self):
        """Constant power P=1 over [0,1] → W(1)=1."""
        t = np.linspace(0, 1, 1000)
        tau = np.ones(1000)
        omega = np.ones(1000)
        W = angular_work_series(tau, omega, t)
        assert W[-1] == pytest.approx(1.0, rel=1e-3)

    def test_zero_power_zero_work(self):
        t = np.linspace(0, 1, 50)
        tau = np.zeros(50)
        omega = np.ones(50)
        W = angular_work_series(tau, omega, t)
        np.testing.assert_allclose(W, 0.0)


class TestLinearWork:
    """Tests for cumulative linear work."""

    def test_starts_at_zero(self):
        t = np.linspace(0, 1, 50)
        F = np.tile([1.0, 0.0], (50, 1))
        v = np.tile([1.0, 0.0], (50, 1))
        W = linear_work_series(F, v, t)
        assert W[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Impulse
# ---------------------------------------------------------------------------


class TestAngularImpulse:
    """Tests for cumulative angular impulse."""

    def test_starts_at_zero(self):
        t = np.linspace(0, 1, 50)
        tau = np.ones(50)
        J = angular_impulse_series(tau, t)
        assert J[0] == pytest.approx(0.0)

    def test_constant_torque(self):
        """Constant tau=2 over [0,1] → J=2."""
        t = np.linspace(0, 1, 1000)
        tau = np.full(1000, 2.0)
        J = angular_impulse_series(tau, t)
        assert J[-1] == pytest.approx(2.0, rel=1e-3)


class TestLinearImpulse:
    """Tests for cumulative linear impulse (2D)."""

    def test_starts_at_zero(self):
        t = np.linspace(0, 1, 50)
        F = np.tile([1.0, 2.0], (50, 1))
        J = linear_impulse_series(F, t)
        np.testing.assert_allclose(J[0], [0.0, 0.0])

    def test_constant_force(self):
        """Constant F=[3,4] over [0,1] → J=[3,4]."""
        t = np.linspace(0, 1, 1000)
        F = np.tile([3.0, 4.0], (1000, 1))
        J = linear_impulse_series(F, t)
        np.testing.assert_allclose(J[-1], [3.0, 4.0], rtol=1e-3)

    def test_shape(self):
        t = np.linspace(0, 1, 50)
        F = np.tile([1.0, 2.0], (50, 1))
        J = linear_impulse_series(F, t)
        assert J.shape == (50, 2)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


class TestComputeAllDynamics:
    """Tests for the convenience wrapper."""

    def test_returns_all_keys(self):
        N = 100
        t = np.linspace(0, 1, N)
        tau = np.ones(N)
        omega = np.ones(N)
        F = np.ones((N, 2))
        v = np.ones((N, 2))
        result = compute_all_dynamics(t, tau, omega, F, v)
        expected_keys = {
            "angular_power",
            "linear_power",
            "angular_work",
            "linear_work",
            "angular_impulse",
            "linear_impulse",
        }
        assert set(result.keys()) == expected_keys

    def test_all_finite(self):
        N = 50
        t = np.linspace(0, 1, N)
        tau = np.sin(t)
        omega = np.cos(t)
        F = np.column_stack([np.sin(t), np.cos(t)])
        v = np.column_stack([np.cos(t), -np.sin(t)])
        result = compute_all_dynamics(t, tau, omega, F, v)
        for key, val in result.items():
            assert np.all(np.isfinite(val)), f"{key} has non-finite values"
