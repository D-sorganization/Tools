"""Tests for shared physics base utilities (DRY — #C1)."""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics_base import (
    chain_positions,
    clamp_torque_ndof,
    friction_torque_ndof,
    hermite_smoothstep,
    kinetic_energy_from_M,
    potential_energy_chain,
    total_energy_from_parts,
)


class TestKineticEnergyFromM:
    """Generic kinetic energy T = 0.5 * qdot^T M qdot."""

    def test_zero_velocity(self) -> None:
        M = np.eye(3)
        qdot = np.zeros(3)
        assert kinetic_energy_from_M(M, qdot) == 0.0

    def test_identity_mass_matrix(self) -> None:
        M = np.eye(2)
        qdot = np.array([3.0, 4.0])
        expected = 0.5 * (9 + 16)
        assert np.isclose(kinetic_energy_from_M(M, qdot), expected)

    def test_symmetric_mass_matrix(self) -> None:
        M = np.array([[2.0, 0.5], [0.5, 1.0]])
        qdot = np.array([1.0, 2.0])
        expected = 0.5 * qdot @ M @ qdot
        assert np.isclose(kinetic_energy_from_M(M, qdot), expected)

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            n = rng.integers(1, 6)
            A = rng.standard_normal((n, n))
            M = A @ A.T + 0.01 * np.eye(n)  # PSD
            qdot = rng.standard_normal(n)
            assert kinetic_energy_from_M(M, qdot) >= -1e-14

    def test_shape_mismatch_raises(self) -> None:
        M = np.eye(3)
        qdot = np.zeros(2)
        with pytest.raises(AssertionError):
            kinetic_energy_from_M(M, qdot)


class TestTotalEnergy:
    def test_simple(self) -> None:
        assert total_energy_from_parts(5.0, -3.0) == 2.0

    def test_nan_raises(self) -> None:
        with pytest.raises(AssertionError):
            total_energy_from_parts(float("nan"), 1.0)


class TestFrictionTorqueNDOF:
    def test_zero_velocity_viscous_only(self) -> None:
        qdot = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        tau = friction_torque_ndof(qdot, b)
        np.testing.assert_array_equal(tau, np.zeros(3))

    def test_opposes_motion(self) -> None:
        qdot = np.array([1.0, -2.0, 0.5])
        b = np.array([1.0, 1.0, 1.0])
        tau = friction_torque_ndof(qdot, b)
        # tau should oppose qdot sign
        for i in range(3):
            if qdot[i] > 0:
                assert tau[i] < 0
            elif qdot[i] < 0:
                assert tau[i] > 0

    def test_with_coulomb(self) -> None:
        qdot = np.array([2.0])
        b = np.array([0.5])
        mu = np.array([0.1])
        tau = friction_torque_ndof(qdot, b, mu)
        expected = -0.5 * 2.0 - 0.1 * 1.0
        assert np.isclose(tau[0], expected)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(AssertionError):
            friction_torque_ndof(np.zeros(2), np.ones(3))


class TestClampTorqueNDOF:
    def test_within_limits(self) -> None:
        tau = np.array([1.0, -2.0])
        limits = np.array([10.0, 10.0])
        result = clamp_torque_ndof(tau, limits)
        np.testing.assert_array_equal(result, tau)

    def test_clamped(self) -> None:
        tau = np.array([15.0, -20.0])
        limits = np.array([10.0, 10.0])
        result = clamp_torque_ndof(tau, limits)
        np.testing.assert_array_equal(result, [10.0, -10.0])

    def test_inf_limits_no_clamp(self) -> None:
        tau = np.array([1e6, -1e6])
        limits = np.array([np.inf, np.inf])
        result = clamp_torque_ndof(tau, limits)
        np.testing.assert_array_equal(result, tau)


class TestChainPositions:
    def test_single_segment_hanging(self) -> None:
        # Angle 0 = straight down: x=0, y=-L (y-up means cos(0)=1, so y = -L*cos(0)?)
        # Actually the code uses y += -L * cos(angle) for y-DOWN,
        # but physics_base uses y-up: x += -L*sin, y += L*cos
        # Wait, let me re-read. The code says y-up: y += -L*cos(angle)
        # Actually: convention is angle 0 = downward vertical.
        # With y-up: hanging straight down means endpoint is at y = -L
        # cos(0)=1, so y += -L*cos(0) = -L. ✓
        angles = np.array([0.0])
        lengths = np.array([1.0])
        pos = chain_positions(angles, lengths)
        assert np.isclose(pos[0, 0], 0.0)  # x = 0
        assert np.isclose(pos[0, 1], -1.0)  # y = -L

    def test_two_segments_straight_down(self) -> None:
        angles = np.array([0.0, 0.0])
        lengths = np.array([1.0, 0.5])
        pos = chain_positions(angles, lengths)
        assert np.isclose(pos[0, 1], -1.0)
        assert np.isclose(pos[1, 1], -1.5)

    def test_with_origin_offset(self) -> None:
        angles = np.array([0.0])
        lengths = np.array([1.0])
        pos = chain_positions(angles, lengths, origin=(2.0, 3.0))
        assert np.isclose(pos[0, 0], 2.0)
        assert np.isclose(pos[0, 1], 2.0)  # 3.0 - 1.0

    def test_horizontal_segment(self) -> None:
        # angle = pi/2 means segment points to the left
        # x += -L*sin(pi/2) = -L, y += -L*cos(pi/2) = 0
        angles = np.array([np.pi / 2])
        lengths = np.array([1.0])
        pos = chain_positions(angles, lengths)
        assert np.isclose(pos[0, 0], -1.0, atol=1e-10)
        assert np.isclose(pos[0, 1], 0.0, atol=1e-10)


class TestPotentialEnergyChain:
    def test_hanging_straight_down(self) -> None:
        # All angles 0 (hanging down), PE should be negative
        angles = np.array([0.0, 0.0])
        lengths = np.array([1.0, 1.0])
        masses = np.array([1.0, 1.0])
        g = 9.81
        V = potential_energy_chain(angles, lengths, masses, g)
        # V = -(m1+m2)*g*L1*cos(0) - m2*g*L2*cos(0)
        # V = -(2)*9.81*1*1 - 1*9.81*1*1 = -3*9.81
        assert np.isclose(V, -3 * 9.81)

    def test_zero_gravity(self) -> None:
        angles = np.array([0.5, 1.0])
        lengths = np.array([1.0, 1.0])
        masses = np.array([1.0, 1.0])
        assert potential_energy_chain(angles, lengths, masses, g=0.0) == 0.0


class TestHermiteSmoothstep:
    def test_endpoints(self) -> None:
        assert hermite_smoothstep(0.0) == 0.0
        assert hermite_smoothstep(1.0) == 1.0

    def test_midpoint(self) -> None:
        assert np.isclose(hermite_smoothstep(0.5), 0.5)

    def test_monotonic(self) -> None:
        xs = np.linspace(0, 1, 100)
        ys = [hermite_smoothstep(x) for x in xs]
        for i in range(1, len(ys)):
            assert ys[i] >= ys[i - 1]

    def test_clamps_outside_range(self) -> None:
        assert hermite_smoothstep(-0.5) == 0.0
        assert hermite_smoothstep(1.5) == 1.0
