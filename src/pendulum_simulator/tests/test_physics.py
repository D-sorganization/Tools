# ruff: noqa: E501
"""Tests for the physics module.

Organized by property being tested, following TDD principles:
each test was conceptualized before the implementation to verify
a specific physical or mathematical property.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from double_pendulum_golf.physics import (
    PendulumParams,
    coriolis_vector,
    equations_of_motion,
    forward_kinematics,
    gravity_vector,
    kinetic_energy,
    mass_matrix,
    net_joint_forces,
    potential_energy,
)

# ======================================================================
# Mass matrix properties
# ======================================================================


class TestMassMatrixSymmetry:
    """The mass matrix must be symmetric: M12 == M21."""

    def test_symmetric_at_zero(self, default_params: PendulumParams) -> None:
        M = mass_matrix(0.0, default_params)
        assert np.isclose(M[0, 1], M[1, 0])

    def test_symmetric_at_arbitrary_angle(self, default_params: PendulumParams) -> None:
        for phi in np.linspace(-np.pi, np.pi, 20):
            M = mass_matrix(phi, default_params)
            assert np.isclose(M[0, 1], M[1, 0]), f"Not symmetric at phi={phi}"


class TestMassMatrixPositiveDefinite:
    """The mass matrix must be positive definite (all eigenvalues > 0)."""

    def test_positive_definite_at_various_angles(
        self, default_params: PendulumParams
    ) -> None:
        for phi in np.linspace(-np.pi, np.pi, 50):
            M = mass_matrix(phi, default_params)
            eigenvalues = np.linalg.eigvalsh(M)
            assert all(
                ev > 0 for ev in eigenvalues
            ), f"Not positive definite at phi={phi}: eigenvalues={eigenvalues}"


class TestMassMatrixCouplingMaximum:
    """Off-diagonal coupling |M12| should be maximized when segments are aligned (phi=0)."""

    def test_coupling_maximized_at_alignment(
        self, default_params: PendulumParams
    ) -> None:
        M12_at_zero = abs(mass_matrix(0.0, default_params)[0, 1])
        for phi in np.linspace(0.1, np.pi, 30):
            M12 = abs(mass_matrix(phi, default_params)[0, 1])
            assert (
                M12 <= M12_at_zero + 1e-10
            ), f"|M12| at phi={phi:.2f} ({M12:.4f}) exceeds value at phi=0 ({M12_at_zero:.4f})"


class TestMassMatrixDiagonalConstant:
    """M22 (club's self-inertia) should be constant — it doesn't depend on phi."""

    def test_m22_independent_of_phi(self, default_params: PendulumParams) -> None:
        M22_ref = mass_matrix(0.0, default_params)[1, 1]
        for phi in np.linspace(-np.pi, np.pi, 30):
            M22 = mass_matrix(phi, default_params)[1, 1]
            assert np.isclose(
                M22, M22_ref
            ), f"M22 changed at phi={phi}: {M22} vs {M22_ref}"

    def test_m22_equals_expected(self, default_params: PendulumParams) -> None:
        """M22 = m2 * L2^2 for point mass at tip."""
        expected = default_params.m2 * default_params.L2**2
        M22 = mass_matrix(0.0, default_params)[1, 1]
        assert np.isclose(M22, expected)


class TestMassMatrixKnownValues:
    """Verify mass matrix against hand-computed values for simple case."""

    def test_aligned_equal_segments(self, equal_params: PendulumParams) -> None:
        """When m1=m2=1, L1=L2=1, phi=0:
        M11 = 2*1 + 1 + 2*1 = 5
        M12 = 1 + 1 = 2
        M22 = 1
        """
        M = mass_matrix(0.0, equal_params)
        assert np.isclose(M[0, 0], 5.0)
        assert np.isclose(M[0, 1], 2.0)
        assert np.isclose(M[1, 1], 1.0)

    def test_perpendicular_equal_segments(self, equal_params: PendulumParams) -> None:
        """phi=pi/2 -> cos(phi)=0:
        M11 = 2 + 1 + 0 = 3
        M12 = 1 + 0 = 1
        M22 = 1
        """
        M = mass_matrix(np.pi / 2, equal_params)
        assert np.isclose(M[0, 0], 3.0)
        assert np.isclose(M[0, 1], 1.0)
        assert np.isclose(M[1, 1], 1.0)


# ======================================================================
# Coriolis / centrifugal
# ======================================================================


class TestCoriolisVector:
    """Tests for the Coriolis/centrifugal force computation."""

    def test_zero_velocity_gives_zero_coriolis(
        self, default_params: PendulumParams
    ) -> None:
        """No velocity => no velocity-dependent forces."""
        C = coriolis_vector(0.5, 0.0, 0.0, default_params)
        assert np.allclose(C, [0.0, 0.0])

    def test_zero_at_alignment_with_only_theta1_velocity(
        self, default_params: PendulumParams
    ) -> None:
        """When phi=0 (sin(phi)=0), Coriolis vanishes regardless of velocities."""
        C = coriolis_vector(0.0, 5.0, 3.0, default_params)
        assert np.allclose(C, [0.0, 0.0], atol=1e-14)

    def test_antisymmetric_in_phi(self, default_params: PendulumParams) -> None:
        """C(-phi, ...) has opposite sign since sin(-phi) = -sin(phi)."""
        phi = 0.7
        C_pos = coriolis_vector(phi, 2.0, 1.0, default_params)
        C_neg = coriolis_vector(-phi, 2.0, 1.0, default_params)
        assert np.allclose(C_pos, -C_neg, atol=1e-14)


# ======================================================================
# Gravity vector
# ======================================================================


class TestGravityVector:
    """Tests for gravitational torques."""

    def test_zero_at_equilibrium(self, default_params: PendulumParams) -> None:
        """When hanging straight down (theta1=0, phi=0), gravity torque is zero."""
        G = gravity_vector(0.0, 0.0, default_params)
        assert np.allclose(G, [0.0, 0.0], atol=1e-14)

    def test_nonzero_when_displaced(self, default_params: PendulumParams) -> None:
        """Any displacement from vertical should produce restoring torques."""
        G = gravity_vector(0.5, 0.0, default_params)
        assert not np.allclose(G, [0.0, 0.0])


# ======================================================================
# Equations of motion
# ======================================================================


class TestEquationsOfMotion:
    """Tests for the complete EOM."""

    def test_equilibrium_at_bottom(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
    ) -> None:
        """At vertical with zero velocity and zero torque, acceleration is zero."""
        state = np.array([0.0, 0.0, 0.0, 0.0])
        sdot = equations_of_motion(state, 0.0, default_params, zero_torque)
        # Velocities are zero, accelerations should be zero at equilibrium
        assert np.allclose(sdot, [0.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_state_shape(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        sdot = equations_of_motion(aligned_state, 0.0, default_params, zero_torque)
        assert sdot.shape == (4,)

    def test_velocity_passthrough(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        """First two components of state_dot should equal the velocities."""
        sdot = equations_of_motion(aligned_state, 0.0, default_params, zero_torque)
        assert np.isclose(sdot[0], aligned_state[2])
        assert np.isclose(sdot[1], aligned_state[3])


# ======================================================================
# Forward kinematics
# ======================================================================


class TestForwardKinematics:
    """Tests for Cartesian position computation."""

    def test_straight_down(self, default_params: PendulumParams) -> None:
        """theta1=0, phi=0 -> wrist at (0, -L1), tip at (0, -(L1+L2))."""
        pos = forward_kinematics(0.0, 0.0, default_params)
        assert np.isclose(pos["shoulder"][0], 0.0)
        assert np.isclose(pos["shoulder"][1], 0.0)
        assert np.isclose(pos["wrist"][0], 0.0)
        assert np.isclose(pos["wrist"][1], -default_params.L1)
        assert np.isclose(pos["tip"][0], 0.0)
        assert np.isclose(pos["tip"][1], -(default_params.L1 + default_params.L2))

    def test_arm_horizontal_club_straight(self, default_params: PendulumParams) -> None:
        """theta1=pi/2, phi=0 -> arm points right, club continues right."""
        pos = forward_kinematics(np.pi / 2, 0.0, default_params)
        L1, L2 = default_params.L1, default_params.L2
        assert np.isclose(pos["wrist"][0], L1, atol=1e-10)
        assert np.isclose(pos["wrist"][1], 0.0, atol=1e-10)
        assert np.isclose(pos["tip"][0], L1 + L2, atol=1e-10)
        assert np.isclose(pos["tip"][1], 0.0, atol=1e-10)


# ======================================================================
# Energy
# ======================================================================


class TestEnergy:
    """Tests for energy computations."""

    def test_zero_kinetic_at_rest(self, default_params: PendulumParams) -> None:
        state = np.array([0.5, 0.3, 0.0, 0.0])
        assert np.isclose(kinetic_energy(state, default_params), 0.0)

    def test_positive_kinetic_when_moving(
        self, default_params: PendulumParams, aligned_state: np.ndarray
    ) -> None:
        T = kinetic_energy(aligned_state, default_params)
        assert T > 0

    def test_potential_minimum_at_bottom(self, default_params: PendulumParams) -> None:
        """Potential energy is minimized when both segments hang down."""
        V_bottom = potential_energy(np.array([0, 0, 0, 0]), default_params)
        V_displaced = potential_energy(np.array([0.5, 0.3, 0, 0]), default_params)
        assert V_bottom < V_displaced


# ======================================================================
# Net joint forces
# ======================================================================


class TestNetJointForces:
    """Net joint forces should balance gravity at rest."""

    def test_forces_at_rest(self, default_params: PendulumParams) -> None:
        state = np.array([0.0, 0.0, 0.0, 0.0])
        qddot = np.array([0.0, 0.0])
        forces = net_joint_forces(state, qddot, default_params)

        m1, m2, g = default_params.m1, default_params.m2, default_params.g
        assert np.isclose(forces["wrist"][0], 0.0)
        assert np.isclose(forces["wrist"][1], m2 * g)
        assert np.isclose(forces["shoulder"][0], 0.0)
        assert np.isclose(forces["shoulder"][1], (m1 + m2) * g)


# ======================================================================
# Design by Contract: precondition violations
# ======================================================================


class TestDbCViolations:
    """Verify that precondition violations raise AssertionError."""

    def test_negative_mass_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            PendulumParams(m1=-1, m2=1, L1=1, L2=1)

    def test_zero_length_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            PendulumParams(m1=1, m2=1, L1=0, L2=1)

    def test_nan_phi_rejected(self, default_params: PendulumParams) -> None:
        with pytest.raises((ValueError, TypeError)):
            mass_matrix(float("nan"), default_params)

    def test_wrong_state_shape_rejected(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
    ) -> None:
        with pytest.raises((ValueError, TypeError)):
            equations_of_motion(np.array([0, 0, 0]), 0.0, default_params, zero_torque)


# ======================================================================
# Joint limit Hermite smoothstep — edge cases (issue #1289, #1290)
# ======================================================================


class TestHermitePenaltyHelper:
    """Test the extracted _hermite_penalty helper for edge-case correctness."""

    def test_imports(self):
        from double_pendulum_golf.physics import _hermite_penalty  # noqa: F401

    def test_zero_penetration_gives_zero(self):
        from double_pendulum_golf.physics import _hermite_penalty

        # pen=0 → blend=0 → smooth=0 → penalty=0
        assert _hermite_penalty(
            0.0, vel=0.0, transition=0.05, stiffness=500.0, damping=20.0
        ) == pytest.approx(0.0)

    def test_full_penetration_no_blend(self):
        from double_pendulum_golf.physics import _hermite_penalty

        # pen >= transition → blend=1 → smooth=1 → full penalty
        pen = 0.05  # exactly at transition
        result = _hermite_penalty(
            pen, vel=0.0, transition=0.05, stiffness=500.0, damping=0.0
        )
        assert result == pytest.approx(500.0 * 0.05, rel=1e-9)

    def test_large_penetration_clamps_blend(self):
        from double_pendulum_golf.physics import _hermite_penalty

        # pen >> transition → blend clamped at 1 → same as full penalty
        r1 = _hermite_penalty(
            0.05, vel=0.0, transition=0.05, stiffness=500.0, damping=0.0
        )
        r2 = _hermite_penalty(
            1.0, vel=0.0, transition=0.05, stiffness=500.0, damping=0.0
        )
        # Both have blend=1; r2 has larger pen so larger result
        assert r2 > r1

    def test_damping_only_when_velocity_into_limit(self):
        from double_pendulum_golf.physics import _hermite_penalty

        # vel > 0 means moving into the limit → damping adds
        pen = 0.05
        r_into = _hermite_penalty(
            pen, vel=1.0, transition=0.05, stiffness=0.0, damping=20.0
        )
        # vel = 0 means no damping contribution
        r_zero = _hermite_penalty(
            pen, vel=0.0, transition=0.05, stiffness=0.0, damping=20.0
        )
        assert r_into > r_zero


class TestJointLimitTorqueEdgeCases:
    """Joint limit torque: at-limit, within, and beyond limit cases."""

    @pytest.fixture
    def limits(self):
        from double_pendulum_golf.physics import JointLimits

        return JointLimits(
            theta1_min=-1.0,
            theta1_max=1.0,
            phi_min=-1.0,
            phi_max=1.0,
            stiffness=500.0,
            damping=20.0,
        )

    def test_within_limits_gives_zero(self, limits):
        from double_pendulum_golf.physics import joint_limit_torque

        tau = joint_limit_torque(
            phi=0.0, dphi=0.0, limits=limits, theta1=0.0, dtheta1=0.0
        )
        np.testing.assert_allclose(tau, [0.0, 0.0], atol=1e-12)

    def test_exactly_at_lower_phi_limit_gives_zero(self, limits):
        from double_pendulum_golf.physics import joint_limit_torque

        # phi == phi_min → penetration=0 → penalty=0
        tau = joint_limit_torque(phi=-1.0, dphi=0.0, limits=limits)
        assert tau[1] == pytest.approx(0.0)

    def test_below_lower_phi_limit_gives_positive_torque(self, limits):
        from double_pendulum_golf.physics import joint_limit_torque

        # phi < phi_min → positive restoring torque
        tau = joint_limit_torque(phi=-1.1, dphi=0.0, limits=limits)
        assert tau[1] > 0.0

    def test_above_upper_phi_limit_gives_negative_torque(self, limits):
        from double_pendulum_golf.physics import joint_limit_torque

        # phi > phi_max → negative restoring torque
        tau = joint_limit_torque(phi=1.1, dphi=0.0, limits=limits)
        assert tau[1] < 0.0


class TestForwardKinematicsPostconditions:
    """Postcondition: segment lengths must match params.L1 and params.L2."""

    def test_segment_lengths_arbitrary_angle(self, default_params: PendulumParams):
        for theta1 in np.linspace(-np.pi, np.pi, 12):
            for phi in np.linspace(-np.pi / 2, np.pi / 2, 6):
                pos = forward_kinematics(theta1, phi, default_params)
                sx, sy = pos["shoulder"]
                wx, wy = pos["wrist"]
                tx, ty = pos["tip"]
                wrist_dist = np.hypot(wx - sx, wy - sy)
                tip_dist = np.hypot(tx - wx, ty - wy)
                assert (
                    abs(wrist_dist - default_params.L1) < 1e-9
                ), f"theta1={theta1:.2f}, phi={phi:.2f}: wrist_dist={wrist_dist:.9f}"
                assert (
                    abs(tip_dist - default_params.L2) < 1e-9
                ), f"theta1={theta1:.2f}, phi={phi:.2f}: tip_dist={tip_dist:.9f}"
