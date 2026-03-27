"""Tests for the triple pendulum physics module using TDD.

Organized by property being tested, following TDD principles.
Each test verifies a specific physical or mathematical property.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from double_pendulum_golf.physics_triple import (
    TriplePendulumParams,
    coriolis_vector as coriolis_vector_triple,
    equations_of_motion as equations_of_motion_triple,
    forward_kinematics as forward_kinematics_triple,
    gravity_vector as gravity_vector_triple,
    mass_matrix as mass_matrix_triple,
    net_joint_forces as net_joint_forces_triple,
)


@pytest.fixture
def triple_params() -> TriplePendulumParams:
    """Default parameters for triple pendulum."""
    return TriplePendulumParams(
        m1=5.0,
        m2=0.5,
        m3=0.2,
        L1=0.6,
        L2=0.6,
        L3=0.6,
    )


@pytest.fixture
def triple_torque_func() -> Callable[[float], tuple[float, float, float]]:
    """Simple constant torque function."""

    def torque(t: float) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    return torque


class TestTripleMassMatrixSymmetry:
    """The 3x3 mass matrix must be symmetric."""

    def test_symmetric_at_zero(self, triple_params: TriplePendulumParams) -> None:
        phi1, phi2 = 0.0, 0.0
        M = mass_matrix_triple(phi1, phi2, triple_params)
        assert M.shape == (3, 3)
        # Check symmetry: M[i,j] == M[j,i]
        for i in range(3):
            for j in range(3):
                assert np.isclose(M[i, j], M[j, i]), f"M[{i},{j}] != M[{j},{i}]"

    def test_symmetric_at_arbitrary_angles(self, triple_params: TriplePendulumParams) -> None:
        for phi1 in np.linspace(-np.pi, np.pi, 10):
            for phi2 in np.linspace(-np.pi, np.pi, 10):
                M = mass_matrix_triple(phi1, phi2, triple_params)
                for i in range(3):
                    for j in range(3):
                        assert np.isclose(M[i, j], M[j, i]), (
                            f"Not symmetric at phi1={phi1}, phi2={phi2}"
                        )


class TestTripleMassMatrixPositiveDefinite:
    """The 3x3 mass matrix must be positive definite."""

    def test_positive_definite_at_various_angles(
        self, triple_params: TriplePendulumParams
    ) -> None:
        test_angles = list(np.linspace(-np.pi, np.pi, 15))
        for phi1 in test_angles:
            for phi2 in test_angles:
                M = mass_matrix_triple(phi1, phi2, triple_params)
                eigenvalues = np.linalg.eigvalsh(M)
                assert all(ev > 0 for ev in eigenvalues), (
                    f"Not positive definite at phi1={phi1}, phi2={phi2}"
                )


class TestTripleCoriolisZeroAtRest:
    """Coriolis/centrifugal vector should be zero when velocities are zero."""

    def test_zero_at_rest(self, triple_params: TriplePendulumParams) -> None:
        phi1, phi2 = 0.5, -0.3
        dtheta1, dphi1, dphi2 = 0.0, 0.0, 0.0
        C = coriolis_vector_triple(phi1, phi2, dtheta1, dphi1, dphi2, triple_params)
        assert C.shape == (3,)
        assert np.allclose(C, 0.0, atol=1e-12)


class TestTripleGravityVectorDirection:
    """Gravity should pull the system downward."""

    def test_gravity_points_downward_at_rest(
        self, triple_params: TriplePendulumParams
    ) -> None:
        # At theta1 = 0 (straight down), gravity should be zero
        theta1, phi1, phi2 = 0.0, 0.0, 0.0
        G = gravity_vector_triple(theta1, phi1, phi2, triple_params)
        assert G.shape == (3,)
        # At equilibrium (all pointing down), gravity torques should be zero
        assert np.allclose(G, 0.0, atol=1e-12)

    def test_gravity_restores_from_lifted_position(
        self, triple_params: TriplePendulumParams
    ) -> None:
        # When lifted (theta1 = pi/2, pointing to the side),
        # gravity should create restoring torque
        theta1, phi1, phi2 = np.pi / 2, 0.0, 0.0
        G = gravity_vector_triple(theta1, phi1, phi2, triple_params)
        assert not np.allclose(G, 0.0)


class TestTripleForwardKinematics:
    """Forward kinematics should give correct positions."""

    def test_all_hanging_down(self, triple_params: TriplePendulumParams) -> None:
        theta1, phi1, phi2 = 0.0, 0.0, 0.0
        pos = forward_kinematics_triple(theta1, phi1, phi2, triple_params)

        # Check keys
        assert set(pos.keys()) == {"hub", "shoulder", "wrist1", "wrist2", "tip"}

        # Hub at origin
        assert np.allclose(pos["hub"], (0.0, 0.0))

        # When straight down, positions should be at negative y
        assert pos["wrist1"][0] == 0.0  # x = 0
        assert pos["wrist1"][1] < 0  # y < 0

        assert pos["tip"][0] == 0.0
        assert pos["tip"][1] < 0

    def test_all_horizontal(self, triple_params: TriplePendulumParams) -> None:
        theta1 = np.pi / 2
        phi1, phi2 = 0.0, 0.0
        pos = forward_kinematics_triple(theta1, phi1, phi2, triple_params)

        # With default scapula offset (0), wrist1 should be at (L1, 0)
        L1 = triple_params.L1
        assert np.isclose(pos["wrist1"][0], L1, atol=1e-6)
        assert np.isclose(pos["wrist1"][1], 0.0, atol=1e-6)


class TestTripleEquationsOfMotion:
    """Equations of motion should produce valid accelerations."""

    def test_eom_produces_valid_state_derivative(
        self,
        triple_params: TriplePendulumParams,
        triple_torque_func: Callable[[float], tuple[float, float, float]],
    ) -> None:
        # State: [theta1, phi1, phi2, dtheta1, dphi1, dphi2]
        state = np.array([0.1, 0.05, -0.05, 0.0, 0.0, 0.0])
        state_dot = equations_of_motion_triple(state, 0.0, triple_params, triple_torque_func)

        assert state_dot.shape == (6,)
        assert all(np.isfinite(state_dot)), f"Invalid values: {state_dot}"

    def test_eom_at_rest_at_equilibrium(
        self,
        triple_params: TriplePendulumParams,
        triple_torque_func: Callable[[float], tuple[float, float, float]],
    ) -> None:
        # At equilibrium with zero velocity, acceleration should be zero
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_dot = equations_of_motion_triple(state, 0.0, triple_params, triple_torque_func)

        # Velocities should match input (first 3 elements should be all zeros)
        assert np.isclose(state_dot[0], 0.0)  # dtheta1
        assert np.isclose(state_dot[1], 0.0)  # dphi1
        assert np.isclose(state_dot[2], 0.0)  # dphi2
        # Accelerations should be small (near zero at equilibrium)
        assert np.allclose(state_dot[3:], 0.0, atol=1e-10)


class TestTripleParameterValidation:
    """Parameters must satisfy constraints."""

    def test_negative_mass_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            TriplePendulumParams(m1=-1.0, m2=0.5, m3=0.2, L1=0.6, L2=0.6, L3=0.6)

    def test_negative_length_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            TriplePendulumParams(m1=1.0, m2=0.5, m3=0.2, L1=-0.6, L2=0.6, L3=0.6)

    def test_zero_mass_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            TriplePendulumParams(m1=1.0, m2=0.0, m3=0.2, L1=0.6, L2=0.6, L3=0.6)


class TestTripleMassMatrixDependenceOnAngles:
    """Mass matrix should depend on the coupling angles."""

    def test_mass_matrix_different_at_different_angles(
        self, triple_params: TriplePendulumParams
    ) -> None:
        M1 = mass_matrix_triple(0.0, 0.0, triple_params)
        M2 = mass_matrix_triple(np.pi / 4, 0.0, triple_params)

        # They should be different
        assert not np.allclose(M1, M2)


class TestTripleCoriolisNonlinearCoupling:
    """Coriolis vector should show nonlinear dependence on velocities."""

    def test_coriolis_scales_with_velocity_squared(
        self, triple_params: TriplePendulumParams
    ) -> None:
        phi1, phi2 = 0.5, -0.3

        dtheta1_small = 0.1
        C_small = coriolis_vector_triple(phi1, phi2, dtheta1_small, 0.1, 0.1, triple_params)

        dtheta1_large = 0.2  # 2x larger
        C_large = coriolis_vector_triple(phi1, phi2, dtheta1_large, 0.1, 0.1, triple_params)

        # The change should not be linear (quadratic in velocity)
        ratio = np.linalg.norm(C_large) / np.linalg.norm(C_small)
        # ratio should be approximately 4x (since velocity is 2x)
        assert ratio > 2 and ratio < 8, f"Unexpected scaling: {ratio}"


class TestTripleNetJointForces:
    """Net joint forces should balance gravity at rest."""

    def test_forces_at_rest(self, triple_params: TriplePendulumParams) -> None:
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        qddot = np.array([0.0, 0.0, 0.0])
        forces = net_joint_forces_triple(state, qddot, triple_params)

        m1, m2, m3, g = (
            triple_params.m1,
            triple_params.m2,
            triple_params.m3,
            triple_params.g,
        )
        assert np.isclose(forces["wrist2"][0], 0.0)
        assert np.isclose(forces["wrist2"][1], m3 * g)
        assert np.isclose(forces["wrist1"][0], 0.0)
        assert np.isclose(forces["wrist1"][1], (m2 + m3) * g)
        assert np.isclose(forces["shoulder"][0], 0.0)
        assert np.isclose(forces["shoulder"][1], (m1 + m2 + m3) * g)
