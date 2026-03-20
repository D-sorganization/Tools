"""Unit tests for _mr_dynamics submodule (GH1649: refactored from modern_robotics.py).

Tests cover:
- EulerStep integration
- InverseDynamics (Newton-Euler, 3-link robot from textbook)
- MassMatrix
- GravityForces
- VelQuadraticForces
- ForwardDynamics
"""

from __future__ import annotations

import numpy as np

from rotation_converter.modern_robotics import (
    EulerStep,
    ForwardDynamics,
    GravityForces,
    InverseDynamics,
    MassMatrix,
    VelQuadraticForces,
    ad,
)

# ---------------------------------------------------------------------------
# 3-Link UR5-like robot fixture (Lynch & Park textbook example)
# ---------------------------------------------------------------------------

M01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])
M12 = np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]])
M23 = np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]])
M34 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.14225], [0, 0, 0, 1]])
Mlist = np.array([M01, M12, M23, M34])

G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
Glist = np.array([G1, G2, G3])

Slist = np.array(
    [
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, -0.089, 0, 0],
        [0, 1, 0, -0.089, 0, 0.425],
    ]
).T

ATOL = 1e-4


class TestEulerStep:
    """EulerStep: first-order Euler integration."""

    def test_zero_acceleration(self) -> None:
        """Zero acceleration — angles and velocities unchanged after step."""
        theta = np.array([0.1, 0.2, 0.3])
        dtheta = np.array([0.5, 0.5, 0.5])
        ddtheta = np.zeros(3)
        theta_next, dtheta_next = EulerStep(theta, dtheta, ddtheta, 0.1)
        np.testing.assert_allclose(theta_next, theta + 0.1 * dtheta, atol=ATOL)
        np.testing.assert_allclose(dtheta_next, dtheta, atol=ATOL)

    def test_basic_step(self) -> None:
        """Basic Euler step matches textbook example."""
        theta = np.array([0.1, 0.1, 0.1])
        dtheta = np.array([0.1, 0.2, 0.3])
        ddtheta = np.array([2.0, 1.5, 1.0])
        dt = 0.1
        theta_next, dtheta_next = EulerStep(theta, dtheta, ddtheta, dt)
        np.testing.assert_allclose(theta_next, np.array([0.11, 0.12, 0.13]), atol=ATOL)
        np.testing.assert_allclose(dtheta_next, np.array([0.3, 0.35, 0.4]), atol=ATOL)


class TestAdBracket:
    """ad(V): Lie bracket / adjoint action."""

    def test_ad_shape(self) -> None:
        V = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = ad(V)
        assert result.shape == (6, 6)

    def test_ad_antisymmetric_upper(self) -> None:
        """The 3x3 upper-left block must be skew-symmetric."""
        V = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        result = ad(V)
        block = result[:3, :3]
        np.testing.assert_allclose(block + block.T, np.zeros((3, 3)), atol=1e-12)


class TestInverseDynamics:
    """InverseDynamics: Newton-Euler algorithm."""

    def test_textbook_example(self) -> None:
        """Reproduce textbook (Lynch & Park) 3-link example."""
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        ddthetalist = np.array([2.0, 1.5, 1.0])
        g = np.array([0.0, 0.0, -9.8])
        Ftip = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        tau = InverseDynamics(
            thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist
        )
        expected = np.array([74.69616155, -33.06766016, -3.23057314])
        np.testing.assert_allclose(tau, expected, atol=1e-4)

    def test_output_shape(self) -> None:
        thetalist = np.array([0.0, 0.0, 0.0])
        tau = InverseDynamics(
            thetalist,
            np.zeros(3),
            np.zeros(3),
            np.array([0.0, 0.0, -9.8]),
            np.zeros(6),
            Mlist,
            Glist,
            Slist,
        )
        assert tau.shape == (3,)


class TestMassMatrix:
    """MassMatrix: inertia matrix computation."""

    def test_shape(self) -> None:
        thetalist = np.array([0.1, 0.1, 0.1])
        M = MassMatrix(thetalist, Mlist, Glist, Slist)
        assert M.shape == (3, 3)

    def test_symmetric(self) -> None:
        """Mass matrix must be symmetric."""
        thetalist = np.array([0.1, 0.1, 0.1])
        M = MassMatrix(thetalist, Mlist, Glist, Slist)
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_positive_definite(self) -> None:
        """Mass matrix must be positive definite (all eigenvalues > 0)."""
        thetalist = np.array([0.1, 0.1, 0.1])
        M = MassMatrix(thetalist, Mlist, Glist, Slist)
        eigenvalues = np.linalg.eigvals(M)
        assert np.all(eigenvalues > 0)


class TestGravityForces:
    """GravityForces: gravity torques."""

    def test_zero_gravity(self) -> None:
        """Zero gravity should return near-zero torques."""
        thetalist = np.array([0.1, 0.2, 0.3])
        g = np.array([0.0, 0.0, 0.0])
        tau = GravityForces(thetalist, g, Mlist, Glist, Slist)
        np.testing.assert_allclose(tau, np.zeros(3), atol=ATOL)

    def test_nonzero_gravity(self) -> None:
        """Non-zero gravity produces non-trivial torques."""
        thetalist = np.array([0.1, 0.1, 0.1])
        g = np.array([0.0, 0.0, -9.8])
        tau = GravityForces(thetalist, g, Mlist, Glist, Slist)
        assert np.any(np.abs(tau) > 0.01), "Gravity torques should be non-zero"


class TestVelQuadraticForces:
    """VelQuadraticForces: Coriolis and centripetal terms."""

    def test_zero_velocity(self) -> None:
        """Zero joint velocity should yield zero quadratic forces."""
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.zeros(3)
        tau = VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
        np.testing.assert_allclose(tau, np.zeros(3), atol=ATOL)


class TestForwardDynamics:
    """ForwardDynamics: compute joint accelerations from torques."""

    def test_output_shape(self) -> None:
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        taulist = np.array([0.5, 0.6, 0.7])
        g = np.array([0.0, 0.0, -9.8])
        Ftip = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ddtheta = ForwardDynamics(
            thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist
        )
        assert ddtheta.shape == (3,)

    def test_consistency_with_inverse(self) -> None:
        """ForwardDynamics(tau) should recover the ddtheta used to compute tau."""
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.0, 0.0, 0.0])
        ddthetalist_target = np.array([1.0, 0.5, -0.5])
        g = np.array([0.0, 0.0, 0.0])
        Ftip = np.zeros(6)
        # Compute required torques
        tau = InverseDynamics(
            thetalist, dthetalist, ddthetalist_target, g, Ftip, Mlist, Glist, Slist
        )
        # Recover accelerations
        ddtheta_recovered = ForwardDynamics(
            thetalist, dthetalist, tau, g, Ftip, Mlist, Glist, Slist
        )
        np.testing.assert_allclose(ddtheta_recovered, ddthetalist_target, atol=1e-6)
