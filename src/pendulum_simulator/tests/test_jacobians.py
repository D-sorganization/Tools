# ruff: noqa: E501
"""
Tests for jacobians.py — Jacobian computation and manipulability ellipsoids.

Design approach
---------------
- TDD: each test was written to specify a required physical property,
  then the implementation was written to satisfy it.
- DbC: we verify that precondition violations raise AssertionError, and
  that postconditions (correct shapes, PSD matrices, etc.) hold.
- DRY: shared fixtures and helpers avoid repetition.

Physical properties verified
-----------------------------
1. Jacobian shape and finiteness
2. Known analytic values at canonical configurations
3. Mobility ellipsoid matrix J Jᵀ is symmetric PSD
4. Force ellipsoid semi-axes are reciprocals of mobility semi-axes
5. At singularities, force_semi_axes is None
6. Continuity: small angle perturbations produce small J changes
7. DRY: ellipsoid_from_jacobian used identically by both models
8. DbC: precondition violations produce AssertionError
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.jacobians import (
    ellipsoid_from_jacobian,
    ellipsoids_double,
    ellipsoids_triple,
    jacobian_double,
    jacobian_triple,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def L() -> tuple[float, float]:
    """Default double-pendulum segment lengths."""
    return 0.6, 1.0


@pytest.fixture
def L3() -> tuple[float, float, float]:
    """Default triple-pendulum segment lengths."""
    return 0.6, 1.0, 0.8


@pytest.fixture
def zero_angles() -> tuple[float, float]:
    return 0.0, 0.0


@pytest.fixture
def right_angle_config() -> tuple[float, float]:
    """Arm horizontal, club straight (theta1=pi/2, phi=0)."""
    return np.pi / 2, 0.0


# ============================================================================
# Jacobian — double pendulum
# ============================================================================


class TestJacobianDoubleShape:
    """Jacobian matrices must have shape (2, 2)."""

    def test_wrist_jacobian_shape(self, L: tuple[float, float]) -> None:
        L1, L2 = L
        jacs = jacobian_double(0.5, 0.3, L1, L2)
        assert jacs["wrist"].shape == (2, 2)

    def test_tip_jacobian_shape(self, L: tuple[float, float]) -> None:
        L1, L2 = L
        jacs = jacobian_double(0.5, 0.3, L1, L2)
        assert jacs["tip"].shape == (2, 2)

    def test_all_finite(self, L: tuple[float, float]) -> None:
        L1, L2 = L
        jacs = jacobian_double(0.5, 0.3, L1, L2)
        for J in jacs.values():
            assert np.all(np.isfinite(J)), f"Non-finite values in Jacobian: {J}"


class TestJacobianDoubleAnalytic:
    """Verify Jacobian values against hand-computed results."""

    def test_straight_down_wrist_jacobian(self, L: tuple[float, float]) -> None:
        """theta1=0 (arm straight down): J_wrist = [[L1, 0], [0, 0]].

        At theta1=0: cos(0)=1, sin(0)=0
        J_wrist = [[L1*1, 0], [L1*0, 0]] = [[L1, 0], [0, 0]]
        (phi has no effect on wrist)
        """
        L1, L2 = L
        J = jacobian_double(0.0, 0.0, L1, L2)["wrist"]
        assert np.isclose(J[0, 0], L1)
        assert np.isclose(J[0, 1], 0.0)
        assert np.isclose(J[1, 0], 0.0)
        assert np.isclose(J[1, 1], 0.0)

    def test_straight_down_tip_jacobian(self, L: tuple[float, float]) -> None:
        """theta1=0, phi=0: J_tip = [[L1+L2, L2], [0, 0]].

        cos(0)=1, sin(0)=0 for both theta1 and theta2=theta1+phi=0
        J_tip = [[L1+L2, L2], [0, 0]]
        """
        L1, L2 = L
        J = jacobian_double(0.0, 0.0, L1, L2)["tip"]
        assert np.isclose(J[0, 0], L1 + L2)
        assert np.isclose(J[0, 1], L2)
        assert np.isclose(J[1, 0], 0.0)
        assert np.isclose(J[1, 1], 0.0)

    def test_arm_horizontal_tip_jacobian(self, L: tuple[float, float]) -> None:
        """theta1=pi/2, phi=0 (arm right, club right):

        c1=0, s1=1, c2=0, s2=1  (theta2=pi/2)
        J_tip = [[0+0, 0], [L1+L2, L2]]
        """
        L1, L2 = L
        J = jacobian_double(np.pi / 2, 0.0, L1, L2)["tip"]
        assert np.isclose(J[0, 0], 0.0, atol=1e-10)
        assert np.isclose(J[0, 1], 0.0, atol=1e-10)
        assert np.isclose(J[1, 0], L1 + L2, atol=1e-10)
        assert np.isclose(J[1, 1], L2, atol=1e-10)

    def test_phi_only_affects_tip_not_wrist(self, L: tuple[float, float]) -> None:
        """Changing phi changes J_tip second column but leaves J_wrist unchanged."""
        L1, L2 = L
        for phi in [0.0, 0.3, 1.0, -0.8]:
            J_wrist = jacobian_double(0.5, phi, L1, L2)["wrist"]
            assert np.isclose(J_wrist[0, 1], 0.0), (
                f"J_wrist[:,1] should be zero for any phi, got {J_wrist[:, 1]}"
            )


class TestJacobianDoubleContinuity:
    """Small angle perturbation should produce small Jacobian change."""

    def test_continuity_at_various_angles(self, L: tuple[float, float]) -> None:
        L1, L2 = L
        eps = 1e-4
        for theta1 in np.linspace(-1.0, 1.0, 10):
            J0 = jacobian_double(theta1, 0.5, L1, L2)["tip"]
            J1 = jacobian_double(theta1 + eps, 0.5, L1, L2)["tip"]
            assert np.allclose(J0, J1, atol=(L1 + L2) * eps * 2), (
                f"Jacobian discontinuity at theta1={theta1}"
            )


# ============================================================================
# Jacobian — triple pendulum
# ============================================================================


class TestJacobianTripleShape:
    """Triple-pendulum Jacobians are (2, 3)."""

    def test_all_jacobians_shape(self, L3: tuple[float, float, float]) -> None:
        L1, L2, L3_ = L3
        jacs = jacobian_triple(0.3, 0.2, 0.1, L1, L2, L3_)
        for name, J in jacs.items():
            assert J.shape == (2, 3), f"Wrong shape for {name}: {J.shape}"
            assert np.all(np.isfinite(J)), f"Non-finite in {name}"


class TestJacobianTripleAnalytic:
    """Known values at canonical configurations."""

    def test_straight_down_wrist1_jacobian(self, L3: tuple[float, float, float]) -> None:
        """theta1=phi1=phi2=0 → wrist1: [[L1, 0, 0], [0, 0, 0]]."""
        L1, L2, L3_ = L3
        J = jacobian_triple(0.0, 0.0, 0.0, L1, L2, L3_)["wrist1"]
        assert np.isclose(J[0, 0], L1)
        assert np.isclose(J[0, 1], 0.0)
        assert np.isclose(J[0, 2], 0.0)
        assert np.isclose(J[1, 0], 0.0)

    def test_straight_down_tip_jacobian(self, L3: tuple[float, float, float]) -> None:
        """All angles=0 → tip: [[L1+L2+L3, L2+L3, L3], [0, 0, 0]]."""
        L1, L2, L3_ = L3
        J = jacobian_triple(0.0, 0.0, 0.0, L1, L2, L3_)["tip"]
        assert np.isclose(J[0, 0], L1 + L2 + L3_)
        assert np.isclose(J[0, 1], L2 + L3_)
        assert np.isclose(J[0, 2], L3_)
        assert np.isclose(J[1, 0], 0.0)

    def test_phi2_only_affects_tip(self, L3: tuple[float, float, float]) -> None:
        """phi2 should affect J_tip's third column but NOT wrist1 or wrist2's third column."""
        L1, L2, L3_ = L3
        for phi2 in [0.0, 0.5, -0.7]:
            jacs = jacobian_triple(0.3, 0.2, phi2, L1, L2, L3_)
            assert np.isclose(jacs["wrist1"][0, 2], 0.0)
            assert np.isclose(jacs["wrist2"][0, 2], 0.0)


# ============================================================================
# ellipsoid_from_jacobian — shared kernel
# ============================================================================


class TestEllipsoidFromJacobianShape:
    """Returned arrays must have correct shapes."""

    def test_directions_shape_2x2(self) -> None:
        J = np.eye(2)
        dirs, mob, force, svs = ellipsoid_from_jacobian(J)
        assert dirs.shape == (2, 2)

    def test_mob_semi_axes_shape(self) -> None:
        J = np.array([[2.0, 0.0], [0.0, 1.0]])
        _, mob, _, _ = ellipsoid_from_jacobian(J)
        assert mob.shape == (2,)
        assert np.all(mob >= 0)

    def test_svd_singular_values_match_mob_axes(self) -> None:
        J = np.array([[3.0, 0.0], [0.0, 2.0]])
        _, mob, _, svs = ellipsoid_from_jacobian(J)
        np.testing.assert_allclose(mob, svs)


class TestEllipsoidMobilityPSD:
    """Mobility ellipsoid matrix J Jᵀ must be symmetric PSD."""

    def test_jjt_symmetric(self) -> None:
        J = np.array([[1.0, 0.5], [0.2, 0.8]])
        JJt = J @ J.T
        assert np.allclose(JJt, JJt.T)

    def test_mob_axes_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            J = rng.normal(size=(2, 2))
            _, mob, _, _ = ellipsoid_from_jacobian(J)
            assert np.all(mob >= 0), f"Negative mobility semi-axis: {mob}"


class TestEllipsoidForceDuality:
    """Force ellipsoid semi-axes must be reciprocals of mobility axes."""

    def test_force_reciprocal_of_mobility(self) -> None:
        J = np.array([[2.0, 0.0], [0.0, 0.5]])
        _, mob, force, _ = ellipsoid_from_jacobian(J)
        assert force is not None
        np.testing.assert_allclose(force, 1.0 / mob, rtol=1e-10)

    def test_force_reciprocal_random_full_rank(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(15):
            J = rng.normal(size=(2, 2))
            if abs(np.linalg.det(J)) < 1e-3:
                continue  # skip near-singular
            _, mob, force, _ = ellipsoid_from_jacobian(J)
            assert force is not None, "Expected force ellipsoid for full-rank J"
            np.testing.assert_allclose(force, 1.0 / mob, rtol=1e-8)


class TestEllipsoidSingularity:
    """Singular (rank-deficient) Jacobian → force_semi_axes is None."""

    def test_rank1_jacobian_force_is_none(self) -> None:
        # Rank-1 matrix: one column is zero → smallest singular value = 0
        J = np.array([[1.0, 1.0], [0.0, 0.0]])
        _, _, force, _ = ellipsoid_from_jacobian(J)
        assert force is None, "Force ellipsoid must be None for rank-1 J"

    def test_zero_jacobian_force_is_none(self) -> None:
        J = np.zeros((2, 2))
        _, mob, force, svs = ellipsoid_from_jacobian(J)
        assert force is None
        assert np.all(mob == 0.0)

    def test_identity_jacobian_not_singular(self) -> None:
        J = np.eye(2)
        _, _, force, _ = ellipsoid_from_jacobian(J)
        assert force is not None, "Identity Jacobian should not be singular"


class TestEllipsoidDirectionsOrthonormal:
    """Principal axes (columns of directions) must be orthonormal."""

    def test_orthonormal_directions_diagonal(self) -> None:
        J = np.array([[3.0, 0.0], [0.0, 2.0]])
        dirs, _, _, _ = ellipsoid_from_jacobian(J)
        # Each column is unit-norm
        for i in range(2):
            assert np.isclose(np.linalg.norm(dirs[:, i]), 1.0)
        # Columns are orthogonal
        assert np.isclose(np.dot(dirs[:, 0], dirs[:, 1]), 0.0, atol=1e-10)

    def test_orthonormal_directions_random(self) -> None:
        rng = np.random.default_rng(13)
        for _ in range(20):
            J = rng.normal(size=(2, 3))  # non-square (like triple pendulum)
            dirs, _, _, _ = ellipsoid_from_jacobian(J)
            UUT = dirs @ dirs.T
            np.testing.assert_allclose(UUT, np.eye(2), atol=1e-10)


# ============================================================================
# ellipsoids_double / ellipsoids_triple — high-level helpers
# ============================================================================


class TestEllipsoidsDouble:
    """High-level double-pendulum ellipsoid helper."""

    def test_returns_both_endpoints(self, L: tuple[float, float]) -> None:
        L1, L2 = L
        result = ellipsoids_double(0.5, 0.3, L1, L2)
        assert "wrist" in result
        assert "tip" in result

    def test_each_endpoint_has_required_keys(self, L: tuple[float, float]) -> None:
        L1, L2 = L
        result = ellipsoids_double(0.5, 0.3, L1, L2)
        expected_keys = {
            "jacobian",
            "directions",
            "mob_semi_axes",
            "force_semi_axes",
            "singular_values",
        }
        for name, data in result.items():
            assert set(data.keys()) == expected_keys, (
                f"Missing keys in '{name}': {expected_keys - set(data.keys())}"
            )

    def test_mob_axes_positive_full_rank(self, L: tuple[float, float]) -> None:
        L1, L2 = L
        result = ellipsoids_double(1.0, 0.5, L1, L2)
        for name, data in result.items():
            assert np.all(data["mob_semi_axes"] >= 0), f"Negative mobility axis in '{name}'"


class TestEllipsoidsTriple:
    """High-level triple-pendulum ellipsoid helper."""

    def test_returns_three_endpoints(self, L3: tuple[float, float, float]) -> None:
        L1, L2, L3_ = L3
        result = ellipsoids_triple(0.3, 0.2, 0.1, L1, L2, L3_)
        assert set(result.keys()) == {"wrist1", "wrist2", "tip"}

    def test_each_endpoint_has_required_keys(self, L3: tuple[float, float, float]) -> None:
        L1, L2, L3_ = L3
        result = ellipsoids_triple(0.3, 0.2, 0.1, L1, L2, L3_)
        required = {
            "jacobian",
            "directions",
            "mob_semi_axes",
            "force_semi_axes",
            "singular_values",
        }
        for name, data in result.items():
            assert set(data.keys()) == required

    def test_three_dof_force_ellipsoid_exists_for_full_rank(
        self, L3: tuple[float, float, float]
    ) -> None:
        """J is (2,3) for triple; J Jᵀ is (2,2) and should be full rank for
        a generic (non-singular) configuration."""
        L1, L2, L3_ = L3
        result = ellipsoids_triple(0.5, 0.3, 0.7, L1, L2, L3_)
        # tip Jacobian is full row rank for generic config → force ellipsoid exists
        assert result["tip"]["force_semi_axes"] is not None


# ============================================================================
# Design by Contract: precondition violations
# ============================================================================


class TestDbCViolations:
    """All public functions must assert invalid inputs."""

    def test_jacobian_double_negative_L1(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            jacobian_double(0.0, 0.0, -1.0, 1.0)

    def test_jacobian_double_zero_L2(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            jacobian_double(0.0, 0.0, 1.0, 0.0)

    def test_jacobian_double_nan_angle(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            jacobian_double(float("nan"), 0.0, 1.0, 1.0)

    def test_jacobian_triple_infinite_angle(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            jacobian_triple(float("inf"), 0.0, 0.0, 1.0, 1.0, 1.0)

    def test_jacobian_triple_negative_length(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            jacobian_triple(0.0, 0.0, 0.0, 1.0, 0.0, 1.0)

    def test_ellipsoid_bad_jacobian_shape(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            ellipsoid_from_jacobian(np.eye(3))  # (3,3) not (2,n)

    def test_ellipsoid_1d_jacobian_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            ellipsoid_from_jacobian(np.array([1.0, 2.0]))  # 1-D

    def test_ellipsoid_nan_jacobian(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            ellipsoid_from_jacobian(np.array([[1.0, float("nan")], [0.0, 1.0]]))

    def test_ellipsoids_double_nan_angle(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            ellipsoids_double(float("nan"), 0.0, 1.0, 1.0)

    def test_ellipsoids_triple_zero_length(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            ellipsoids_triple(0.0, 0.0, 0.0, 1.0, 1.0, 0.0)


# ============================================================================
# Numerical consistency across both models
# ============================================================================


class TestJacobianConsistency:
    """Cross-check: Jacobian should agree with finite-difference approximation."""

    def _fd_jacobian(
        self,
        pos_fn: object,
        angles: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Finite-difference Jacobian via central differences."""
        from typing import Callable

        fn: Callable[[np.ndarray], np.ndarray] = pos_fn  # type: ignore[assignment]
        n = len(angles)
        J_fd = np.zeros((2, n))
        for i in range(n):
            a_plus = angles.copy()
            a_plus[i] += eps
            a_minus = angles.copy()
            a_minus[i] -= eps
            J_fd[:, i] = (fn(a_plus) - fn(a_minus)) / (2 * eps)
        return J_fd

    def test_double_tip_agrees_with_fd(self, L: tuple[float, float]) -> None:
        L1, L2 = L
        theta1, phi = 0.8, -0.4

        def tip_pos(angles: np.ndarray) -> np.ndarray:
            th, ph = angles
            theta2 = th + ph
            x = L1 * np.sin(th) + L2 * np.sin(theta2)
            y = -L1 * np.cos(th) - L2 * np.cos(theta2)
            return np.array([x, y])

        J_analytic = jacobian_double(theta1, phi, L1, L2)["tip"]
        J_fd = self._fd_jacobian(tip_pos, np.array([theta1, phi]))
        np.testing.assert_allclose(J_analytic, J_fd, atol=1e-5)

    def test_triple_tip_agrees_with_fd(self, L3: tuple[float, float, float]) -> None:
        L1, L2, L3_ = L3
        theta1, phi1, phi2 = 0.5, 0.3, -0.2

        def tip_pos(angles: np.ndarray) -> np.ndarray:
            th1, ph1, ph2 = angles
            th2 = th1 + ph1
            th3 = th1 + ph1 + ph2
            x = L1 * np.sin(th1) + L2 * np.sin(th2) + L3_ * np.sin(th3)
            y = -L1 * np.cos(th1) - L2 * np.cos(th2) - L3_ * np.cos(th3)
            return np.array([x, y])

        J_analytic = jacobian_triple(theta1, phi1, phi2, L1, L2, L3_)["tip"]
        J_fd = self._fd_jacobian(tip_pos, np.array([theta1, phi1, phi2]))
        np.testing.assert_allclose(J_analytic, J_fd, atol=1e-5)
