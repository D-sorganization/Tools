# ruff: noqa: E501
"""Tests for jacobians.py and jacobians_golfer.py.

Covers:
- jacobians.py: ellipsoid_from_jacobian, jacobian_double, ellipsoids_double,
                jacobian_triple, ellipsoids_triple
- jacobians_golfer.py: jacobian_golfer, ellipsoids_golfer, delta_matrix, ztcf_matrix
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
from double_pendulum_golf.jacobians_golfer import (
    delta_matrix,
    ellipsoids_golfer,
    jacobian_golfer,
    ztcf_matrix,
)
from double_pendulum_golf.physics_golfer import GolferParams, N_DOF

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def golfer_params() -> GolferParams:
    return GolferParams(
        m_hub=2.0,
        m_r_upper=3.0,
        m_r_fore=2.0,
        m_l_upper=3.0,
        m_l_fore=2.0,
        m_club=0.5,
        L_hub=0.15,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.10,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
    )


@pytest.fixture
def zero_q() -> np.ndarray:
    return np.zeros(N_DOF)


# ===========================================================================
# Tests for ellipsoid_from_jacobian
# ===========================================================================


class TestEllipsoidFromJacobian:
    def test_returns_four_items(self) -> None:
        J = np.eye(2)
        result = ellipsoid_from_jacobian(J)
        assert len(result) == 4

    def test_directions_shape(self) -> None:
        J = np.array([[1.0, 0.0], [0.0, 1.0]])
        dirs, _, _, _ = ellipsoid_from_jacobian(J)
        assert dirs.shape == (2, 2)

    def test_mob_semi_axes_shape(self) -> None:
        J = np.eye(2)
        _, mob, _, _ = ellipsoid_from_jacobian(J)
        assert mob.shape == (2,)

    def test_mob_semi_axes_non_negative(self) -> None:
        J = np.array([[2.0, 0.5], [0.0, 1.5]])
        _, mob, _, _ = ellipsoid_from_jacobian(J)
        assert np.all(mob >= 0)

    def test_force_semi_axes_none_at_singularity(self) -> None:
        """Near-singular J should return None for force ellipsoid."""
        J = np.array([[1.0, 0.0], [0.0, 1e-12]])  # nearly singular
        _, _, force, _ = ellipsoid_from_jacobian(J)
        assert force is None

    def test_force_semi_axes_valid_away_from_singularity(self) -> None:
        J = np.array([[2.0, 0.5], [0.5, 1.5]])
        _, _, force, _ = ellipsoid_from_jacobian(J)
        assert force is not None
        assert force.shape == (2,)
        assert np.all(force > 0)

    def test_wide_jacobian_2xN(self) -> None:
        J = np.random.default_rng(42).standard_normal((2, 5))
        dirs, mob, _, svs = ellipsoid_from_jacobian(J)
        assert dirs.shape == (2, 2)
        assert mob.shape == (2,)
        assert svs.shape == (2,)

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            ellipsoid_from_jacobian(np.ones((3, 2)))  # wrong row count

    def test_non_finite_raises(self) -> None:
        J = np.array([[1.0, np.nan], [0.0, 1.0]])
        with pytest.raises((ValueError, TypeError)):
            ellipsoid_from_jacobian(J)

    def test_not_array_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            ellipsoid_from_jacobian([[1, 0], [0, 1]])  # list, not ndarray


# ===========================================================================
# Tests for jacobian_double
# ===========================================================================


class TestJacobianDouble:
    def test_returns_dict_with_two_keys(self) -> None:
        result = jacobian_double(0.0, 0.0, L1=0.6, L2=1.0)
        assert set(result.keys()) == {"wrist", "tip"}

    def test_wrist_shape(self) -> None:
        result = jacobian_double(0.3, -0.1, L1=0.6, L2=1.0)
        assert result["wrist"].shape == (2, 2)

    def test_tip_shape(self) -> None:
        result = jacobian_double(0.3, -0.1, L1=0.6, L2=1.0)
        assert result["tip"].shape == (2, 2)

    def test_wrist_independent_of_phi(self) -> None:
        """Wrist Jacobian should have zero in second column (phi doesn't affect wrist)."""
        result = jacobian_double(0.5, 0.2, L1=0.6, L2=1.0)
        assert result["wrist"][0, 1] == pytest.approx(0.0)
        assert result["wrist"][1, 1] == pytest.approx(0.0)

    def test_finite(self) -> None:
        result = jacobian_double(np.pi / 4, np.pi / 6, L1=0.6, L2=1.0)
        for J in result.values():
            assert np.all(np.isfinite(J))

    def test_negative_L_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            jacobian_double(0.0, 0.0, L1=-0.5, L2=1.0)


class TestEllipsoidsDouble:
    def test_returns_two_endpoints(self) -> None:
        result = ellipsoids_double(0.0, 0.0, L1=0.6, L2=1.0)
        assert set(result.keys()) == {"wrist", "tip"}

    def test_has_required_subkeys(self) -> None:
        result = ellipsoids_double(0.3, -0.2, L1=0.5, L2=0.8)
        for endpoint in ("wrist", "tip"):
            for key in ("jacobian", "directions", "mob_semi_axes", "singular_values"):
                assert key in result[endpoint], f"Missing {key} in {endpoint}"

    def test_mob_semi_axes_non_negative(self) -> None:
        result = ellipsoids_double(0.2, 0.1, L1=0.6, L2=1.0)
        for endpoint in result.values():
            assert np.all(endpoint["mob_semi_axes"] >= 0)


# ===========================================================================
# Tests for jacobian_triple
# ===========================================================================


class TestJacobianTriple:
    def test_returns_three_keys(self) -> None:
        result = jacobian_triple(0.0, 0.0, 0.0, L1=0.6, L2=0.6, L3=0.6)
        assert set(result.keys()) == {"wrist1", "wrist2", "tip"}

    def test_wrist1_shape(self) -> None:
        result = jacobian_triple(0.1, 0.05, -0.05, L1=0.6, L2=0.6, L3=0.6)
        assert result["wrist1"].shape == (2, 3)

    def test_wrist1_independent_of_phi(self) -> None:
        """Wrist1 only depends on theta1; columns 1 and 2 should be zero."""
        result = jacobian_triple(0.3, 0.2, 0.1, L1=0.6, L2=0.6, L3=0.6)
        assert result["wrist1"][0, 1] == pytest.approx(0.0)
        assert result["wrist1"][0, 2] == pytest.approx(0.0)

    def test_wrist2_independent_of_phi2(self) -> None:
        """Wrist2 column 2 (d/dphi2) should be zero."""
        result = jacobian_triple(0.3, 0.2, 0.1, L1=0.6, L2=0.6, L3=0.6)
        assert result["wrist2"][0, 2] == pytest.approx(0.0)
        assert result["wrist2"][1, 2] == pytest.approx(0.0)

    def test_finite(self) -> None:
        result = jacobian_triple(np.pi / 4, np.pi / 6, np.pi / 8, 0.5, 0.6, 0.5)
        for J in result.values():
            assert np.all(np.isfinite(J))

    def test_zero_L3_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            jacobian_triple(0.0, 0.0, 0.0, L1=0.6, L2=0.6, L3=0.0)


class TestEllipsoidsTriple:
    def test_returns_three_endpoints(self) -> None:
        result = ellipsoids_triple(0.0, 0.0, 0.0, L1=0.6, L2=0.6, L3=0.6)
        assert set(result.keys()) == {"wrist1", "wrist2", "tip"}

    def test_singular_values_non_negative(self) -> None:
        result = ellipsoids_triple(0.1, 0.05, -0.05, 0.6, 0.6, 0.6)
        for ep in result.values():
            assert np.all(ep["singular_values"] >= 0)


# ===========================================================================
# Tests for jacobians_golfer
# ===========================================================================


class TestJacobianGolfer:
    def test_returns_dict(
        self, golfer_params: GolferParams, zero_q: np.ndarray
    ) -> None:
        J = jacobian_golfer(zero_q, golfer_params)
        assert isinstance(J, dict)

    def test_joint_key_shapes(
        self, golfer_params: GolferParams, zero_q: np.ndarray
    ) -> None:
        J = jacobian_golfer(zero_q, golfer_params)
        for name, mat in J.items():
            assert mat.shape == (2, N_DOF), f"Wrong shape for joint {name}"

    def test_finite(self, golfer_params: GolferParams, zero_q: np.ndarray) -> None:
        J = jacobian_golfer(zero_q, golfer_params)
        for mat in J.values():
            assert np.all(np.isfinite(mat))


class TestEllipsoidsGolfer:
    def test_returns_dict(
        self, golfer_params: GolferParams, zero_q: np.ndarray
    ) -> None:
        result = ellipsoids_golfer(zero_q, golfer_params)
        assert isinstance(result, dict)

    def test_has_required_subkeys(
        self, golfer_params: GolferParams, zero_q: np.ndarray
    ) -> None:
        result = ellipsoids_golfer(zero_q, golfer_params)
        for endpoint, data in result.items():
            for key in ("jacobian", "directions", "mob_semi_axes"):
                assert key in data, f"Missing {key} in {endpoint}"


class TestDeltaMatrix:
    def test_shape(self, golfer_params: GolferParams, zero_q: np.ndarray) -> None:
        D = delta_matrix(zero_q, golfer_params)
        assert D.shape == (N_DOF, N_DOF)

    def test_finite(self, golfer_params: GolferParams, zero_q: np.ndarray) -> None:
        D = delta_matrix(zero_q, golfer_params)
        assert np.all(np.isfinite(D))


class TestZtcfMatrix:
    def test_shape(self, golfer_params: GolferParams, zero_q: np.ndarray) -> None:
        Z = ztcf_matrix(zero_q, golfer_params)
        assert Z.shape == (2, N_DOF)

    def test_finite(self, golfer_params: GolferParams, zero_q: np.ndarray) -> None:
        Z = ztcf_matrix(zero_q, golfer_params)
        assert np.all(np.isfinite(Z))

    def test_different_joint(
        self, golfer_params: GolferParams, zero_q: np.ndarray
    ) -> None:
        Z_tip = ztcf_matrix(zero_q, golfer_params, joint_name="club_tip")
        Z_rh = ztcf_matrix(zero_q, golfer_params, joint_name="rh")
        # Different joints should give different matrices
        assert not np.allclose(Z_tip, Z_rh)
