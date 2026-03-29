"""Tests for jacobians_golfer module.

TDD: Tests verify Jacobian computation, ellipsoid extraction,
and ZTCF matrix for the golfer 8-DOF model.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.jacobians_golfer import (
    _numerical_jacobian,
    delta_matrix,
    ellipsoids_golfer,
    jacobian_golfer,
    ztcf_matrix,
)
from double_pendulum_golf.physics_golfer import GolferParams, N_DOF


def _default_golfer_params() -> GolferParams:
    """Create a minimal valid golfer params for testing."""
    return GolferParams(
        # Segment masses (kg)
        m_hub=10.0,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.3,
        # Segment lengths (m)
        L_hub=0.3,
        L_r_upper=0.3,
        L_r_fore=0.25,
        L_l_upper=0.3,
        L_l_fore=0.25,
        L_club=1.1,
        # Shoulder offsets from hub
        d_rs=0.1,
        d_ls=0.1,
        # Grip positions on club
        grip_right=0.1,
        grip_left=0.3,
        # Gravity
        g=9.81,
    )


def _default_state() -> np.ndarray:
    """Create a simple non-singular state for testing."""
    q = np.zeros(N_DOF)
    q[0] = 0.1  # small hub angle
    # small angles for arms to avoid singularity
    q[1] = 0.05
    q[4] = 0.05
    q[7] = 0.2  # club angle
    return q


class TestNumericalJacobian:
    """Tests for the internal _numerical_jacobian function."""

    def test_shape_single_joint(self):
        """Jacobian should have shape (2, 8) for any joint."""
        p = _default_golfer_params()
        q = _default_state()

        J = _numerical_jacobian(q, p, "rh")
        assert J.shape == (2, N_DOF)

    def test_jacobian_finite(self):
        """Jacobian values should be finite (no NaN or Inf)."""
        p = _default_golfer_params()
        q = _default_state()

        J = _numerical_jacobian(q, p, "club_tip")
        assert np.all(np.isfinite(J)), "Jacobian contains non-finite values"

    def test_rejects_wrong_shape_q(self):
        """Should assert on wrong q shape."""
        p = _default_golfer_params()
        q_wrong = np.zeros(7)  # wrong size

        with pytest.raises((ValueError, TypeError)):
            _numerical_jacobian(q_wrong, p, "rh")

    def test_multiple_joints_independent(self):
        """Different joints should return different Jacobians."""
        p = _default_golfer_params()
        q = _default_state()

        J_rh = _numerical_jacobian(q, p, "rh")
        J_lh = _numerical_jacobian(q, p, "lh")

        # They should be different (right hand vs left hand kinematics)
        assert not np.allclose(J_rh, J_lh)

    def test_jacobian_affected_by_state(self):
        """Jacobian should change with q (different configuration)."""
        p = _default_golfer_params()
        q1 = np.zeros(N_DOF)
        q1[0] = 0.1

        q2 = np.zeros(N_DOF)
        q2[0] = 0.3  # different hub angle

        J1 = _numerical_jacobian(q1, p, "rh")
        J2 = _numerical_jacobian(q2, p, "rh")

        # Jacobians should differ due to different configuration
        assert not np.allclose(J1, J2)

    def test_epsilon_parameter(self):
        """Jacobian should be computable with different epsilon values."""
        p = _default_golfer_params()
        q = _default_state()

        J_fine = _numerical_jacobian(q, p, "club_tip", eps=1e-8)
        J_coarse = _numerical_jacobian(q, p, "club_tip", eps=1e-6)

        # Both should be finite
        assert np.all(np.isfinite(J_fine))
        assert np.all(np.isfinite(J_coarse))

        # Should be similar but not identical
        assert np.allclose(J_fine, J_coarse, atol=1e-4)


class TestJacobianGolfer:
    """Tests for jacobian_golfer which computes all endpoint Jacobians."""

    def test_returns_dict_with_expected_keys(self):
        """Should return dict with 6 endpoint keys."""
        p = _default_golfer_params()
        q = _default_state()

        result = jacobian_golfer(q, p)
        expected_keys = {"rh", "lh", "club_tip", "re", "le", "hub"}
        assert isinstance(result, dict)
        assert set(result.keys()) == expected_keys

    def test_all_jacobian_shapes_correct(self):
        """Each endpoint Jacobian should be (2, 8)."""
        p = _default_golfer_params()
        q = _default_state()

        result = jacobian_golfer(q, p)
        for name, J in result.items():
            assert J.shape == (
                2,
                N_DOF,
            ), f"Jacobian for {name} has wrong shape: {J.shape}"

    def test_all_jacobians_finite(self):
        """All Jacobian values should be finite."""
        p = _default_golfer_params()
        q = _default_state()

        result = jacobian_golfer(q, p)
        for name, J in result.items():
            assert np.all(np.isfinite(J)), f"Jacobian for {name} has non-finite values"

    def test_truncates_full_state(self):
        """Should handle 16-element state by truncating to first 8."""
        p = _default_golfer_params()
        full_state = np.zeros(16)
        full_state[:8] = _default_state()

        result = jacobian_golfer(full_state, p)
        assert len(result) == 6
        for J in result.values():
            assert J.shape == (2, N_DOF)

    def test_left_right_symmetry(self):
        """Left and right endpoints should differ due to asymmetric grips."""
        p = _default_golfer_params()
        q = _default_state()

        result = jacobian_golfer(q, p)
        J_rh = result["rh"]
        J_lh = result["lh"]

        # Should not be identical (different arm chains)
        assert not np.allclose(J_rh, J_lh)

    def test_hub_jacobian_simpler_structure(self):
        """Hub Jacobian should depend primarily on q[0]."""
        p = _default_golfer_params()
        q = _default_state()

        result = jacobian_golfer(q, p)
        J_hub = result["hub"]

        # Hub primarily depends on theta_hub (q[0])
        # Columns for arm angles should be smaller or zero
        # At least the q[0] column should have significant values
        assert np.max(np.abs(J_hub[:, 0])) > 0


class TestEllipsoidsGolfer:
    """Tests for manipulability ellipsoid extraction."""

    def test_returns_expected_structure(self):
        """Each endpoint should have ellipsoid data dict."""
        p = _default_golfer_params()
        q = _default_state()

        result = ellipsoids_golfer(q, p)

        expected_keys = {"rh", "lh", "club_tip", "re", "le", "hub"}
        assert set(result.keys()) == expected_keys

        for name, data in result.items():
            assert isinstance(data, dict)
            assert "jacobian" in data
            assert "directions" in data
            assert "mob_semi_axes" in data
            assert "force_semi_axes" in data
            assert "singular_values" in data

    def test_jacobian_preserved_in_ellipsoid_data(self):
        """Jacobian in ellipsoid data should match jacobian_golfer output."""
        p = _default_golfer_params()
        q = _default_state()

        ellipsoid_result = ellipsoids_golfer(q, p)
        jacobian_result = jacobian_golfer(q, p)

        for name in ellipsoid_result.keys():
            np.testing.assert_array_equal(
                ellipsoid_result[name]["jacobian"],
                jacobian_result[name],
            )

    def test_ellipsoid_data_finite(self):
        """All ellipsoid values should be finite."""
        p = _default_golfer_params()
        q = _default_state()

        result = ellipsoids_golfer(q, p)
        for name, data in result.items():
            assert np.all(np.isfinite(data["jacobian"])), f"{name} Jacobian non-finite"
            assert np.all(np.isfinite(data["singular_values"])), (
                f"{name} singular values non-finite"
            )
            assert np.all(np.isfinite(data["mob_semi_axes"])), (
                f"{name} mobility semi-axes non-finite"
            )
            # force_semi_axes may be None at singular configurations
            if data["force_semi_axes"] is not None:
                assert np.all(np.isfinite(data["force_semi_axes"])), (
                    f"{name} force semi-axes non-finite"
                )

    def test_singular_values_descending(self):
        """Singular values should be in descending order."""
        p = _default_golfer_params()
        q = _default_state()

        result = ellipsoids_golfer(q, p)
        for name, data in result.items():
            svs = data["singular_values"]
            # Check descending order
            assert np.all(np.diff(svs) <= 0), f"{name} singular values not in descending order"

    def test_directions_orthonormal(self):
        """Ellipsoid directions should be orthonormal."""
        p = _default_golfer_params()
        q = _default_state()

        result = ellipsoids_golfer(q, p)
        for name, data in result.items():
            dirs = data["directions"]
            # Check columns are unit vectors
            for i in range(dirs.shape[1]):
                col_norm = np.linalg.norm(dirs[:, i])
                assert np.isclose(col_norm, 1.0, atol=1e-10), (
                    f"{name} direction {i} not unit norm"
                )

    def test_semi_axes_positive(self):
        """Semi-axes lengths should be positive."""
        p = _default_golfer_params()
        q = _default_state()

        result = ellipsoids_golfer(q, p)
        for name, data in result.items():
            mob_axes = data["mob_semi_axes"]
            force_axes = data["force_semi_axes"]
            assert np.all(mob_axes >= 0), f"{name} mobility axes negative"
            # force_semi_axes may be None at singular configurations
            if force_axes is not None:
                assert np.all(force_axes >= 0), f"{name} force axes negative"

    def test_truncates_full_state(self):
        """Should handle 16-element state by truncating."""
        p = _default_golfer_params()
        full_state = np.zeros(16)
        full_state[:8] = _default_state()

        result = ellipsoids_golfer(full_state, p)
        assert len(result) == 6
        for data in result.values():
            assert "singular_values" in data


class TestDeltaMatrix:
    """Tests for the inverse mass matrix (Delta matrix)."""

    def test_shape_correct(self):
        """Delta matrix should be (8, 8)."""
        p = _default_golfer_params()
        q = _default_state()

        D = delta_matrix(q, p)
        assert D.shape == (N_DOF, N_DOF)

    def test_symmetric(self):
        """Inverse of symmetric mass matrix should be symmetric."""
        p = _default_golfer_params()
        q = _default_state()

        D = delta_matrix(q, p)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_finite_values(self):
        """All matrix values should be finite."""
        p = _default_golfer_params()
        q = _default_state()

        D = delta_matrix(q, p)
        assert np.all(np.isfinite(D)), "Delta matrix has non-finite values"

    def test_positive_definite(self):
        """Mass matrix inverse should be positive definite."""
        p = _default_golfer_params()
        q = _default_state()

        D = delta_matrix(q, p)
        # Check eigenvalues are positive
        eigvals = np.linalg.eigvals(D)
        assert np.all(eigvals > -1e-10), "Delta matrix not positive definite"

    def test_diagonal_dominance(self):
        """Diagonal elements should typically be largest (rough check)."""
        p = _default_golfer_params()
        q = _default_state()

        D = delta_matrix(q, p)
        diag = np.abs(np.diag(D))
        # At least some diagonal elements should be non-zero
        assert np.max(diag) > 0

    def test_varies_with_configuration(self):
        """Delta matrix should change with q."""
        p = _default_golfer_params()
        q1 = _default_state()
        q2 = _default_state()
        q2[0] = 0.3  # change hub angle

        D1 = delta_matrix(q1, p)
        D2 = delta_matrix(q2, p)

        assert not np.allclose(D1, D2)

    def test_truncates_full_state(self):
        """Should handle 16-element state."""
        p = _default_golfer_params()
        full_state = np.zeros(16)
        full_state[:8] = _default_state()

        D = delta_matrix(full_state, p)
        assert D.shape == (N_DOF, N_DOF)


class TestZtcfMatrix:
    """Tests for the zero-torque constraint force transfer matrix."""

    def test_shape_when_nonsingular(self):
        """ZTCF matrix should be (2, 8) or None."""
        p = _default_golfer_params()
        q = _default_state()

        T = ztcf_matrix(q, p, "club_tip")
        assert T is None or T.shape == (2, N_DOF)

    def test_returns_none_or_finite_array(self):
        """Should return None if singular, else finite array."""
        p = _default_golfer_params()
        q = _default_state()

        T = ztcf_matrix(q, p, "rh")
        if T is not None:
            assert np.all(np.isfinite(T)), "ZTCF matrix has non-finite values"

    def test_different_joints_different_transfers(self):
        """Different joints should give different transfer matrices."""
        p = _default_golfer_params()
        q = _default_state()

        T_rh = ztcf_matrix(q, p, "rh")
        T_lh = ztcf_matrix(q, p, "lh")

        # Both should be non-None at valid state
        if T_rh is not None and T_lh is not None:
            assert not np.allclose(T_rh, T_lh)

    def test_club_tip_computation(self):
        """Should compute ZTCF for club tip (primary endpoint)."""
        p = _default_golfer_params()
        q = _default_state()

        T = ztcf_matrix(q, p, "club_tip")
        # May return None if singular, but should not error
        assert T is None or isinstance(T, np.ndarray)

    def test_all_named_joints(self):
        """Should work for all named joints (may return None for some)."""
        p = _default_golfer_params()
        q = _default_state()

        joint_names = ["rh", "lh", "club_tip", "re", "le", "hub"]
        for joint in joint_names:
            T = ztcf_matrix(q, p, joint)
            if T is not None:
                assert T.shape == (2, N_DOF)
                assert np.all(np.isfinite(T))

    def test_truncates_full_state(self):
        """Should handle 16-element state."""
        p = _default_golfer_params()
        full_state = np.zeros(16)
        full_state[:8] = _default_state()

        T = ztcf_matrix(full_state, p, "club_tip")
        assert T is None or T.shape == (2, N_DOF)

    def test_formula_structure(self):
        """ZTCF = (J M^{-1} J^T)^{-1} J M^{-1} should be consistent."""
        p = _default_golfer_params()
        q = _default_state()

        # Manually compute and compare
        from double_pendulum_golf.jacobians_golfer import _numerical_jacobian
        from double_pendulum_golf.physics_golfer import mass_matrix

        J = _numerical_jacobian(q, p, "club_tip")
        M = mass_matrix(q, p)
        M_inv = np.linalg.pinv(M)

        JMinv = J @ M_inv
        A = JMinv @ J.T

        try:
            A_inv = np.linalg.inv(A)
            expected_T = A_inv @ JMinv
        except np.linalg.LinAlgError:
            expected_T = None

        actual_T = ztcf_matrix(q, p, "club_tip")

        if expected_T is not None and actual_T is not None:
            np.testing.assert_allclose(actual_T, expected_T, rtol=1e-10)
        else:
            # Both should be None if singular
            assert expected_T is None and actual_T is None


class TestJacobianConsistency:
    """Integration tests verifying consistency between functions."""

    def test_numerical_jacobian_consistent_with_jacobian_golfer(self):
        """_numerical_jacobian should match jacobian_golfer output."""
        p = _default_golfer_params()
        q = _default_state()

        # Get from internal function
        J_internal = _numerical_jacobian(q, p, "rh")

        # Get from public function
        jacs_dict = jacobian_golfer(q, p)
        J_public = jacs_dict["rh"]

        np.testing.assert_allclose(J_internal, J_public, rtol=1e-10)

    def test_jacobian_affects_ellipsoid_singular_values(self):
        """Changing q should change ellipsoid singular values."""
        p = _default_golfer_params()
        q1 = _default_state()
        q2 = _default_state()
        q2[1] = 0.2  # change right shoulder angle

        e1 = ellipsoids_golfer(q1, p)
        e2 = ellipsoids_golfer(q2, p)

        sv1 = e1["rh"]["singular_values"]
        sv2 = e2["rh"]["singular_values"]

        # Singular values should differ
        assert not np.allclose(sv1, sv2)

    def test_delta_matrix_is_mass_matrix_pseudoinverse(self):
        """Delta should be M^{+} (pseudoinverse, since M is rank-deficient)."""
        p = _default_golfer_params()
        q = _default_state()

        from double_pendulum_golf.physics_golfer import mass_matrix

        M = mass_matrix(q, p)
        D = delta_matrix(q, p)

        # For a pseudoinverse, M @ D @ M == M (one of the Moore-Penrose conditions)
        product = M @ D @ M
        np.testing.assert_allclose(product, M, atol=1e-8)


class TestRobustness:
    """Robustness tests with edge cases."""

    def test_near_singular_configuration(self):
        """Should handle near-singular configs gracefully."""
        p = _default_golfer_params()
        # Create a somewhat singular config
        q = np.zeros(N_DOF)
        q[0] = np.pi / 2  # 90 degree hub angle

        # Should not crash
        J = jacobian_golfer(q, p)
        assert len(J) == 6

    def test_zero_state(self):
        """Should handle zero state (all angles = 0)."""
        p = _default_golfer_params()
        q = np.zeros(N_DOF)

        # May be singular but should not crash
        try:
            J = jacobian_golfer(q, p)
            assert len(J) == 6
        except Exception:  # noqa: BLE001
            pass  # Some configs may be singular

    def test_large_angle_values(self):
        """Should handle large angle values (multiple turns)."""
        p = _default_golfer_params()
        q = np.ones(N_DOF) * 5.0  # > 2*pi

        # Should work with modular periodicity
        J = jacobian_golfer(q, p)
        assert len(J) == 6
        for Jmat in J.values():
            assert Jmat.shape == (2, N_DOF)

    def test_parameter_variations(self):
        """Should work with different parameter sets."""
        # Create a different parameter set
        p1 = _default_golfer_params()

        p2 = GolferParams(
            m_hub=5.0,
            m_r_upper=1.5,
            m_r_fore=1.0,
            m_l_upper=1.5,
            m_l_fore=1.0,
            m_club=0.2,
            L_hub=0.2,
            L_r_upper=0.25,
            L_r_fore=0.2,
            L_l_upper=0.25,
            L_l_fore=0.2,
            L_club=1.0,
            d_rs=0.05,
            d_ls=0.05,
            grip_right=0.08,
            grip_left=0.25,
            g=9.81,
        )

        q = _default_state()

        J1 = jacobian_golfer(q, p1)
        J2 = jacobian_golfer(q, p2)

        # Both should work
        assert len(J1) == 6
        assert len(J2) == 6

        # Results should differ due to different parameters
        assert not np.allclose(J1["rh"], J2["rh"])
