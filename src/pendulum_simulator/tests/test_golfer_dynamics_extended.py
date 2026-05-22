# ruff: noqa: E501
"""Extended tests for golfer_dynamics.py — covering internal helpers and API.

These tests augment test_golfer_dynamics.py to reach >80% coverage by
exercising:
- _TrigCache trig computation
- _hub_and_shoulder_jacobians / _right_arm_chain_jacobian / _left_arm_chain_jacobian
- _club_jacobians helper
- potential_energy() (takes full state vector)
- total_energy() (T + V from full state)
- analytical_mass_matrix() mathematical properties
- analytical_coriolis() shape and zero-velocity edge case
- analytical_gravity_vector() mathematical properties
- kinetic_energy() properties
- State with shape (16,) truncation paths
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from double_pendulum_golf.golfer_dynamics import (
    _TrigCache,
    _club_jacobians,
    _hub_and_shoulder_jacobians,
    _left_arm_chain_jacobian,
    _mass_point_positions,
    _right_arm_chain_jacobian,
    analytical_coriolis,
    analytical_fk_jacobians,
    analytical_gravity_vector,
    analytical_mass_matrix,
    kinetic_energy,
    potential_energy,
    potential_energy_from_q,
    total_energy,
)
from double_pendulum_golf.physics_golfer import GolferParams, N_DOF

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def params() -> GolferParams:
    return GolferParams(
        m_hub=0.01,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.3,
        L_hub=0.2,
        L_r_upper=0.3,
        L_r_fore=0.3,
        L_l_upper=0.3,
        L_l_fore=0.3,
        L_club=1.0,
        d_rs=0.1,
        d_ls=0.1,
        grip_right=0.2,
        grip_left=0.3,
        m_clubhead=0.2,
    )


@pytest.fixture
def zero_q() -> np.ndarray:
    return np.zeros(N_DOF)


@pytest.fixture
def zero_qdot() -> np.ndarray:
    return np.zeros(N_DOF)


@pytest.fixture
def full_state(zero_q: np.ndarray, zero_qdot: np.ndarray) -> np.ndarray:
    return np.concatenate([zero_q, zero_qdot])


# ---------------------------------------------------------------------------
# Tests for _TrigCache
# ---------------------------------------------------------------------------


class TestTrigCache:
    """_TrigCache must pre-compute correct trig values from q."""

    def test_zero_angles(self) -> None:
        q = np.zeros(N_DOF)
        tc = _TrigCache(q)
        assert tc.sin_hub == pytest.approx(0.0, abs=1e-12)
        assert tc.cos_hub == pytest.approx(1.0, abs=1e-12)
        assert tc.sin_rs == pytest.approx(0.0, abs=1e-12)
        assert tc.cos_rs == pytest.approx(1.0, abs=1e-12)
        assert tc.sin_re == pytest.approx(0.0, abs=1e-12)
        assert tc.cos_re == pytest.approx(1.0, abs=1e-12)
        assert tc.sin_ls == pytest.approx(0.0, abs=1e-12)
        assert tc.cos_ls == pytest.approx(1.0, abs=1e-12)
        assert tc.sin_le == pytest.approx(0.0, abs=1e-12)
        assert tc.cos_le == pytest.approx(1.0, abs=1e-12)
        assert tc.sin_club == pytest.approx(0.0, abs=1e-12)
        assert tc.cos_club == pytest.approx(1.0, abs=1e-12)

    def test_hub_angle_propagates(self) -> None:
        q = np.zeros(N_DOF)
        q[0] = np.pi / 4
        tc = _TrigCache(q)
        assert tc.sin_hub == pytest.approx(np.sin(np.pi / 4), abs=1e-12)
        assert tc.cos_hub == pytest.approx(np.cos(np.pi / 4), abs=1e-12)

    def test_absolute_angle_for_rs(self) -> None:
        """theta_rs_abs = theta_hub + alpha_rs."""
        q = np.zeros(N_DOF)
        q[0] = 0.3  # hub
        q[1] = 0.5  # alpha_rs
        tc = _TrigCache(q)
        expected_abs = 0.3 + 0.5
        assert tc.sin_rs == pytest.approx(np.sin(expected_abs))
        assert tc.cos_rs == pytest.approx(np.cos(expected_abs))

    def test_absolute_angle_for_re(self) -> None:
        """theta_re_abs = theta_hub + alpha_rs + alpha_re."""
        q = np.zeros(N_DOF)
        q[0] = 0.2
        q[1] = 0.3
        q[2] = 0.4
        tc = _TrigCache(q)
        expected_abs = 0.2 + 0.3 + 0.4
        assert tc.sin_re == pytest.approx(np.sin(expected_abs))
        assert tc.cos_re == pytest.approx(np.cos(expected_abs))

    def test_absolute_angle_for_ls(self) -> None:
        """theta_ls_abs = theta_hub + alpha_ls (q[4])."""
        q = np.zeros(N_DOF)
        q[0] = 0.1
        q[4] = 0.6
        tc = _TrigCache(q)
        expected_abs = 0.1 + 0.6
        assert tc.sin_ls == pytest.approx(np.sin(expected_abs))
        assert tc.cos_ls == pytest.approx(np.cos(expected_abs))

    def test_club_angle(self) -> None:
        """Club angle is q[7] (absolute, not relative)."""
        q = np.zeros(N_DOF)
        q[7] = np.pi / 3
        tc = _TrigCache(q)
        assert tc.sin_club == pytest.approx(np.sin(np.pi / 3))
        assert tc.cos_club == pytest.approx(np.cos(np.pi / 3))

    def test_trig_identity(self) -> None:
        """sin^2 + cos^2 = 1 for every joint."""
        q = np.random.default_rng(5).uniform(-np.pi, np.pi, N_DOF)
        tc = _TrigCache(q)
        for s, c in [
            (tc.sin_hub, tc.cos_hub),
            (tc.sin_rs, tc.cos_rs),
            (tc.sin_re, tc.cos_re),
            (tc.sin_ls, tc.cos_ls),
            (tc.sin_le, tc.cos_le),
            (tc.sin_club, tc.cos_club),
        ]:
            assert s**2 + c**2 == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Tests for Jacobian helpers
# ---------------------------------------------------------------------------


class TestHubAndShoulderJacobians:
    def test_returns_three_jacobians(self, params: GolferParams) -> None:
        tc = _TrigCache(np.zeros(N_DOF))
        result = _hub_and_shoulder_jacobians(params, tc)
        assert set(result.keys()) == {"hub", "rs", "ls"}

    def test_each_jacobian_shape(self, params: GolferParams) -> None:
        tc = _TrigCache(np.zeros(N_DOF))
        result = _hub_and_shoulder_jacobians(params, tc)
        for key, J in result.items():
            assert J.shape == (2, N_DOF), f"Wrong shape for {key}: {J.shape}"

    def test_hub_jacobian_depends_on_q0_only(self, params: GolferParams) -> None:
        """Hub jacobian should only have non-zero entries in column 0."""
        tc = _TrigCache(np.zeros(N_DOF))
        J_hub = _hub_and_shoulder_jacobians(params, tc)["hub"]
        # Only column 0 should be non-zero
        assert np.all(J_hub[:, 1:] == 0.0)
        assert not np.all(J_hub[:, 0] == 0.0)


class TestRightArmChainJacobian:
    def test_returns_three_jacobians(self, params: GolferParams) -> None:
        tc = _TrigCache(np.zeros(N_DOF))
        J_re, J_rh_1, J_rh_2 = _right_arm_chain_jacobian(params, tc)
        assert J_re.shape == (2, N_DOF)
        assert J_rh_1.shape == (2, N_DOF)
        assert J_rh_2.shape == (2, N_DOF)

    def test_j_rh_returned_twice(self, params: GolferParams) -> None:
        """The function returns J_rh as both 2nd and 3rd return values."""
        tc = _TrigCache(np.zeros(N_DOF))
        _, J_rh_1, J_rh_2 = _right_arm_chain_jacobian(params, tc)
        np.testing.assert_array_equal(J_rh_1, J_rh_2)

    def test_re_not_depend_on_q2(self, params: GolferParams) -> None:
        """RE position only depends on q[0] and q[1], not q[2]+."""
        tc = _TrigCache(np.zeros(N_DOF))
        J_re, _, _ = _right_arm_chain_jacobian(params, tc)
        # Columns 2..7 should all be zero for RE
        assert np.all(J_re[:, 2:] == 0.0)


class TestLeftArmChainJacobian:
    def test_returns_two_jacobians(self, params: GolferParams) -> None:
        tc = _TrigCache(np.zeros(N_DOF))
        J_le, J_lh = _left_arm_chain_jacobian(params, tc)
        assert J_le.shape == (2, N_DOF)
        assert J_lh.shape == (2, N_DOF)

    def test_le_not_depend_on_right_arm_angles(self, params: GolferParams) -> None:
        """LE should not depend on q[1], q[2], q[3] (right arm joints)."""
        tc = _TrigCache(np.zeros(N_DOF))
        J_le, _ = _left_arm_chain_jacobian(params, tc)
        assert np.all(J_le[:, 1:4] == 0.0)


class TestClubJacobians:
    def test_returns_two_jacobians(self, params: GolferParams) -> None:
        tc = _TrigCache(np.zeros(N_DOF))
        _, J_rh, _ = _right_arm_chain_jacobian(params, tc)
        J_com, J_tip = _club_jacobians(params, tc, J_rh)
        assert J_com.shape == (2, N_DOF)
        assert J_tip.shape == (2, N_DOF)

    def test_club_jacobian_modifies_column_7(self, params: GolferParams) -> None:
        """Club Jacobians must set column 7 based on club angle."""
        q = np.zeros(N_DOF)
        q[7] = np.pi / 4  # non-trivial club angle
        tc = _TrigCache(q)
        _, J_rh, _ = _right_arm_chain_jacobian(params, tc)
        J_com, J_tip = _club_jacobians(params, tc, J_rh)
        # Column 7 should be non-zero
        assert not np.all(J_com[:, 7] == 0.0) or not np.all(J_tip[:, 7] == 0.0)


# ---------------------------------------------------------------------------
# Tests for analytical FK Jacobians (public API)
# ---------------------------------------------------------------------------


class TestAnalyticalFKJacobians:
    def test_all_keys_present(self, params: GolferParams, zero_q: np.ndarray) -> None:
        J = analytical_fk_jacobians(zero_q, params)
        expected = {"hub", "rs", "re", "rh", "ls", "le", "lh", "club_com", "club_tip"}
        assert set(J.keys()) == expected

    def test_all_shapes_correct(self, params: GolferParams, zero_q: np.ndarray) -> None:
        J = analytical_fk_jacobians(zero_q, params)
        for key, jac in J.items():
            assert jac.shape == (2, N_DOF), f"Wrong shape for {key}: {jac.shape}"

    def test_extended_q_truncated(self, params: GolferParams) -> None:
        q_extended = np.zeros(2 * N_DOF)
        J_ext = analytical_fk_jacobians(q_extended, params)
        J_short = analytical_fk_jacobians(np.zeros(N_DOF), params)
        np.testing.assert_allclose(J_ext["hub"], J_short["hub"])


# ---------------------------------------------------------------------------
# Tests for analytical_mass_matrix (mathematical properties)
# ---------------------------------------------------------------------------


class TestAnalyticalMassMatrix:
    def test_shape(self, params: GolferParams, zero_q: np.ndarray) -> None:
        M = analytical_mass_matrix(zero_q, params)
        assert M.shape == (N_DOF, N_DOF)

    def test_symmetric(self, params: GolferParams) -> None:
        rng = np.random.default_rng(3)
        for _ in range(5):
            q = rng.uniform(-np.pi / 4, np.pi / 4, N_DOF)
            with patch(
                "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
                return_value=None,
            ):
                M = analytical_mass_matrix(q, params)
            np.testing.assert_allclose(M, M.T, atol=1e-8)

    def test_positive_semidefinite(
        self, params: GolferParams, zero_q: np.ndarray
    ) -> None:
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            M = analytical_mass_matrix(zero_q, params)
        eigs = np.linalg.eigvalsh(M)
        assert np.all(eigs >= -1e-10)

    def test_extended_q_truncated(self, params: GolferParams) -> None:
        q_ext = np.zeros(2 * N_DOF)
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            M_ext = analytical_mass_matrix(q_ext, params)
            M_nrm = analytical_mass_matrix(np.zeros(N_DOF), params)
        np.testing.assert_allclose(M_ext, M_nrm)

    def test_native_bypass(self, params: GolferParams, zero_q: np.ndarray) -> None:
        fake_M = np.eye(N_DOF)
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=fake_M,
        ):
            M = analytical_mass_matrix(zero_q, params)
        assert M is fake_M


# ---------------------------------------------------------------------------
# Tests for analytical_gravity_vector
# ---------------------------------------------------------------------------


class TestAnalyticalGravityVector:
    def test_shape(self, params: GolferParams, zero_q: np.ndarray) -> None:
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_gravity_vector",
            return_value=None,
        ):
            G = analytical_gravity_vector(zero_q, params)
        assert G.shape == (N_DOF,)

    def test_finite(self, params: GolferParams, zero_q: np.ndarray) -> None:
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_gravity_vector",
            return_value=None,
        ):
            G = analytical_gravity_vector(zero_q, params)
        assert np.all(np.isfinite(G))

    def test_nonzero_when_displaced(self, params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        q[0] = np.pi / 3
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_gravity_vector",
            return_value=None,
        ):
            G = analytical_gravity_vector(q, params)
        assert not np.allclose(G, 0.0)

    def test_extended_q_truncated(self, params: GolferParams) -> None:
        q_ext = np.zeros(2 * N_DOF)
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_gravity_vector",
            return_value=None,
        ):
            G_ext = analytical_gravity_vector(q_ext, params)
            G_nrm = analytical_gravity_vector(np.zeros(N_DOF), params)
        np.testing.assert_allclose(G_ext, G_nrm)

    def test_native_bypass(self, params: GolferParams, zero_q: np.ndarray) -> None:
        fake_G = np.ones(N_DOF)
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_gravity_vector",
            return_value=fake_G,
        ):
            G = analytical_gravity_vector(zero_q, params)
        assert G is fake_G


# ---------------------------------------------------------------------------
# Tests for analytical_coriolis
# ---------------------------------------------------------------------------


class TestAnalyticalCoriolis:
    def test_zero_velocity_gives_zero_coriolis(
        self, params: GolferParams, zero_q: np.ndarray, zero_qdot: np.ndarray
    ) -> None:
        """Coriolis terms vanish at zero velocity."""
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            C = analytical_coriolis(zero_q, zero_qdot, params)
        np.testing.assert_allclose(C, 0.0, atol=1e-8)

    def test_shape(
        self, params: GolferParams, zero_q: np.ndarray, zero_qdot: np.ndarray
    ) -> None:
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            C = analytical_coriolis(zero_q, zero_qdot, params)
        assert C.shape == (N_DOF,)

    def test_finite_with_nonzero_velocity(self, params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        q[0] = 0.2
        qdot = np.ones(N_DOF) * 0.1
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            C = analytical_coriolis(q, qdot, params)
        assert np.all(np.isfinite(C))

    def test_extended_state_truncation(self, params: GolferParams) -> None:
        q_ext = np.zeros(2 * N_DOF)
        qdot_ext = np.zeros(2 * N_DOF)
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            C_ext = analytical_coriolis(q_ext, qdot_ext, params)
            C_nrm = analytical_coriolis(np.zeros(N_DOF), np.zeros(N_DOF), params)
        np.testing.assert_allclose(C_ext, C_nrm, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests for kinetic_energy
# ---------------------------------------------------------------------------


class TestKineticEnergy:
    def test_zero_at_rest(
        self, params: GolferParams, zero_q: np.ndarray, zero_qdot: np.ndarray
    ) -> None:
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            T = kinetic_energy(zero_q, zero_qdot, params)
        assert T == pytest.approx(0.0, abs=1e-12)

    def test_positive_with_velocity(
        self, params: GolferParams, zero_q: np.ndarray
    ) -> None:
        qdot = np.ones(N_DOF) * 0.5
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            T = kinetic_energy(zero_q, qdot, params)
        assert T > 0

    def test_scales_quadratically_with_speed(
        self, params: GolferParams, zero_q: np.ndarray
    ) -> None:
        """Doubling velocity should quadruple kinetic energy."""
        qdot = np.ones(N_DOF) * 0.3
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            T1 = kinetic_energy(zero_q, qdot, params)
            T2 = kinetic_energy(zero_q, 2 * qdot, params)
        assert T2 == pytest.approx(4 * T1, rel=1e-6)

    def test_type_error_non_array_q(
        self, params: GolferParams, zero_qdot: np.ndarray
    ) -> None:
        with pytest.raises(TypeError):
            kinetic_energy([0.0] * N_DOF, zero_qdot, params)

    def test_value_error_wrong_shape_q(
        self, params: GolferParams, zero_qdot: np.ndarray
    ) -> None:
        with pytest.raises(ValueError):
            kinetic_energy(np.zeros(2), zero_qdot, params)


# ---------------------------------------------------------------------------
# Tests for potential_energy (takes full state vector)
# ---------------------------------------------------------------------------


class TestPotentialEnergy:
    def test_calls_potential_energy_from_q(
        self, params: GolferParams, full_state: np.ndarray
    ) -> None:
        V_state = potential_energy(full_state, params)
        V_q = potential_energy_from_q(full_state[:N_DOF], params)
        assert V_state == pytest.approx(V_q)

    def test_returns_float(self, params: GolferParams, full_state: np.ndarray) -> None:
        V = potential_energy(full_state, params)
        assert isinstance(V, float)

    def test_different_configurations_give_different_pe(
        self, params: GolferParams
    ) -> None:
        q1 = np.zeros(N_DOF)
        state1 = np.concatenate([q1, np.zeros(N_DOF)])
        q2 = np.zeros(N_DOF)
        q2[0] = np.pi / 4
        state2 = np.concatenate([q2, np.zeros(N_DOF)])
        V1 = potential_energy(state1, params)
        V2 = potential_energy(state2, params)
        assert V1 != pytest.approx(V2)


# ---------------------------------------------------------------------------
# Tests for total_energy
# ---------------------------------------------------------------------------


class TestTotalEnergy:
    def test_equals_T_plus_V_at_zero_velocity(
        self, params: GolferParams, full_state: np.ndarray
    ) -> None:
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            E = total_energy(full_state, params)
        # At zero velocity, KE = 0, so E should equal PE
        V = potential_energy(full_state, params)
        assert E == pytest.approx(V, abs=1e-10)

    def test_total_energy_finite(self, params: GolferParams) -> None:
        state = np.zeros(2 * N_DOF)
        state[0] = 0.3
        state[N_DOF] = 0.5
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            E = total_energy(state, params)
        assert np.isfinite(E)

    def test_total_is_T_plus_V(self, params: GolferParams) -> None:
        state = np.zeros(2 * N_DOF)
        state[0] = 0.2
        state[N_DOF + 1] = 0.4
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            T = kinetic_energy(state[:N_DOF], state[N_DOF:], params)
        V = potential_energy(state, params)
        with patch(
            "double_pendulum_golf.golfer_dynamics._native_backend.golfer_mass_matrix",
            return_value=None,
        ):
            E = total_energy(state, params)
        assert E == pytest.approx(T + V, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests for _mass_point_positions (additional physical properties)
# ---------------------------------------------------------------------------


class TestMassPointPositions:
    def test_returns_seven_points(
        self, params: GolferParams, zero_q: np.ndarray
    ) -> None:
        points = _mass_point_positions(zero_q, params)
        assert len(points) == 7

    def test_all_callable(self, params: GolferParams, zero_q: np.ndarray) -> None:
        points = _mass_point_positions(zero_q, params)
        for mass_val, pos_func in points:
            result = pos_func(zero_q)
            assert len(result) == 2

    def test_masses_match_params(
        self, params: GolferParams, zero_q: np.ndarray
    ) -> None:
        points = _mass_point_positions(zero_q, params)
        masses = [m for m, _ in points]
        assert params.m_hub in masses
        assert params.m_r_upper in masses
        assert params.m_club in masses

    def test_all_positions_finite(
        self, params: GolferParams, zero_q: np.ndarray
    ) -> None:
        points = _mass_point_positions(zero_q, params)
        for _, pos_func in points:
            x, y = pos_func(zero_q)
            assert np.isfinite(x) and np.isfinite(y)

    def test_extended_q_accepted(self, params: GolferParams) -> None:
        """Should accept arrays longer than N_DOF."""
        q_long = np.zeros(2 * N_DOF)
        points = _mass_point_positions(q_long, params)
        assert len(points) == 7
