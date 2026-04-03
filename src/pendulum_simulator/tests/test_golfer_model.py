"""TDD tests for golfer kinematics, dynamics, and constraints (#1221).

Design by Contract
------------------
- Forward kinematics positions must be bounded by segment lengths
- Constraint vector must be near zero for consistent configurations
- Analytical Jacobian must match numerical Jacobian
- Joint forces satisfy F = m*a
- Friction torque opposes velocity
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics_golfer import (
    GolferParams,
    N_DOF,
    forward_kinematics,
)
from double_pendulum_golf.golfer_constraints import (
    analytical_constraint_jacobian,
    constraint_vector,
    friction_torque_vector,
    net_joint_forces,
    numerical_constraint_jacobian,
)


@pytest.fixture()
def default_params() -> GolferParams:
    """Realistic golfer model parameters for testing."""
    return GolferParams(
        m_hub=0.01,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.3,
        L_hub=0.15,
        L_r_upper=0.30,
        L_r_fore=0.25,
        L_l_upper=0.30,
        L_l_fore=0.25,
        L_club=1.10,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.30,
    )


@pytest.fixture()
def zero_state() -> np.ndarray:
    """All joints at zero angle (hanging straight down)."""
    return np.zeros(N_DOF)


@pytest.fixture()
def random_state() -> np.ndarray:
    """Random but bounded joint angles."""
    rng = np.random.default_rng(42)
    q = rng.uniform(-0.5, 0.5, size=N_DOF)
    return q


# ---------------------------------------------------------------------------
# Forward Kinematics Tests
# ---------------------------------------------------------------------------


class TestGolferForwardKinematics:
    """Tests for golfer forward kinematics."""

    def test_fk_returns_all_joints(
        self, default_params: GolferParams, zero_state: np.ndarray
    ) -> None:
        """FK must return all expected joint positions."""
        fk = forward_kinematics(zero_state, default_params)
        expected_keys = {
            "origin",
            "hub",
            "rs",
            "re",
            "rh",
            "ls",
            "le",
            "lh",
            "club_base",
            "club_tip",
            "grip_right",
            "grip_left",
        }
        assert expected_keys.issubset(set(fk.keys()))

    def test_origin_is_zero(
        self, default_params: GolferParams, zero_state: np.ndarray
    ) -> None:
        """Origin must always be at (0, 0)."""
        fk = forward_kinematics(zero_state, default_params)
        assert fk["origin"] == (0.0, 0.0)

    def test_hub_bounded_by_L_hub(
        self, default_params: GolferParams, random_state: np.ndarray
    ) -> None:
        """Hub distance from origin must equal L_hub."""
        fk = forward_kinematics(random_state, default_params)
        hub = np.array(fk["hub"])
        dist = np.linalg.norm(hub)
        assert dist == pytest.approx(default_params.L_hub, rel=1e-10)

    def test_club_tip_finite(
        self, default_params: GolferParams, random_state: np.ndarray
    ) -> None:
        """Club tip must be at a finite position."""
        fk = forward_kinematics(random_state, default_params)
        tip = np.array(fk["club_tip"])
        assert np.all(np.isfinite(tip))

    def test_all_positions_finite(
        self, default_params: GolferParams, random_state: np.ndarray
    ) -> None:
        """All FK positions must be finite."""
        fk = forward_kinematics(random_state, default_params)
        for name, pos in fk.items():
            arr = np.array(pos)
            assert np.all(np.isfinite(arr)), f"Non-finite position at {name}"

    def test_elbow_distance_from_shoulder(
        self, default_params: GolferParams, random_state: np.ndarray
    ) -> None:
        """Right elbow must be L_r_upper from right shoulder."""
        fk = forward_kinematics(random_state, default_params)
        rs = np.array(fk["rs"])
        re = np.array(fk["re"])
        dist = np.linalg.norm(re - rs)
        assert dist == pytest.approx(default_params.L_r_upper, rel=1e-6)


# ---------------------------------------------------------------------------
# Constraint Tests
# ---------------------------------------------------------------------------


class TestGolferConstraints:
    """Tests for loop-closure constraint vector and Jacobian."""

    def test_constraint_size(
        self, default_params: GolferParams, zero_state: np.ndarray
    ) -> None:
        """Constraint vector must have 4 components."""
        phi = constraint_vector(zero_state, default_params)
        assert phi.shape == (4,)

    def test_jacobian_shape(
        self, default_params: GolferParams, zero_state: np.ndarray
    ) -> None:
        """Constraint Jacobian must be (4, N_DOF)."""
        J = numerical_constraint_jacobian(zero_state, default_params)
        assert J.shape == (4, N_DOF)

    def test_analytical_matches_numerical_jacobian(
        self, default_params: GolferParams, random_state: np.ndarray
    ) -> None:
        """Analytical and numerical Jacobians must agree."""
        J_num = numerical_constraint_jacobian(random_state, default_params)
        J_ana = analytical_constraint_jacobian(random_state, default_params)
        np.testing.assert_allclose(J_ana, J_num, atol=5e-2, rtol=0.1)

    def test_constraint_vector_finite(
        self, default_params: GolferParams, random_state: np.ndarray
    ) -> None:
        """Constraint vector must be finite for any valid configuration."""
        phi = constraint_vector(random_state, default_params)
        assert np.all(np.isfinite(phi))


# ---------------------------------------------------------------------------
# Joint Forces Tests
# ---------------------------------------------------------------------------


class TestGolferJointForces:
    """Tests for joint force and torque calculations."""

    def test_friction_torque_shape(self, default_params: GolferParams) -> None:
        """Friction torque vector must have N_DOF components."""
        qdot = np.ones(N_DOF) * 0.1
        tau = friction_torque_vector(qdot, default_params)
        assert tau.shape == (N_DOF,)

    def test_friction_opposes_velocity(self, default_params: GolferParams) -> None:
        """Friction torque must oppose joint velocity direction."""
        qdot = np.array([1.0, -1.0, 0.5, -0.3, 0.7, -0.2, 0.4, 0.0])
        tau = friction_torque_vector(qdot, default_params)
        # For each DOF with nonzero damping, sign(tau) = -sign(qdot)
        for i in range(N_DOF - 1):  # Skip club DOF (no damping)
            if abs(qdot[i]) > 0 and abs(tau[i]) > 0:
                assert np.sign(tau[i]) == -np.sign(
                    qdot[i]
                ), f"Friction at DOF {i} does not oppose velocity"

    def test_zero_velocity_zero_friction(self, default_params: GolferParams) -> None:
        """Zero velocity must produce zero friction torque."""
        qdot = np.zeros(N_DOF)
        tau = friction_torque_vector(qdot, default_params)
        np.testing.assert_allclose(tau, 0.0, atol=1e-15)

    def test_net_joint_forces_finite(
        self, default_params: GolferParams, zero_state: np.ndarray
    ) -> None:
        """Joint forces must be finite."""
        qdot = np.zeros(N_DOF)
        qddot = np.zeros(N_DOF)
        forces = net_joint_forces(zero_state, qdot, qddot, default_params)
        for name, f in forces.items():
            f_arr = np.array(f)
            assert np.all(np.isfinite(f_arr)), f"Non-finite force at {name}"

    def test_gravity_only_forces(
        self, default_params: GolferParams, zero_state: np.ndarray
    ) -> None:
        """At rest with zero acceleration, forces should equal -m*g."""
        qdot = np.zeros(N_DOF)
        qddot = np.zeros(N_DOF)
        forces = net_joint_forces(zero_state, qdot, qddot, default_params)
        # Hub has mass m_hub, force should be (0, m_hub * g)
        # because F = m*a - m*g_vec = 0 - m*(0, -g) = (0, m*g)
        hub_f = np.array(forces["hub"])
        expected_fy = default_params.m_hub * default_params.g
        assert hub_f[1] == pytest.approx(expected_fy, rel=0.1)


# ---------------------------------------------------------------------------
# Dynamics Tests (#1221, #1222)
# ---------------------------------------------------------------------------


class TestGolferDynamics:
    """Tests for mass matrix, energy, and gravity vector."""

    def test_mass_matrix_symmetric(
        self, default_params: GolferParams, random_state: np.ndarray
    ) -> None:
        """Mass matrix must be symmetric."""
        from double_pendulum_golf.golfer_dynamics import analytical_mass_matrix

        M = analytical_mass_matrix(random_state, default_params)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_mass_matrix_psd(
        self, default_params: GolferParams, random_state: np.ndarray
    ) -> None:
        """Mass matrix eigenvalues must be non-negative (PSD)."""
        from double_pendulum_golf.golfer_dynamics import analytical_mass_matrix

        M = analytical_mass_matrix(random_state, default_params)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(
            eigenvalues >= -1e-10
        ), f"Negative eigenvalue in mass matrix: {eigenvalues}"

    def test_mass_matrix_shape(
        self, default_params: GolferParams, zero_state: np.ndarray
    ) -> None:
        """Mass matrix must be (N_DOF, N_DOF)."""
        from double_pendulum_golf.golfer_dynamics import analytical_mass_matrix

        M = analytical_mass_matrix(zero_state, default_params)
        assert M.shape == (N_DOF, N_DOF)

    def test_gravity_vector_shape(
        self, default_params: GolferParams, zero_state: np.ndarray
    ) -> None:
        """Gravity vector must have N_DOF components."""
        from double_pendulum_golf.golfer_dynamics import analytical_gravity_vector

        G = analytical_gravity_vector(zero_state, default_params)
        assert G.shape == (N_DOF,)

    def test_gravity_vector_finite(
        self, default_params: GolferParams, random_state: np.ndarray
    ) -> None:
        """Gravity vector must be finite."""
        from double_pendulum_golf.golfer_dynamics import analytical_gravity_vector

        G = analytical_gravity_vector(random_state, default_params)
        assert np.all(np.isfinite(G))

    def test_kinetic_energy_non_negative(
        self, default_params: GolferParams, random_state: np.ndarray
    ) -> None:
        """Kinetic energy must be non-negative."""
        from double_pendulum_golf.golfer_dynamics import kinetic_energy

        qdot = np.random.default_rng(99).uniform(-2, 2, size=N_DOF)
        T = kinetic_energy(random_state, qdot, default_params)
        assert T >= 0.0

    def test_kinetic_energy_zero_at_rest(
        self, default_params: GolferParams, zero_state: np.ndarray
    ) -> None:
        """Kinetic energy must be zero when velocity is zero."""
        from double_pendulum_golf.golfer_dynamics import kinetic_energy

        qdot = np.zeros(N_DOF)
        T = kinetic_energy(zero_state, qdot, default_params)
        assert T == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# GolferParams DbC Validation (#1223)
# ---------------------------------------------------------------------------


class TestGolferParamsDbC:
    """Tests for GolferParams precondition assertions."""

    def _valid_kwargs(self) -> dict:
        """Return valid kwargs for GolferParams."""
        return dict(
            m_hub=0.01,
            m_r_upper=2.0,
            m_r_fore=1.5,
            m_l_upper=2.0,
            m_l_fore=1.5,
            m_club=0.3,
            L_hub=0.15,
            L_r_upper=0.30,
            L_r_fore=0.25,
            L_l_upper=0.30,
            L_l_fore=0.25,
            L_club=1.10,
            d_rs=0.20,
            d_ls=0.20,
            grip_right=0.05,
            grip_left=0.30,
        )

    def test_negative_mass_raises(self) -> None:
        """Negative mass must trigger assertion."""
        kw = self._valid_kwargs()
        kw["m_r_upper"] = -1.0
        with pytest.raises((ValueError, TypeError), match="positive"):
            GolferParams(**kw)

    def test_zero_mass_raises(self) -> None:
        """Zero mass must trigger assertion."""
        kw = self._valid_kwargs()
        kw["m_club"] = 0.0
        with pytest.raises((ValueError, TypeError), match="positive"):
            GolferParams(**kw)

    def test_negative_length_raises(self) -> None:
        """Negative segment length must trigger assertion."""
        kw = self._valid_kwargs()
        kw["L_r_upper"] = -0.1
        with pytest.raises((ValueError, TypeError), match="positive"):
            GolferParams(**kw)

    def test_grip_exceeds_club_raises(self) -> None:
        """Grip offset must not exceed club length."""
        kw = self._valid_kwargs()
        kw["grip_right"] = 2.0  # > L_club=1.10
        with pytest.raises((ValueError, TypeError), match="L_club"):
            GolferParams(**kw)

    def test_negative_damping_raises(self) -> None:
        """Negative damping must trigger assertion."""
        kw = self._valid_kwargs()
        kw["b_rs"] = -0.5
        with pytest.raises((ValueError, TypeError), match="non-negative"):
            GolferParams(**kw)

    def test_valid_params_accepted(self) -> None:
        """Valid parameters must not raise."""
        kw = self._valid_kwargs()
        p = GolferParams(**kw)
        assert p.m_club == 0.3
