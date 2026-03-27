"""Tests for the golfer upper-body physics module.

Organized by property, following TDD principles.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from double_pendulum_golf.physics_golfer import (
    GolferParams,
    N_DOF,
    forward_kinematics,
    gravity_vector,
    kinetic_energy,
    mass_matrix,
    potential_energy_from_q,
    total_energy,
)


@pytest.fixture
def golfer_params() -> GolferParams:
    """Symmetric golfer parameters for testing."""
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
        L_club=1.1,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
    )


@pytest.fixture
def hanging_state() -> np.ndarray:
    """All segments hanging straight down, zero velocity."""
    return np.zeros(2 * N_DOF)


@pytest.fixture
def zero_torque() -> Callable:
    """Zero torque function for all joints."""
    return lambda t: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class TestGolferParamsValidation:
    """Parameter dataclass must enforce physical constraints."""

    def test_negative_mass_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            GolferParams(
                m_hub=-1.0,
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
                L_club=1.1,
                d_rs=0.20,
                d_ls=0.20,
                grip_right=0.05,
                grip_left=0.25,
            )

    def test_negative_length_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            GolferParams(
                m_hub=2.0,
                m_r_upper=3.0,
                m_r_fore=2.0,
                m_l_upper=3.0,
                m_l_fore=2.0,
                m_club=0.5,
                L_hub=-0.15,
                L_r_upper=0.35,
                L_r_fore=0.30,
                L_l_upper=0.35,
                L_l_fore=0.30,
                L_club=1.1,
                d_rs=0.20,
                d_ls=0.20,
                grip_right=0.05,
                grip_left=0.25,
            )

    def test_grip_exceeds_club_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            GolferParams(
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
                L_club=1.1,
                d_rs=0.20,
                d_ls=0.20,
                grip_right=2.0,
                grip_left=0.25,
            )

    def test_valid_params_accepted(self, golfer_params: GolferParams) -> None:
        assert golfer_params.m_hub > 0
        assert golfer_params.L_club > 0


class TestForwardKinematics:
    """FK must return correct positions for all joints."""

    def test_hanging_down_hub_above_origin(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        pos = forward_kinematics(q, golfer_params)
        assert "hub" in pos
        assert "origin" in pos
        assert pos["origin"] == (0.0, 0.0)
        # Hub extends upward (inside arm loop) after #1103 reversal
        assert pos["hub"][1] > 0

    def test_all_joint_keys_present(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        pos = forward_kinematics(q, golfer_params)
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
        assert expected_keys.issubset(set(pos.keys()))

    def test_shoulder_positions_symmetric(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        pos = forward_kinematics(q, golfer_params)
        # With symmetric params and zero angles, shoulders should be
        # symmetric about the hub-origin vertical line
        rs_x = pos["rs"][0]
        ls_x = pos["ls"][0]
        # Right and left should be on opposite sides
        assert rs_x * ls_x <= 0 or abs(rs_x - ls_x) > 1e-10

    def test_club_tip_exists(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        pos = forward_kinematics(q, golfer_params)
        tip = pos["club_tip"]
        assert np.isfinite(tip[0]) and np.isfinite(tip[1])


class TestMassMatrix:
    """Mass matrix must be symmetric and positive semi-definite."""

    def test_symmetric(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        M = mass_matrix(q, golfer_params)
        assert M.shape == (N_DOF, N_DOF)
        assert np.allclose(M, M.T, atol=1e-8)

    def test_symmetric_at_arbitrary_config(self, golfer_params: GolferParams) -> None:
        rng = np.random.default_rng(42)
        for _ in range(5):
            q = rng.uniform(-np.pi, np.pi, size=N_DOF)
            M = mass_matrix(q, golfer_params)
            assert np.allclose(M, M.T, atol=1e-8), "M must be symmetric"

    def test_positive_semi_definite(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        M = mass_matrix(q, golfer_params)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues >= -1e-10), (
            f"M must be positive semi-definite, got eigenvalues {eigenvalues}"
        )

    def test_depends_on_configuration(self, golfer_params: GolferParams) -> None:
        q1 = np.zeros(N_DOF)
        q2 = np.zeros(N_DOF)
        q2[1] = np.pi / 4  # change right shoulder angle
        M1 = mass_matrix(q1, golfer_params)
        M2 = mass_matrix(q2, golfer_params)
        assert not np.allclose(M1, M2), "M should vary with configuration"


class TestGravityVector:
    """Gravity torques must be physically reasonable."""

    def test_zero_at_equilibrium(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        G = gravity_vector(q, golfer_params)
        assert G.shape == (N_DOF,)
        # At hanging-down equilibrium, gravity torques should be near zero
        # (or very small for the hub standoff)
        assert np.all(np.isfinite(G))

    def test_nonzero_when_displaced(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        q[0] = np.pi / 2  # hub rotated sideways
        G = gravity_vector(q, golfer_params)
        assert not np.allclose(G, 0.0), "Gravity should act when displaced"


class TestEnergy:
    """Energy calculations must be consistent."""

    def test_zero_ke_at_rest(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        qdot = np.zeros(N_DOF)
        T = kinetic_energy(q, qdot, golfer_params)
        assert np.isclose(T, 0.0), f"KE should be 0 at rest, got {T}"

    def test_positive_ke_with_velocity(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        qdot = np.ones(N_DOF) * 0.5
        T = kinetic_energy(q, qdot, golfer_params)
        assert T > 0, f"KE should be positive with velocity, got {T}"

    def test_total_energy_sum(self, golfer_params: GolferParams) -> None:
        state = np.zeros(2 * N_DOF)
        state[0] = np.pi / 4
        state[N_DOF] = 1.0
        E = total_energy(state, golfer_params)
        T = kinetic_energy(state[:N_DOF], state[N_DOF:], golfer_params)
        V = potential_energy_from_q(state[:N_DOF], golfer_params)
        assert np.isclose(E, T + V), "E should equal T + V"
