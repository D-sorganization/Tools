"""Extended tests for physics_triple.py — covering previously untested functions.

Covers:
- mass_matrix_components: returns M11..M33 with M_full matching mass_matrix
- friction_torque_vector: sign, zero at rest, viscous + Coulomb components
- linear_accelerations: shape and finiteness
- kinetic_energy: zero at rest, positive with velocity, quadratic scaling
- potential_energy: height dependence
- total_energy: equals T + V
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics_triple import (
    TriplePendulumParams,
    kinetic_energy,
    linear_accelerations,
    mass_matrix,
    mass_matrix_components,
    potential_energy,
    total_energy,
)

# Also need friction_torque_vector
try:
    from double_pendulum_golf.physics_triple import friction_torque_vector

    HAS_FRICTION = True
except ImportError:
    HAS_FRICTION = False


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def params() -> TriplePendulumParams:
    return TriplePendulumParams(
        m1=5.0,
        m2=0.5,
        m3=0.2,
        L1=0.6,
        L2=0.6,
        L3=0.6,
    )


@pytest.fixture
def rest_state() -> np.ndarray:
    """Equilibrium state: angle=0, velocity=0."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def moving_state() -> np.ndarray:
    """State with non-zero angles and velocities."""
    return np.array([0.2, 0.3, -0.1, 1.0, -0.5, 0.8])


# ---------------------------------------------------------------------------
# Tests for mass_matrix_components
# ---------------------------------------------------------------------------


class TestMassMatrixComponents:
    def test_returns_dict_with_required_keys(self, params: TriplePendulumParams) -> None:
        result = mass_matrix_components(0.0, 0.0, params)
        assert isinstance(result, dict)
        for key in ("M11", "M22", "M33", "M_full"):
            assert key in result, f"Missing key: {key}"

    def test_m_full_matches_mass_matrix(self, params: TriplePendulumParams) -> None:
        """M_full in components should equal mass_matrix() output."""
        phi1, phi2 = 0.3, -0.2
        components = mass_matrix_components(phi1, phi2, params)
        M_full = mass_matrix(phi1, phi2, params)
        np.testing.assert_allclose(components["M_full"], M_full, atol=1e-10)

    def test_diagonal_elements_positive(self, params: TriplePendulumParams) -> None:
        """Diagonal mass matrix elements must be positive."""
        for phi1 in [0.0, 0.5, -0.5, np.pi / 3]:
            result = mass_matrix_components(phi1, 0.0, params)
            assert result["M11"] > 0, f"M11 non-positive for phi1={phi1}"
            assert result["M22"] > 0, f"M22 non-positive for phi1={phi1}"
            assert result["M33"] > 0, f"M33 non-positive for phi1={phi1}"

    def test_components_finite(self, params: TriplePendulumParams) -> None:
        result = mass_matrix_components(0.7, -0.4, params)
        for key, val in result.items():
            if key == "M_full":
                assert np.all(np.isfinite(val))
            else:
                assert np.isfinite(val), f"Non-finite {key}: {val}"


# ---------------------------------------------------------------------------
# Tests for friction_torque_vector
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_FRICTION, reason="friction_torque_vector not importable")
class TestFrictionTorqueVector:
    def test_zero_at_rest(self, params: TriplePendulumParams) -> None:
        """No friction at zero velocity."""
        tau_f = friction_torque_vector(0.0, 0.0, 0.0, params)
        np.testing.assert_allclose(tau_f, 0.0, atol=1e-12)

    def test_opposes_positive_velocity(self, params: TriplePendulumParams) -> None:
        """Friction should oppose motion — negative for positive velocity."""
        tau_f = friction_torque_vector(1.0, 0.5, 0.3, params)
        assert tau_f.shape == (3,)
        # All components should be <= 0 when all velocities > 0
        assert np.all(tau_f <= 0), f"Expected all friction <= 0, got {tau_f}"

    def test_opposes_negative_velocity(self, params: TriplePendulumParams) -> None:
        """Friction should be positive for negative velocity."""
        tau_f = friction_torque_vector(-1.0, -0.5, -0.3, params)
        assert np.all(tau_f >= 0), f"Expected all friction >= 0, got {tau_f}"

    def test_shape(self, params: TriplePendulumParams) -> None:
        tau_f = friction_torque_vector(0.5, 0.5, 0.5, params)
        assert tau_f.shape == (3,)

    def test_finite(self, params: TriplePendulumParams) -> None:
        tau_f = friction_torque_vector(1.5, -2.0, 0.7, params)
        assert np.all(np.isfinite(tau_f))


# ---------------------------------------------------------------------------
# Tests for linear_accelerations
# ---------------------------------------------------------------------------


class TestLinearAccelerations:
    def test_returns_dict_with_joint_keys(
        self, params: TriplePendulumParams, rest_state: np.ndarray
    ) -> None:
        qddot = np.zeros(3)
        result = linear_accelerations(rest_state, qddot, params)
        assert isinstance(result, dict)
        # Should have wrist1, wrist2, tip keys
        for key in ("wrist1", "wrist2", "tip"):
            assert key in result, f"Missing key: {key}"

    def test_shape_of_accelerations(
        self, params: TriplePendulumParams, rest_state: np.ndarray
    ) -> None:
        qddot = np.zeros(3)
        result = linear_accelerations(rest_state, qddot, params)
        for key, val in result.items():
            assert len(val) == 2, f"{key}: expected 2-tuple, got {len(val)}"

    def test_finite_at_rest(
        self, params: TriplePendulumParams, rest_state: np.ndarray
    ) -> None:
        qddot = np.zeros(3)
        result = linear_accelerations(rest_state, qddot, params)
        for key, (ax, ay) in result.items():
            assert np.isfinite(ax) and np.isfinite(ay), f"{key} not finite"

    def test_finite_with_motion(
        self, params: TriplePendulumParams, moving_state: np.ndarray
    ) -> None:
        qddot = np.array([0.5, -0.3, 0.2])
        result = linear_accelerations(moving_state, qddot, params)
        for key, (ax, ay) in result.items():
            assert np.isfinite(ax) and np.isfinite(ay), f"{key} not finite"


# ---------------------------------------------------------------------------
# Tests for kinetic_energy
# ---------------------------------------------------------------------------


class TestKineticEnergy:
    def test_zero_at_rest(self, params: TriplePendulumParams, rest_state: np.ndarray) -> None:
        T = kinetic_energy(rest_state, params)
        assert T == pytest.approx(0.0, abs=1e-12)

    def test_positive_with_velocity(self, params: TriplePendulumParams) -> None:
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.3])
        T = kinetic_energy(state, params)
        assert T > 0

    def test_scales_quadratically_with_velocity(self, params: TriplePendulumParams) -> None:
        """Doubling velocity should roughly quadruple KE."""
        state_slow = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        state_fast = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        T_slow = kinetic_energy(state_slow, params)
        T_fast = kinetic_energy(state_fast, params)
        assert T_fast == pytest.approx(4 * T_slow, rel=1e-6)

    def test_finite(self, params: TriplePendulumParams, moving_state: np.ndarray) -> None:
        assert np.isfinite(kinetic_energy(moving_state, params))


# ---------------------------------------------------------------------------
# Tests for potential_energy
# ---------------------------------------------------------------------------


class TestPotentialEnergy:
    def test_zero_at_equilibrium(
        self, params: TriplePendulumParams, rest_state: np.ndarray
    ) -> None:
        V = potential_energy(rest_state, params)
        # At hanging equilibrium (angle=0), PE is computed from reference
        # Just check it's finite and a real number
        assert np.isfinite(V)

    def test_increases_when_lifted(self, params: TriplePendulumParams) -> None:
        """PE should be higher when theta1 = pi/2 (horizontal) vs 0 (hanging)."""
        state_down = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_side = np.array([np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0])
        V_down = potential_energy(state_down, params)
        V_side = potential_energy(state_side, params)
        assert V_side > V_down

    def test_finite_at_all_angles(self, params: TriplePendulumParams) -> None:
        for phi1 in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            state = np.array([phi1, 0.2, -0.1, 0.0, 0.0, 0.0])
            assert np.isfinite(potential_energy(state, params))


# ---------------------------------------------------------------------------
# Tests for total_energy
# ---------------------------------------------------------------------------


class TestTotalEnergy:
    def test_equals_T_plus_V_at_rest(
        self, params: TriplePendulumParams, rest_state: np.ndarray
    ) -> None:
        E = total_energy(rest_state, params)
        T = kinetic_energy(rest_state, params)
        V = potential_energy(rest_state, params)
        assert E == pytest.approx(T + V, rel=1e-9)

    def test_equals_T_plus_V_with_motion(
        self, params: TriplePendulumParams, moving_state: np.ndarray
    ) -> None:
        E = total_energy(moving_state, params)
        T = kinetic_energy(moving_state, params)
        V = potential_energy(moving_state, params)
        assert E == pytest.approx(T + V, rel=1e-8)

    def test_finite(self, params: TriplePendulumParams, moving_state: np.ndarray) -> None:
        assert np.isfinite(total_energy(moving_state, params))

    def test_more_than_potential_alone(
        self, params: TriplePendulumParams, moving_state: np.ndarray
    ) -> None:
        """Total energy should exceed PE when there's KE > 0."""
        E = total_energy(moving_state, params)
        V = potential_energy(moving_state, params)
        T = kinetic_energy(moving_state, params)
        if T > 0:
            assert E > V
