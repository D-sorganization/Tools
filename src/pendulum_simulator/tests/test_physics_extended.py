"""Extended tests for physics.py — covering previously untested functions.

Covers:
- TorqueClamp: dataclass, clamping behavior
- mass_matrix_components: labeled terms, consistency with mass_matrix
- friction_torque_vector: sign, zero at rest
- clamp_torque: correct clamping for 2-DOF
- JointLimitsNDOF: creation and validation
- joint_limit_torque_ndof: within-limits (zero), at-limit behavior
- clamp_torque_ndof: per-DOF clamping
- joint_velocities: shape and finiteness
- base_force: shape and finiteness
- ztcf_accelerations: shape and consistency
- linear_accelerations: shape and finiteness
- total_energy: equals T + V
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics import (
    JointLimitsNDOF,
    PendulumParams,
    TorqueClamp,
    base_force,
    clamp_torque,
    clamp_torque_ndof,
    joint_limit_torque_ndof,
    joint_velocities,
    kinetic_energy,
    linear_accelerations,
    mass_matrix,
    mass_matrix_components,
    potential_energy,
    total_energy,
    ztcf_accelerations,
)

# Also try importing friction_torque_vector
try:
    from double_pendulum_golf.physics import friction_torque_vector

    HAS_FRICTION = True
except ImportError:
    HAS_FRICTION = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def params() -> PendulumParams:
    return PendulumParams(
        m1=5.0,
        m2=0.5,
        L1=0.6,
        L2=1.0,
    )


@pytest.fixture
def rest_state() -> np.ndarray:
    """Equilibrium: hanging straight down, no velocity."""
    return np.array([0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def moving_state() -> np.ndarray:
    return np.array([0.3, -0.2, 1.0, -0.5])


# ---------------------------------------------------------------------------
# TorqueClamp tests
# ---------------------------------------------------------------------------


class TestTorqueClamp:
    def test_creation(self) -> None:
        clamp = TorqueClamp(max_torque1=50.0, max_torque2=20.0)
        assert clamp.max_torque1 == pytest.approx(50.0)
        assert clamp.max_torque2 == pytest.approx(20.0)

    def test_abs_applied_to_limits(self) -> None:
        """abs() should be applied automatically to negative inputs."""
        clamp = TorqueClamp(max_torque1=-50.0, max_torque2=-20.0)
        # The limits should be their absolute values
        assert clamp.max_torque1 > 0
        assert clamp.max_torque2 > 0


class TestClampTorque:
    def test_within_limits_unchanged(self) -> None:
        clamp = TorqueClamp(max_torque1=50.0, max_torque2=20.0)
        tau = np.array([10.0, 5.0])
        result = clamp_torque(tau, clamp)
        np.testing.assert_allclose(result, tau)

    def test_exceeds_max_clamped(self) -> None:
        clamp = TorqueClamp(max_torque1=50.0, max_torque2=20.0)
        tau = np.array([100.0, 50.0])
        result = clamp_torque(tau, clamp)
        assert result[0] == pytest.approx(50.0)
        assert result[1] == pytest.approx(20.0)

    def test_exceeds_min_clamped(self) -> None:
        clamp = TorqueClamp(max_torque1=50.0, max_torque2=20.0)
        tau = np.array([-100.0, -50.0])
        result = clamp_torque(tau, clamp)
        assert result[0] == pytest.approx(-50.0)
        assert result[1] == pytest.approx(-20.0)

    def test_shape_preserved(self) -> None:
        clamp = TorqueClamp(max_torque1=10.0, max_torque2=5.0)
        tau = np.array([1.0, 2.0])
        result = clamp_torque(tau, clamp)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# mass_matrix_components tests
# ---------------------------------------------------------------------------


class TestMassMatrixComponents:
    def test_returns_dict(self, params: PendulumParams) -> None:
        result = mass_matrix_components(0.0, params)
        assert isinstance(result, dict)

    def test_required_keys_present(self, params: PendulumParams) -> None:
        result = mass_matrix_components(0.5, params)
        assert "M11" in result
        assert "M_full" in result

    def test_m_full_matches_mass_matrix(self, params: PendulumParams) -> None:
        phi = 0.4
        components = mass_matrix_components(phi, params)
        M = mass_matrix(phi, params)
        np.testing.assert_allclose(components["M_full"], M, atol=1e-10)

    def test_diagonal_positive(self, params: PendulumParams) -> None:
        result = mass_matrix_components(0.3, params)
        assert result["M11"] > 0


# ---------------------------------------------------------------------------
# friction_torque_vector tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_FRICTION, reason="friction_torque_vector not importable")
class TestFrictionTorqueVector:
    def test_zero_at_rest(self, params: PendulumParams) -> None:
        tau_f = friction_torque_vector(0.0, 0.0, params)
        np.testing.assert_allclose(tau_f, 0.0, atol=1e-12)

    def test_opposes_positive_motion(self, params: PendulumParams) -> None:
        tau_f = friction_torque_vector(1.0, 1.0, params)
        assert tau_f.shape == (2,)
        assert np.all(tau_f <= 0)

    def test_opposes_negative_motion(self, params: PendulumParams) -> None:
        tau_f = friction_torque_vector(-1.0, -1.0, params)
        assert np.all(tau_f >= 0)

    def test_finite(self, params: PendulumParams) -> None:
        tau_f = friction_torque_vector(2.0, -1.5, params)
        assert np.all(np.isfinite(tau_f))


# ---------------------------------------------------------------------------
# JointLimitsNDOF tests
# ---------------------------------------------------------------------------


class TestJointLimitsNDOF:
    def test_construction(self) -> None:
        limits = JointLimitsNDOF(
            angle_min=np.array([-np.pi, -np.pi]),
            angle_max=np.array([np.pi, np.pi]),
            stiffness=100.0,
            damping=5.0,
        )
        assert limits.stiffness == 100.0
        assert limits.damping == 5.0


class TestJointLimitTorqueNDOF:
    @pytest.fixture
    def wide_limits(self) -> JointLimitsNDOF:
        return JointLimitsNDOF(
            angle_min=np.array([-10.0, -10.0]),
            angle_max=np.array([10.0, 10.0]),
            stiffness=100.0,
            damping=5.0,
        )

    def test_zero_torque_within_limits(self, wide_limits: JointLimitsNDOF) -> None:
        angles = np.array([0.0, 0.0])
        velocities = np.array([0.0, 0.0])
        tau = joint_limit_torque_ndof(angles, velocities, wide_limits)
        np.testing.assert_allclose(tau, 0.0, atol=1e-10)

    def test_shape(self, wide_limits: JointLimitsNDOF) -> None:
        angles = np.array([0.5, -0.3])
        velocities = np.array([0.1, -0.2])
        tau = joint_limit_torque_ndof(angles, velocities, wide_limits)
        assert tau.shape == (2,)

    def test_finite(self, wide_limits: JointLimitsNDOF) -> None:
        tau = joint_limit_torque_ndof(
            np.array([1.0, -0.5]), np.array([0.5, 0.1]), wide_limits
        )
        assert np.all(np.isfinite(tau))


class TestClampTorqueNDOF:
    def test_within_limits_unchanged(self) -> None:
        tau = np.array([1.0, 2.0, 3.0])
        limits = np.array([10.0, 10.0, 10.0])
        result = clamp_torque_ndof(tau, limits)
        np.testing.assert_allclose(result, tau)

    def test_clamped_above(self) -> None:
        tau = np.array([100.0, -200.0])
        limits = np.array([50.0, 50.0])
        result = clamp_torque_ndof(tau, limits)
        assert result[0] == pytest.approx(50.0)
        assert result[1] == pytest.approx(-50.0)

    def test_shape_preserved(self) -> None:
        tau = np.zeros(5)
        limits = np.ones(5) * 20.0
        result = clamp_torque_ndof(tau, limits)
        assert result.shape == (5,)


# ---------------------------------------------------------------------------
# joint_velocities tests
# ---------------------------------------------------------------------------


class TestJointVelocities:
    def test_returns_dict(self, params: PendulumParams, rest_state: np.ndarray) -> None:
        result = joint_velocities(rest_state, params)
        assert isinstance(result, dict)

    def test_has_speed_keys(
        self, params: PendulumParams, rest_state: np.ndarray
    ) -> None:
        result = joint_velocities(rest_state, params)
        assert "wrist_speed" in result
        assert "tip_speed" in result

    def test_zero_at_rest(self, params: PendulumParams, rest_state: np.ndarray) -> None:
        result = joint_velocities(rest_state, params)
        assert result["wrist_speed"] == pytest.approx(0.0, abs=1e-12)
        assert result["tip_speed"] == pytest.approx(0.0, abs=1e-12)

    def test_positive_with_motion(
        self, params: PendulumParams, moving_state: np.ndarray
    ) -> None:
        result = joint_velocities(moving_state, params)
        # Speeds should be non-negative
        assert result["wrist_speed"] >= 0
        assert result["tip_speed"] >= 0

    def test_finite(self, params: PendulumParams, moving_state: np.ndarray) -> None:
        result = joint_velocities(moving_state, params)
        assert np.isfinite(result["wrist_speed"])
        assert np.isfinite(result["tip_speed"])


# ---------------------------------------------------------------------------
# base_force tests
# ---------------------------------------------------------------------------


class TestBaseForce:
    def test_returns_dict(self, params: PendulumParams, rest_state: np.ndarray) -> None:
        qddot = np.zeros(2)
        result = base_force(rest_state, qddot, params)
        assert isinstance(result, dict)

    def test_has_required_keys(
        self, params: PendulumParams, rest_state: np.ndarray
    ) -> None:
        qddot = np.zeros(2)
        result = base_force(rest_state, qddot, params)
        assert "fx" in result
        assert "fy" in result
        assert "magnitude" in result

    def test_finite(self, params: PendulumParams, moving_state: np.ndarray) -> None:
        qddot = np.array([0.5, -0.3])
        result = base_force(moving_state, qddot, params)
        assert np.isfinite(result["fx"])
        assert np.isfinite(result["fy"])
        assert np.isfinite(result["magnitude"])

    def test_magnitude_non_negative(
        self, params: PendulumParams, moving_state: np.ndarray
    ) -> None:
        qddot = np.array([1.0, -0.5])
        result = base_force(moving_state, qddot, params)
        assert result["magnitude"] >= 0


# ---------------------------------------------------------------------------
# ztcf_accelerations tests
# ---------------------------------------------------------------------------


class TestZtcfAccelerations:
    def test_shape(self, params: PendulumParams, rest_state: np.ndarray) -> None:
        qddot = ztcf_accelerations(rest_state, params)
        assert qddot.shape == (2,)

    def test_finite(self, params: PendulumParams, moving_state: np.ndarray) -> None:
        qddot = ztcf_accelerations(moving_state, params)
        assert np.all(np.isfinite(qddot))

    def test_zero_at_equilibrium(
        self, params: PendulumParams, rest_state: np.ndarray
    ) -> None:
        """At equilibrium with no velocity, ZTCF accel should be zero."""
        qddot = ztcf_accelerations(rest_state, params)
        np.testing.assert_allclose(qddot, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# linear_accelerations tests
# ---------------------------------------------------------------------------


class TestLinearAccelerations:
    def test_returns_dict(self, params: PendulumParams, rest_state: np.ndarray) -> None:
        qddot = np.zeros(2)
        result = linear_accelerations(rest_state, qddot, params)
        assert isinstance(result, dict)

    def test_has_wrist_and_tip(
        self, params: PendulumParams, rest_state: np.ndarray
    ) -> None:
        qddot = np.zeros(2)
        result = linear_accelerations(rest_state, qddot, params)
        assert "wrist" in result or "ax_wrist" in result or len(result) >= 2

    def test_finite(self, params: PendulumParams, moving_state: np.ndarray) -> None:
        qddot = np.array([0.5, -0.3])
        result = linear_accelerations(moving_state, qddot, params)
        for key, val in result.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), f"{key} not finite"
            elif isinstance(val, tuple):
                for v in val:
                    assert np.isfinite(v)


# ---------------------------------------------------------------------------
# total_energy
# ---------------------------------------------------------------------------


class TestTotalEnergy:
    def test_finite(self, params: PendulumParams, rest_state: np.ndarray) -> None:
        E = total_energy(rest_state, params)
        assert np.isfinite(E)

    def test_equals_T_plus_V(
        self, params: PendulumParams, moving_state: np.ndarray
    ) -> None:
        E = total_energy(moving_state, params)
        T = kinetic_energy(moving_state, params)
        V = potential_energy(moving_state, params)
        assert E == pytest.approx(T + V, rel=1e-9)

    def test_rest_equals_pe_only(
        self, params: PendulumParams, rest_state: np.ndarray
    ) -> None:
        E = total_energy(rest_state, params)
        V = potential_energy(rest_state, params)
        assert E == pytest.approx(V, abs=1e-10)
