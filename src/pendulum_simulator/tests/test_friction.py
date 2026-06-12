# ruff: noqa: E501
"""
Tests for damping and Coulomb friction in the double pendulum physics.

Following TDD / DbC pattern established in the codebase.
"""

import numpy as np
import pytest

from double_pendulum_golf.physics import (
    PendulumParams,
    friction_torque_vector,
)
from double_pendulum_golf.simulation import (
    SimulationResult,
    make_polynomial_torque,
    run_simulation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_params() -> PendulumParams:
    return PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)


@pytest.fixture
def damped_params() -> PendulumParams:
    """Params with moderate viscous damping at both joints."""
    return PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0, b1=0.5, b2=0.2)


@pytest.fixture
def frictional_params() -> PendulumParams:
    """Params with Coulomb friction only (no viscous damping)."""
    return PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0, mu1=0.1, mu2=0.05)


@pytest.fixture
def combined_params() -> PendulumParams:
    """Params with both viscous damping and Coulomb friction."""
    return PendulumParams(
        m1=5.0,
        m2=0.5,
        L1=0.6,
        L2=1.0,
        b1=0.3,
        b2=0.1,
        mu1=0.05,
        mu2=0.02,
    )


# ---------------------------------------------------------------------------
# PendulumParams contract tests
# ---------------------------------------------------------------------------


class TestPendulumParamsContracts:
    """Verify new DbC constraints on damping/friction parameters."""

    def test_default_no_dissipation(self, base_params: PendulumParams) -> None:
        assert base_params.b1 == 0.0
        assert base_params.b2 == 0.0
        assert base_params.mu1 == 0.0
        assert base_params.mu2 == 0.0

    def test_valid_damping(self) -> None:
        p = PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, b1=0.5, b2=1.0)
        assert p.b1 == 0.5
        assert p.b2 == 1.0

    def test_valid_coulomb(self) -> None:
        p = PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, mu1=0.1, mu2=0.2)
        assert p.mu1 == 0.1
        assert p.mu2 == 0.2

    def test_negative_b1_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError), match="b1 must be non-negative"):
            PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, b1=-0.1)

    def test_negative_b2_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError), match="b2 must be non-negative"):
            PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, b2=-0.1)

    def test_negative_mu1_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError), match="mu1 must be non-negative"):
            PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, mu1=-0.01)

    def test_negative_mu2_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError), match="mu2 must be non-negative"):
            PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, mu2=-0.01)


# ---------------------------------------------------------------------------
# friction_torque_vector tests
# ---------------------------------------------------------------------------


class TestFrictionTorqueVector:
    """Unit tests for the friction torque computation."""

    def test_zero_dissipation_gives_zero(self, base_params: PendulumParams) -> None:
        tf = friction_torque_vector(dtheta1=2.0, dphi=-1.0, params=base_params)
        assert tf.shape == (2,)
        assert np.allclose(tf, [0.0, 0.0])

    def test_viscous_opposes_velocity(self, damped_params: PendulumParams) -> None:
        """Viscous friction must oppose motion (same sign as -velocity)."""
        dtheta1, dphi = 3.0, -2.0
        tf = friction_torque_vector(dtheta1, dphi, damped_params)
        # Joint 1: positive velocity → negative friction
        assert tf[0] < 0.0, "Friction at joint 1 must oppose positive velocity"
        # Joint 2: negative velocity → positive friction
        assert tf[1] > 0.0, "Friction at joint 2 must oppose negative velocity"

    def test_viscous_magnitude_linear(self, damped_params: PendulumParams) -> None:
        """Viscous torque magnitude = b * |qdot|."""
        dtheta1 = 2.0
        tf = friction_torque_vector(dtheta1=dtheta1, dphi=0.0, params=damped_params)
        expected_tau_f1 = -damped_params.b1 * dtheta1
        assert np.isclose(tf[0], expected_tau_f1)

    def test_coulomb_has_constant_magnitude(self, frictional_params: PendulumParams) -> None:
        """Coulomb friction magnitude is mu regardless of velocity magnitude."""
        for speed in [0.1, 1.0, 10.0, 100.0]:
            tf = friction_torque_vector(dtheta1=speed, dphi=speed, params=frictional_params)
            assert np.isclose(abs(tf[0]), frictional_params.mu1), (
                f"Expected |tau_f1|={frictional_params.mu1}, got {abs(tf[0])} at speed={speed}"
            )

    def test_coulomb_zero_at_rest(self, frictional_params: PendulumParams) -> None:
        """np.sign(0) == 0, so Coulomb friction is zero when stationary."""
        tf = friction_torque_vector(dtheta1=0.0, dphi=0.0, params=frictional_params)
        assert np.allclose(tf, [0.0, 0.0])

    def test_combined_friction_superposition(self, combined_params: PendulumParams) -> None:
        """Combined damping+friction = viscous + Coulomb separately."""
        dtheta1, dphi = 1.5, -0.8
        tf = friction_torque_vector(dtheta1, dphi, combined_params)

        expected_1 = -combined_params.b1 * dtheta1 - combined_params.mu1 * np.sign(dtheta1)
        expected_2 = -combined_params.b2 * dphi - combined_params.mu2 * np.sign(dphi)
        assert np.isclose(tf[0], expected_1)
        assert np.isclose(tf[1], expected_2)

    def test_output_shape_and_finiteness(self, combined_params: PendulumParams) -> None:
        tf = friction_torque_vector(dtheta1=5.0, dphi=3.0, params=combined_params)
        assert tf.shape == (2,)
        assert all(np.isfinite(tf))


# ---------------------------------------------------------------------------
# EOM integration tests with dissipation
# ---------------------------------------------------------------------------


class TestEquationsOfMotionWithDissipation:
    """Verify that EOM correctly incorporates friction into dynamics."""

    def test_undamped_conserves_energy_approximately(
        self, base_params: PendulumParams
    ) -> None:
        """Without dissipation, total energy should be nearly conserved."""
        from double_pendulum_golf.physics import total_energy

        state0 = np.array([np.pi / 4, 0.0, 0.0, 0.0])
        torque_func = make_polynomial_torque([0.0], [0.0])

        result = run_simulation(
            params=base_params,
            initial_state=state0,
            t_end=2.0,
            torque_func=torque_func,
            dt=0.001,
        )

        e_start = total_energy(result.states[0], base_params)
        e_end = total_energy(result.states[-1], base_params)
        # Allow ~1% drift from numerical integration
        assert abs(e_end - e_start) / max(abs(e_start), 1e-9) < 0.01, (
            f"Energy drift too large: {e_start:.4f} → {e_end:.4f}"
        )

    def test_damped_pendulum_loses_energy(self, damped_params: PendulumParams) -> None:
        """With viscous damping, total energy must decrease over time."""
        from double_pendulum_golf.physics import total_energy

        state0 = np.array([np.pi / 4, 0.0, 0.0, 0.0])
        torque_func = make_polynomial_torque([0.0], [0.0])

        result = run_simulation(
            params=damped_params,
            initial_state=state0,
            t_end=3.0,
            torque_func=torque_func,
            dt=0.005,
        )

        e_start = total_energy(result.states[0], damped_params)
        e_end = total_energy(result.states[-1], damped_params)
        assert e_end < e_start, (
            f"Damped pendulum energy should decrease: {e_start:.4f} → {e_end:.4f}"
        )

    def test_friction_does_not_blow_up(self, combined_params: PendulumParams) -> None:
        """Simulation with both friction types must remain numerically stable."""
        state0 = np.array([np.radians(120), np.radians(-90), 0.0, 0.0])
        torque_func = make_polynomial_torque([-25.0, 10.0], [0.0])

        result = run_simulation(
            params=combined_params,
            initial_state=state0,
            t_end=2.0,
            torque_func=torque_func,
            dt=0.005,
        )

        assert result.n_steps >= 2
        assert all(np.isfinite(result.states.flatten())), (
            "Simulation with combined friction/damping produced non-finite states"
        )


# ---------------------------------------------------------------------------
# SimulationResult friction accessor tests
# ---------------------------------------------------------------------------


class TestSimulationResultFrictionAccessors:
    """Verify friction_torques_at and total_torques_at methods."""

    @pytest.fixture
    def friction_result(self, combined_params: PendulumParams) -> SimulationResult:
        state0 = np.array([np.radians(90), 0.0, 0.5, 0.0])
        torque_func = make_polynomial_torque([5.0], [0.0])
        return run_simulation(
            params=combined_params,
            initial_state=state0,
            t_end=1.0,
            torque_func=torque_func,
            dt=0.005,
        )

    def test_friction_torques_shape(self, friction_result: SimulationResult) -> None:
        tf = friction_result.friction_torques_at(0)
        assert tf.shape == (2,)

    def test_total_torques_equals_drive_plus_friction(
        self, friction_result: SimulationResult
    ) -> None:
        idx = friction_result.n_steps // 2
        drive = np.array(friction_result.torques_at(idx))
        friction = friction_result.friction_torques_at(idx)
        total = friction_result.total_torques_at(idx)
        assert np.allclose(total, drive + friction)

    def test_no_dissipation_zero_friction_torques(self, base_params: PendulumParams) -> None:
        state0 = np.array([np.radians(45), 0.0, 1.0, 0.0])
        result = run_simulation(
            params=base_params,
            initial_state=state0,
            t_end=0.5,
            torque_func=make_polynomial_torque([0.0], [0.0]),
            dt=0.005,
        )
        for i in range(0, result.n_steps, 20):
            tf = result.friction_torques_at(i)
            assert np.allclose(tf, [0.0, 0.0]), (
                f"Expected zero friction torques at step {i}, got {tf}"
            )
