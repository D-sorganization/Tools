# ruff: noqa: E501
"""
Tests for damping and Coulomb friction in the triple pendulum physics.

Following TDD / DbC pattern established in test_friction.py for the double pendulum.
Ensures parity between double and triple pendulum friction implementations.
"""

import numpy as np
import pytest

from double_pendulum_golf.physics_triple import (
    TriplePendulumParams,
    friction_torque_vector,
    mass_matrix,
    total_energy,
)
from double_pendulum_golf.simulation_triple import (
    TripleSimulationResult,
    make_polynomial_torque,
    run_simulation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_params() -> TriplePendulumParams:
    return TriplePendulumParams(m1=5.0, m2=2.0, m3=0.5, L1=0.5, L2=0.4, L3=0.3)


@pytest.fixture
def damped_params() -> TriplePendulumParams:
    """Params with moderate viscous damping at all three joints."""
    return TriplePendulumParams(
        m1=5.0,
        m2=2.0,
        m3=0.5,
        L1=0.5,
        L2=0.4,
        L3=0.3,
        b1=0.5,
        b2=0.3,
        b3=0.1,
    )


@pytest.fixture
def frictional_params() -> TriplePendulumParams:
    """Params with Coulomb friction only (no viscous damping)."""
    return TriplePendulumParams(
        m1=5.0,
        m2=2.0,
        m3=0.5,
        L1=0.5,
        L2=0.4,
        L3=0.3,
        mu1=0.1,
        mu2=0.05,
        mu3=0.02,
    )


@pytest.fixture
def combined_params() -> TriplePendulumParams:
    """Params with both viscous damping and Coulomb friction."""
    return TriplePendulumParams(
        m1=5.0,
        m2=2.0,
        m3=0.5,
        L1=0.5,
        L2=0.4,
        L3=0.3,
        b1=0.3,
        b2=0.2,
        b3=0.1,
        mu1=0.05,
        mu2=0.03,
        mu3=0.01,
    )


# ---------------------------------------------------------------------------
# TriplePendulumParams contract tests
# ---------------------------------------------------------------------------


class TestTripleParamsContracts:
    """Verify DbC constraints on damping/friction parameters."""

    def test_default_no_dissipation(self, base_params: TriplePendulumParams) -> None:
        assert base_params.b1 == 0.0
        assert base_params.b2 == 0.0
        assert base_params.b3 == 0.0
        assert base_params.mu1 == 0.0
        assert base_params.mu2 == 0.0
        assert base_params.mu3 == 0.0

    def test_valid_damping(self) -> None:
        p = TriplePendulumParams(
            m1=1.0,
            m2=1.0,
            m3=1.0,
            L1=1.0,
            L2=1.0,
            L3=1.0,
            b1=0.5,
            b2=1.0,
            b3=0.2,
        )
        assert p.b1 == 0.5
        assert p.b2 == 1.0
        assert p.b3 == 0.2

    def test_valid_coulomb(self) -> None:
        p = TriplePendulumParams(
            m1=1.0,
            m2=1.0,
            m3=1.0,
            L1=1.0,
            L2=1.0,
            L3=1.0,
            mu1=0.1,
            mu2=0.2,
            mu3=0.3,
        )
        assert p.mu1 == 0.1
        assert p.mu2 == 0.2
        assert p.mu3 == 0.3

    def test_negative_b1_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError), match="b1 must be non-negative"):
            TriplePendulumParams(
                m1=1.0,
                m2=1.0,
                m3=1.0,
                L1=1.0,
                L2=1.0,
                L3=1.0,
                b1=-0.1,
            )

    def test_negative_b2_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError), match="b2 must be non-negative"):
            TriplePendulumParams(
                m1=1.0,
                m2=1.0,
                m3=1.0,
                L1=1.0,
                L2=1.0,
                L3=1.0,
                b2=-0.1,
            )

    def test_negative_b3_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError), match="b3 must be non-negative"):
            TriplePendulumParams(
                m1=1.0,
                m2=1.0,
                m3=1.0,
                L1=1.0,
                L2=1.0,
                L3=1.0,
                b3=-0.1,
            )

    def test_negative_mu1_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError), match="mu1 must be non-negative"):
            TriplePendulumParams(
                m1=1.0,
                m2=1.0,
                m3=1.0,
                L1=1.0,
                L2=1.0,
                L3=1.0,
                mu1=-0.01,
            )

    def test_negative_mu2_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError), match="mu2 must be non-negative"):
            TriplePendulumParams(
                m1=1.0,
                m2=1.0,
                m3=1.0,
                L1=1.0,
                L2=1.0,
                L3=1.0,
                mu2=-0.01,
            )

    def test_negative_mu3_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError), match="mu3 must be non-negative"):
            TriplePendulumParams(
                m1=1.0,
                m2=1.0,
                m3=1.0,
                L1=1.0,
                L2=1.0,
                L3=1.0,
                mu3=-0.01,
            )


# ---------------------------------------------------------------------------
# friction_torque_vector tests
# ---------------------------------------------------------------------------


class TestTripleFrictionTorqueVector:
    """Unit tests for triple pendulum friction torque computation."""

    def test_zero_dissipation_gives_zero(
        self,
        base_params: TriplePendulumParams,
    ) -> None:
        tf = friction_torque_vector(
            dtheta1=2.0,
            dphi1=-1.0,
            dphi2=0.5,
            params=base_params,
        )
        assert tf.shape == (3,)
        assert np.allclose(tf, [0.0, 0.0, 0.0])

    def test_viscous_opposes_velocity(
        self,
        damped_params: TriplePendulumParams,
    ) -> None:
        """Viscous friction must oppose motion (same sign as -velocity)."""
        dtheta1, dphi1, dphi2 = 3.0, -2.0, 1.5
        tf = friction_torque_vector(dtheta1, dphi1, dphi2, damped_params)
        # Joint 1: positive velocity → negative friction
        assert tf[0] < 0.0, "Friction at joint 1 must oppose positive velocity"
        # Joint 2: negative velocity → positive friction
        assert tf[1] > 0.0, "Friction at joint 2 must oppose negative velocity"
        # Joint 3: positive velocity → negative friction
        assert tf[2] < 0.0, "Friction at joint 3 must oppose positive velocity"

    def test_viscous_magnitude_linear(
        self,
        damped_params: TriplePendulumParams,
    ) -> None:
        """Viscous torque magnitude = b * |qdot|."""
        dtheta1 = 2.0
        tf = friction_torque_vector(
            dtheta1=dtheta1,
            dphi1=0.0,
            dphi2=0.0,
            params=damped_params,
        )
        expected_tau_f1 = -damped_params.b1 * dtheta1
        assert np.isclose(tf[0], expected_tau_f1)

    def test_coulomb_has_constant_magnitude(
        self,
        frictional_params: TriplePendulumParams,
    ) -> None:
        """Coulomb friction magnitude is mu regardless of velocity magnitude."""
        for speed in [0.1, 1.0, 10.0, 100.0]:
            tf = friction_torque_vector(
                dtheta1=speed,
                dphi1=speed,
                dphi2=speed,
                params=frictional_params,
            )
            assert np.isclose(
                abs(tf[0]),
                frictional_params.mu1,
            ), f"Expected |tau_f1|={frictional_params.mu1}, got {abs(tf[0])} at speed={speed}"
            assert np.isclose(
                abs(tf[2]),
                frictional_params.mu3,
            ), f"Expected |tau_f3|={frictional_params.mu3}, got {abs(tf[2])} at speed={speed}"

    def test_coulomb_zero_at_rest(
        self,
        frictional_params: TriplePendulumParams,
    ) -> None:
        """np.sign(0) == 0, so Coulomb friction is zero when stationary."""
        tf = friction_torque_vector(
            dtheta1=0.0,
            dphi1=0.0,
            dphi2=0.0,
            params=frictional_params,
        )
        assert np.allclose(tf, [0.0, 0.0, 0.0])

    def test_combined_friction_superposition(
        self,
        combined_params: TriplePendulumParams,
    ) -> None:
        """Combined damping+friction = viscous + Coulomb separately."""
        dtheta1, dphi1, dphi2 = 1.5, -0.8, 0.3
        tf = friction_torque_vector(dtheta1, dphi1, dphi2, combined_params)

        expected_1 = -combined_params.b1 * dtheta1 - combined_params.mu1 * np.sign(dtheta1)
        expected_2 = -combined_params.b2 * dphi1 - combined_params.mu2 * np.sign(dphi1)
        expected_3 = -combined_params.b3 * dphi2 - combined_params.mu3 * np.sign(dphi2)
        assert np.isclose(tf[0], expected_1)
        assert np.isclose(tf[1], expected_2)
        assert np.isclose(tf[2], expected_3)

    def test_output_shape_and_finiteness(
        self,
        combined_params: TriplePendulumParams,
    ) -> None:
        tf = friction_torque_vector(
            dtheta1=5.0,
            dphi1=3.0,
            dphi2=-1.0,
            params=combined_params,
        )
        assert tf.shape == (3,)
        assert all(np.isfinite(tf))


# ---------------------------------------------------------------------------
# EOM integration tests with dissipation
# ---------------------------------------------------------------------------


class TestTripleEOMWithDissipation:
    """Verify that EOM correctly incorporates friction into dynamics."""

    def test_undamped_conserves_energy_approximately(
        self,
        base_params: TriplePendulumParams,
    ) -> None:
        """Without dissipation, total energy should be nearly conserved."""
        from double_pendulum_golf.physics_triple import total_energy

        state0 = np.array([np.pi / 6, 0.0, 0.0, 0.0, 0.0, 0.0])
        torque_func = make_polynomial_torque([0.0], [0.0], [0.0])

        result = run_simulation(
            params=base_params,
            initial_state=state0,
            t_end=2.0,
            torque_func=torque_func,
            dt=0.005,
        )

        e_start = total_energy(result.states[0], base_params)
        e_end = total_energy(result.states[-1], base_params)
        # Allow ~2% drift for chaotic triple pendulum
        assert abs(e_end - e_start) / max(abs(e_start), 1e-9) < 0.02, (
            f"Energy drift too large: {e_start:.4f} → {e_end:.4f}"
        )

    def test_damped_pendulum_loses_energy(
        self,
        damped_params: TriplePendulumParams,
    ) -> None:
        """With viscous damping, total energy must decrease over time."""
        from double_pendulum_golf.physics_triple import total_energy

        state0 = np.array([np.pi / 4, 0.0, 0.0, 0.0, 0.0, 0.0])
        torque_func = make_polynomial_torque([0.0], [0.0], [0.0])

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

    def test_friction_does_not_blow_up(
        self,
        combined_params: TriplePendulumParams,
    ) -> None:
        """Simulation with both friction types must remain numerically stable."""
        state0 = np.array([np.radians(90), np.radians(-45), np.radians(30), 0.0, 0.0, 0.0])
        torque_func = make_polynomial_torque([-15.0, 5.0], [0.0], [0.0])

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
# TripleSimulationResult friction accessor tests
# ---------------------------------------------------------------------------


class TestTripleSimulationResultFrictionAccessors:
    """Verify friction_torques_at and total_torques_at methods."""

    @pytest.fixture
    def friction_result(
        self,
        combined_params: TriplePendulumParams,
    ) -> TripleSimulationResult:
        state0 = np.array([np.radians(60), 0.0, 0.0, 0.5, 0.0, 0.0])
        torque_func = make_polynomial_torque([5.0], [0.0], [0.0])
        return run_simulation(
            params=combined_params,
            initial_state=state0,
            t_end=1.0,
            torque_func=torque_func,
            dt=0.005,
        )

    def test_friction_torques_shape(
        self,
        friction_result: TripleSimulationResult,
    ) -> None:
        tf = friction_result.friction_torques_at(0)
        assert tf.shape == (3,)

    def test_total_torques_equals_drive_plus_friction(
        self,
        friction_result: TripleSimulationResult,
    ) -> None:
        idx = friction_result.n_steps // 2
        drive = np.array(friction_result.torques_at(idx))
        friction = friction_result.friction_torques_at(idx)
        total = friction_result.total_torques_at(idx)
        assert np.allclose(total, drive + friction)

    def test_no_dissipation_zero_friction_torques(self) -> None:
        base = TriplePendulumParams(
            m1=5.0,
            m2=2.0,
            m3=0.5,
            L1=0.5,
            L2=0.4,
            L3=0.3,
        )
        state0 = np.array([np.radians(30), 0.0, 0.0, 1.0, 0.0, 0.0])
        result = run_simulation(
            params=base,
            initial_state=state0,
            t_end=0.5,
            torque_func=make_polynomial_torque([0.0], [0.0], [0.0]),
            dt=0.005,
        )
        for i in range(0, result.n_steps, 20):
            tf = result.friction_torques_at(i)
            assert np.allclose(
                tf,
                [0.0, 0.0, 0.0],
            ), f"Expected zero friction torques at step {i}, got {tf}"


# ---------------------------------------------------------------------------
# Mass matrix correctness tests
# ---------------------------------------------------------------------------


class TestMassMatrixCorrectness:
    """Verify mass matrix structure and known identities."""

    @pytest.fixture
    def equal_params(self) -> TriplePendulumParams:
        return TriplePendulumParams(m1=1.0, m2=1.0, m3=1.0, L1=1.0, L2=1.0, L3=1.0)

    def test_symmetric_at_random_angles(self, equal_params: TriplePendulumParams) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            phi1, phi2 = rng.uniform(-np.pi, np.pi, size=2)
            M = mass_matrix(phi1, phi2, equal_params)
            assert np.allclose(M, M.T), f"Not symmetric at phi1={phi1}, phi2={phi2}"

    def test_positive_definite_at_random_angles(
        self, equal_params: TriplePendulumParams
    ) -> None:
        rng = np.random.default_rng(123)
        for _ in range(20):
            phi1, phi2 = rng.uniform(-np.pi, np.pi, size=2)
            M = mass_matrix(phi1, phi2, equal_params)
            eigvals = np.linalg.eigvalsh(M)
            assert all(eigvals > 0), f"Not positive definite at phi1={phi1}, phi2={phi2}"

    def test_aligned_configuration_known_value(self) -> None:
        """When phi1=phi2=0 (all segments aligned), M has a known closed form."""
        p = TriplePendulumParams(m1=1.0, m2=1.0, m3=1.0, L1=1.0, L2=1.0, L3=1.0)
        M = mass_matrix(0.0, 0.0, p)
        # M11 = (1+1+1)*1 + (1+1)*1 + 1*1 + 2*(1+1)*1 + 2*1*1 + 2*1*1 = 3+2+1+4+2+2 = 14
        assert np.isclose(M[0, 0], 14.0), f"M11 at aligned should be 14, got {M[0, 0]}"
        # M12 = (1+1)*1 + 1*1 + (1+1)*1 + 1*1 + 2*1*1 = 2+1+2+1+2 = 8
        assert np.isclose(M[0, 1], 8.0), f"M12 at aligned should be 8, got {M[0, 1]}"
        # M13 = 1*1 + 1*1 + 1*1 = 3
        assert np.isclose(M[0, 2], 3.0), f"M13 at aligned should be 3, got {M[0, 2]}"
        # M22 = (1+1)*1 + 1*1 + 2*1*1 = 2+1+2 = 5
        assert np.isclose(M[1, 1], 5.0), f"M22 at aligned should be 5, got {M[1, 1]}"
        # M23 = 1*1 + 1*1 = 2
        assert np.isclose(M[1, 2], 2.0), f"M23 at aligned should be 2, got {M[1, 2]}"
        # M33 = 1*1 = 1
        assert np.isclose(M[2, 2], 1.0), f"M33 at aligned should be 1, got {M[2, 2]}"

    def test_perpendicular_configuration(self) -> None:
        """phi1=pi/2, phi2=0: cos(phi1)=0, cos(phi2)=1, cos(phi1+phi2)=0."""
        p = TriplePendulumParams(m1=1.0, m2=1.0, m3=1.0, L1=1.0, L2=1.0, L3=1.0)
        M = mass_matrix(np.pi / 2, 0.0, p)
        # c1=0, c2=1, c12=0
        # M11 = 3+2+1+0+0+2 = 8
        assert np.isclose(M[0, 0], 8.0), f"M11 expected 8, got {M[0, 0]}"
        # M12 = 2+1+0+0+2 = 5
        assert np.isclose(M[0, 1], 5.0), f"M12 expected 5, got {M[0, 1]}"
        # M13 = 1+0+1 = 2
        assert np.isclose(M[0, 2], 2.0), f"M13 expected 2, got {M[0, 2]}"


# ---------------------------------------------------------------------------
# Energy conservation with corrected mass matrix + Coriolis
# ---------------------------------------------------------------------------


class TestEnergyConservation:
    """Verify M and C consistency via energy conservation in conservative systems."""

    @pytest.mark.parametrize(
        "state0",
        [
            np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.5, -0.3, 0.0, 0.0, 0.0]),
            np.array([0.2, -0.4, 0.3, 1.0, -0.5, 0.3]),
            np.array([np.pi / 3, np.pi / 6, -np.pi / 4, 0.5, -0.3, 0.2]),
        ],
        ids=["tilt-only", "relative-only", "mixed-slow", "mixed-fast"],
    )
    def test_conservative_energy_conservation(self, state0: np.ndarray) -> None:
        """Without friction or external torques, E = T + V must be conserved."""
        params = TriplePendulumParams(m1=3.0, m2=2.0, m3=1.0, L1=0.5, L2=0.4, L3=0.3)
        torque_func = make_polynomial_torque([0.0], [0.0], [0.0])
        result = run_simulation(
            params=params,
            initial_state=state0,
            t_end=2.0,
            torque_func=torque_func,
            dt=0.002,
            rtol=1e-10,
            atol=1e-12,
        )
        energies = [total_energy(result.states[i], params) for i in range(result.n_steps)]
        e0 = energies[0]
        max_drift = max(abs(e - e0) for e in energies)
        assert max_drift < 1e-6, (
            f"Energy drift {max_drift:.2e} exceeds 1e-6 for state0={state0}"
        )
