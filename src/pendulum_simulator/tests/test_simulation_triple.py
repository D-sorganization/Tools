"""Tests for the triple pendulum simulation module."""

import numpy as np
import pytest

from double_pendulum_golf.physics_triple import TriplePendulumParams, total_energy
from double_pendulum_golf.simulation_triple import (
    TripleSimulationResult,
    make_polynomial_torque,
    run_simulation,
)


@pytest.fixture
def triple_params():
    return TriplePendulumParams(m1=5.0, m2=0.5, m3=0.2, L1=0.6, L2=0.6, L3=0.6)


@pytest.fixture
def zero_torque():
    return lambda t: (0.0, 0.0, 0.0)


@pytest.fixture
def aligned_state():
    return np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.2])


class TestTriplePolynomialTorque:
    def test_constant_torque(self):
        func = make_polynomial_torque([5.0], [0.0], [1.0])
        assert func(0.0) == (5.0, 0.0, 1.0)
        assert func(2.0) == (5.0, 0.0, 1.0)

    def test_empty_coefficients_rejected(self):
        with pytest.raises(AssertionError):
            make_polynomial_torque([], [1.0], [0.0])


class TestTripleSimulationBasics:
    def test_produces_result(self, triple_params, zero_torque, aligned_state):
        result = run_simulation(
            triple_params, aligned_state, t_end=0.5, torque_func=zero_torque, dt=0.01
        )
        assert isinstance(result, TripleSimulationResult)
        assert result.n_steps > 10

    def test_initial_state_preserved(self, triple_params, zero_torque, aligned_state):
        result = run_simulation(
            triple_params, aligned_state, t_end=0.1, torque_func=zero_torque, dt=0.01
        )
        assert np.allclose(result.states[0], aligned_state, atol=1e-6)

    def test_time_monotonically_increases(self, triple_params, zero_torque, aligned_state):
        result = run_simulation(
            triple_params, aligned_state, t_end=1.0, torque_func=zero_torque, dt=0.01
        )
        assert all(np.diff(result.t) > 0)


class TestTripleEnergyConservation:
    def test_energy_conserved_free_pendulum(self, triple_params, zero_torque):
        state0 = np.array([np.radians(45), np.radians(30), np.radians(-15), 0.0, 0.0, 0.0])
        result = run_simulation(
            triple_params, state0, t_end=1.0, torque_func=zero_torque, dt=0.005
        )
        E0 = total_energy(result.states[0], triple_params)
        energies = np.array([total_energy(s, triple_params) for s in result.states])
        max_drift = np.max(np.abs(energies - E0))
        relative_drift = max_drift / abs(E0) if abs(E0) > 1e-10 else max_drift
        assert relative_drift < 1e-3
