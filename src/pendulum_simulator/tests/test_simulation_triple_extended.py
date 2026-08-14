"""Extended tests for simulation_triple.py — covering untested methods.

Runs a short simulation and verifies all TripleSimulationResult query methods.
Covers: mass_matrix_at, positions_at, torques_at, accelerations_at,
        joint_forces_at, coriolis_at, gravity_at, energy_at, friction_torques_at
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from double_pendulum_golf.physics_triple import TriplePendulumParams
from double_pendulum_golf.simulation_triple import (
    TripleSimulationResult,
    run_simulation,
)


@pytest.fixture(scope="module")
def params() -> TriplePendulumParams:
    return TriplePendulumParams(
        m1=5.0,
        m2=0.5,
        m3=0.2,
        L1=0.6,
        L2=0.6,
        L3=0.6,
    )


@pytest.fixture(scope="module")
def torque_func() -> Callable[[float], tuple[float, float, float]]:
    def zero_torque(t: float) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    return zero_torque


@pytest.fixture(scope="module")
def result(
    params: TriplePendulumParams,
    torque_func: Callable,
) -> TripleSimulationResult:
    """Run a short simulation and cache the result for all tests in the module."""
    initial_state = np.array([0.1, 0.05, -0.05, 0.0, 0.0, 0.0])
    return run_simulation(
        params, initial_state, t_end=0.1, torque_func=torque_func, dt=0.01
    )


class TestRunSimulation:
    def test_returns_result_object(self, result: TripleSimulationResult) -> None:
        assert isinstance(result, TripleSimulationResult)

    def test_has_states(self, result: TripleSimulationResult) -> None:
        assert hasattr(result, "states")
        assert len(result.states) > 0

    def test_has_time(self, result: TripleSimulationResult) -> None:
        assert hasattr(result, "t")
        assert len(result.t) > 0

    def test_states_shape(self, result: TripleSimulationResult) -> None:
        for s in result.states:
            assert s.shape == (6,)

    def test_time_monotonic(self, result: TripleSimulationResult) -> None:
        for i in range(len(result.t) - 1):
            assert result.t[i] < result.t[i + 1]

    def test_params_stored(
        self, result: TripleSimulationResult, params: TriplePendulumParams
    ) -> None:
        assert result.params is params


class TestTripleSimulationResultMethods:
    def test_mass_matrix_at_shape(self, result: TripleSimulationResult) -> None:
        mm = result.mass_matrix_at(0)
        assert isinstance(mm, dict)
        assert "M_full" in mm
        assert mm["M_full"].shape == (3, 3)

    def test_mass_matrix_at_finite(self, result: TripleSimulationResult) -> None:
        mm = result.mass_matrix_at(0)
        assert np.all(np.isfinite(mm["M_full"]))

    def test_positions_at_keys(self, result: TripleSimulationResult) -> None:
        pos = result.positions_at(0)
        assert isinstance(pos, dict)
        for key in ("hub", "shoulder", "wrist1", "wrist2", "tip"):
            assert key in pos, f"Missing key: {key}"

    def test_positions_at_finite(self, result: TripleSimulationResult) -> None:
        pos = result.positions_at(0)
        for key, val in pos.items():
            assert np.all(np.isfinite(val)), f"{key} not finite"

    def test_torques_at_returns_tuple(self, result: TripleSimulationResult) -> None:
        tau = result.torques_at(0)
        assert isinstance(tau, tuple)
        assert len(tau) == 3

    def test_torques_at_at_rest_zero(self, result: TripleSimulationResult) -> None:
        tau = result.torques_at(0)
        assert all(t == 0.0 for t in tau)

    def test_accelerations_at_shape(self, result: TripleSimulationResult) -> None:
        acc = result.accelerations_at(0)
        assert acc.shape == (3,)

    def test_accelerations_at_finite(self, result: TripleSimulationResult) -> None:
        acc = result.accelerations_at(0)
        assert np.all(np.isfinite(acc))

    def test_joint_forces_at_keys(self, result: TripleSimulationResult) -> None:
        forces = result.joint_forces_at(0)
        assert isinstance(forces, dict)
        # Triple pendulum has wrist1, wrist2, shoulder joints
        assert len(forces) > 0

    def test_joint_forces_at_finite(self, result: TripleSimulationResult) -> None:
        forces = result.joint_forces_at(0)
        for key, val in forces.items():
            for v in val:
                assert np.isfinite(v), f"{key} not finite"

    def test_coriolis_at_shape(self, result: TripleSimulationResult) -> None:
        C = result.coriolis_at(0)
        assert C.shape == (3,)

    def test_coriolis_at_finite(self, result: TripleSimulationResult) -> None:
        C = result.coriolis_at(0)
        assert np.all(np.isfinite(C))

    def test_gravity_at_shape(self, result: TripleSimulationResult) -> None:
        G = result.gravity_at(0)
        assert G.shape == (3,)

    def test_gravity_at_finite(self, result: TripleSimulationResult) -> None:
        G = result.gravity_at(0)
        assert np.all(np.isfinite(G))

    def test_energy_at_finite(self, result: TripleSimulationResult) -> None:
        e = result.energy_at(0)
        assert isinstance(e, dict)
        for key in ("kinetic", "potential", "total"):
            assert key in e
            assert np.isfinite(e[key])

    def test_energy_at_equals_T_plus_V(self, result: TripleSimulationResult) -> None:
        """Total energy should equal kinetic + potential."""
        e = result.energy_at(0)
        assert e["total"] == pytest.approx(e["kinetic"] + e["potential"], rel=1e-8)

    def test_friction_torques_at_shape(self, result: TripleSimulationResult) -> None:
        tau_f = result.friction_torques_at(0)
        assert tau_f.shape == (3,)

    def test_friction_torques_at_zero_when_no_damping(
        self, result: TripleSimulationResult
    ) -> None:
        """Default params have zero damping; friction torques should be zero at t=0."""
        # At t=0, velocities are zero → friction is zero
        tau_f = result.friction_torques_at(0)
        np.testing.assert_allclose(tau_f, 0.0, atol=1e-14)

    def test_last_index_valid(self, result: TripleSimulationResult) -> None:
        last = len(result.t) - 1
        pos = result.positions_at(last)
        assert isinstance(pos, dict)

    def test_out_of_bounds_raises(self, result: TripleSimulationResult) -> None:
        with pytest.raises((IndexError, ValueError)):
            result.mass_matrix_at(len(result.t) + 999)
