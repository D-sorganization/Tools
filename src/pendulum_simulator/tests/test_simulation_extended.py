"""Extended tests for simulation.py (double pendulum) — covering untested methods.

Runs a short simulation and verifies all SimulationResult query methods:
theta1, phi, dtheta1, dphi, mass_matrix_at, positions_at, torques_at,
accelerations_at, joint_forces_at, joint_velocities_at, base_force_at,
control_vector_at, energy_at, coriolis_at, gravity_at, friction_torques_at,
total_torques_at
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from double_pendulum_golf.physics import PendulumParams
from double_pendulum_golf.simulation import SimulationResult, run_simulation


@pytest.fixture(scope="module")
def params() -> PendulumParams:
    return PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)


@pytest.fixture(scope="module")
def torque_func() -> Callable[[float], tuple[float, float]]:
    def zero_torque(t: float) -> tuple[float, float]:
        return (0.0, 0.0)

    return zero_torque


@pytest.fixture(scope="module")
def result(
    params: PendulumParams,
    torque_func: Callable,
) -> SimulationResult:
    """Run a short double-pendulum simulation (module-scoped for speed)."""
    initial_state = np.array([0.1, 0.05, 0.0, 0.0])
    return run_simulation(
        params,
        initial_state,
        t_end=0.05,
        torque_func=torque_func,
        dt=0.01,
    )


class TestRunSimulation:
    def test_returns_result(self, result: SimulationResult) -> None:
        assert isinstance(result, SimulationResult)

    def test_has_states(self, result: SimulationResult) -> None:
        assert len(result.states) > 0

    def test_states_are_4d(self, result: SimulationResult) -> None:
        for s in result.states:
            assert s.shape == (4,)

    def test_time_monotonic(self, result: SimulationResult) -> None:
        for i in range(len(result.t) - 1):
            assert result.t[i] < result.t[i + 1]


class TestSimulationResultProperties:
    def test_theta1(self, result: SimulationResult) -> None:
        arr = result.theta1
        assert len(arr) == len(result.t)
        assert np.all(np.isfinite(arr))

    def test_phi(self, result: SimulationResult) -> None:
        arr = result.phi
        assert len(arr) == len(result.t)
        assert np.all(np.isfinite(arr))

    def test_dtheta1(self, result: SimulationResult) -> None:
        arr = result.dtheta1
        assert len(arr) == len(result.t)

    def test_dphi(self, result: SimulationResult) -> None:
        arr = result.dphi
        assert len(arr) == len(result.t)


class TestSimulationResultMethods:
    def test_mass_matrix_at_shape(self, result: SimulationResult) -> None:
        mm = result.mass_matrix_at(0)
        assert isinstance(mm, dict)
        assert "M_full" in mm
        assert mm["M_full"].shape == (2, 2)

    def test_positions_at_keys(self, result: SimulationResult) -> None:
        pos = result.positions_at(0)
        assert isinstance(pos, dict)
        for key in ("shoulder", "wrist", "tip"):
            assert key in pos

    def test_torques_at_at_rest_zero(self, result: SimulationResult) -> None:
        tau = result.torques_at(0)
        assert len(tau) == 2
        assert all(t == 0.0 for t in tau)

    def test_accelerations_at_shape(self, result: SimulationResult) -> None:
        acc = result.accelerations_at(0)
        assert acc.shape == (2,)

    def test_accelerations_at_finite(self, result: SimulationResult) -> None:
        acc = result.accelerations_at(0)
        assert np.all(np.isfinite(acc))

    def test_joint_forces_at_keys(self, result: SimulationResult) -> None:
        jf = result.joint_forces_at(0)
        assert isinstance(jf, dict)
        assert "shoulder" in jf or len(jf) > 0

    def test_joint_forces_at_finite(self, result: SimulationResult) -> None:
        jf = result.joint_forces_at(0)
        for key, val in jf.items():
            for v in val:
                assert np.isfinite(v), f"{key} not finite"

    def test_joint_velocities_at_keys(self, result: SimulationResult) -> None:
        jv = result.joint_velocities_at(0)
        assert isinstance(jv, dict)
        assert "wrist_speed" in jv
        assert "tip_speed" in jv

    def test_base_force_at_keys(self, result: SimulationResult) -> None:
        bf = result.base_force_at(0)
        assert isinstance(bf, dict)
        assert "fx" in bf
        assert "fy" in bf
        assert "magnitude" in bf

    def test_base_force_magnitude_non_negative(self, result: SimulationResult) -> None:
        bf = result.base_force_at(0)
        assert bf["magnitude"] >= 0

    def test_control_vector_at(self, result: SimulationResult) -> None:
        cv = result.control_vector_at(0)
        assert isinstance(cv, dict)
        assert "cvx" in cv or "magnitude" in cv

    def test_energy_at(self, result: SimulationResult) -> None:
        e = result.energy_at(0)
        assert isinstance(e, dict)
        for key in ("kinetic", "potential", "total"):
            assert key in e
            assert np.isfinite(e[key])
        assert e["total"] == pytest.approx(e["kinetic"] + e["potential"], rel=1e-8)

    def test_coriolis_at_shape(self, result: SimulationResult) -> None:
        C = result.coriolis_at(0)
        assert C.shape == (2,)

    def test_coriolis_at_finite(self, result: SimulationResult) -> None:
        C = result.coriolis_at(0)
        assert np.all(np.isfinite(C))

    def test_gravity_at_shape(self, result: SimulationResult) -> None:
        G = result.gravity_at(0)
        assert G.shape == (2,)

    def test_gravity_at_finite(self, result: SimulationResult) -> None:
        G = result.gravity_at(0)
        assert np.all(np.isfinite(G))

    def test_friction_torques_at_shape(self, result: SimulationResult) -> None:
        tf = result.friction_torques_at(0)
        assert tf.shape == (2,)

    def test_friction_torques_zero_at_rest(self, result: SimulationResult) -> None:
        """Initial velocity is zero; friction torques should be zero."""
        tf = result.friction_torques_at(0)
        np.testing.assert_allclose(tf, 0.0, atol=1e-14)

    def test_total_torques_at_shape(self, result: SimulationResult) -> None:
        tt = result.total_torques_at(0)
        assert tt.shape == (2,)

    def test_last_index_valid(self, result: SimulationResult) -> None:
        last = len(result.t) - 1
        acc = result.accelerations_at(last)
        assert acc.shape == (2,)
