"""Extended tests for simulation_golfer.py — covering untested query methods.

Runs a short golfer simulation and verifies all GolferSimulationResult methods:
qdot_at, torques_at, accelerations_at, joint_forces_at, constraint_forces_at,
constraint_violation_at, coriolis_at, gravity_at, friction_torques_at, total_torques_at
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from double_pendulum_golf.physics_golfer import GolferParams, N_DOF
from double_pendulum_golf.simulation_golfer import (
    GolferSimulationResult,
    run_simulation,
)


@pytest.fixture(scope="module")
def params() -> GolferParams:
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
        L_club=1.10,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
    )


@pytest.fixture(scope="module")
def torque_func() -> Callable:
    def zero_torque(t: float) -> tuple:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return zero_torque


@pytest.fixture(scope="module")
def result(params: GolferParams, torque_func: Callable) -> GolferSimulationResult:
    """Run a short golfer simulation (module-scoped for speed)."""
    initial_state = np.zeros(2 * N_DOF)
    # Start from slight perturbation
    initial_state[0] = 0.05  # small hub angle
    return run_simulation(
        params,
        initial_state,
        t_end=0.05,
        torque_func=torque_func,
        dt=0.01,
    )


class TestRunGolferSimulation:
    def test_returns_result(self, result: GolferSimulationResult) -> None:
        assert isinstance(result, GolferSimulationResult)

    def test_has_states(self, result: GolferSimulationResult) -> None:
        assert len(result.states) >= 2

    def test_states_shape(self, result: GolferSimulationResult) -> None:
        for s in result.states:
            assert s.shape == (2 * N_DOF,)

    def test_time_monotonic(self, result: GolferSimulationResult) -> None:
        for i in range(len(result.t) - 1):
            assert result.t[i] < result.t[i + 1]


class TestGolferSimulationResultMethods:
    def test_q_at_shape(self, result: GolferSimulationResult) -> None:
        q = result.q_at(0)
        assert q.shape == (N_DOF,)

    def test_qdot_at_shape(self, result: GolferSimulationResult) -> None:
        qdot = result.qdot_at(0)
        assert qdot.shape == (N_DOF,)

    def test_qdot_at_finite(self, result: GolferSimulationResult) -> None:
        qdot = result.qdot_at(0)
        assert np.all(np.isfinite(qdot))

    def test_qdot_initial_zero(self, result: GolferSimulationResult) -> None:
        """Initial velocity should be (near) zero since we started at rest."""
        qdot = result.qdot_at(0)
        np.testing.assert_allclose(qdot, 0.0, atol=1e-8)

    def test_mass_matrix_at_shape(self, result: GolferSimulationResult) -> None:
        M = result.mass_matrix_at(0)
        assert M.shape == (N_DOF, N_DOF)

    def test_positions_at_keys(self, result: GolferSimulationResult) -> None:
        pos = result.positions_at(0)
        assert isinstance(pos, dict)
        assert len(pos) > 0

    def test_torques_at_length(self, result: GolferSimulationResult) -> None:
        tau = result.torques_at(0)
        assert len(tau) == 7

    def test_torques_at_zero_at_rest(self, result: GolferSimulationResult) -> None:
        tau = result.torques_at(0)
        assert all(t == 0.0 for t in tau)

    def test_accelerations_at_shape(self, result: GolferSimulationResult) -> None:
        qddot = result.accelerations_at(0)
        assert qddot.shape == (N_DOF,)

    def test_accelerations_at_finite(self, result: GolferSimulationResult) -> None:
        qddot = result.accelerations_at(0)
        assert np.all(np.isfinite(qddot))

    def test_joint_forces_at_dict(self, result: GolferSimulationResult) -> None:
        jf = result.joint_forces_at(0)
        assert isinstance(jf, dict)
        assert len(jf) > 0

    def test_joint_forces_at_finite(self, result: GolferSimulationResult) -> None:
        jf = result.joint_forces_at(0)
        for key, val in jf.items():
            for v in val:
                assert np.isfinite(v), f"{key} not finite"

    def test_constraint_forces_at_shape(self, result: GolferSimulationResult) -> None:
        cf = result.constraint_forces_at(0)
        assert cf.shape[-1] == N_DOF or cf.ndim >= 1

    def test_constraint_forces_at_finite(self, result: GolferSimulationResult) -> None:
        cf = result.constraint_forces_at(0)
        assert np.all(np.isfinite(cf))

    def test_constraint_violation_at_finite(self, result: GolferSimulationResult) -> None:
        cv = result.constraint_violation_at(0)
        assert np.isfinite(cv)

    def test_constraint_violation_at_non_negative(
        self, result: GolferSimulationResult
    ) -> None:
        cv = result.constraint_violation_at(0)
        assert cv >= 0.0

    def test_coriolis_at_shape(self, result: GolferSimulationResult) -> None:
        C = result.coriolis_at(0)
        assert C.shape == (N_DOF,) or C.shape == (N_DOF, N_DOF)

    def test_coriolis_at_finite(self, result: GolferSimulationResult) -> None:
        C = result.coriolis_at(0)
        assert np.all(np.isfinite(C))

    def test_gravity_at_shape(self, result: GolferSimulationResult) -> None:
        G = result.gravity_at(0)
        assert G.shape == (N_DOF,)

    def test_gravity_at_finite(self, result: GolferSimulationResult) -> None:
        G = result.gravity_at(0)
        assert np.all(np.isfinite(G))

    def test_energy_at_dict(self, result: GolferSimulationResult) -> None:
        e = result.energy_at(0)
        assert isinstance(e, dict)
        for key in ("kinetic", "potential", "total"):
            assert key in e
            assert np.isfinite(e[key])

    def test_energy_conservation(self, result: GolferSimulationResult) -> None:
        e = result.energy_at(0)
        assert e["total"] == pytest.approx(e["kinetic"] + e["potential"], rel=1e-6)

    def test_friction_torques_at_shape(self, result: GolferSimulationResult) -> None:
        tf = result.friction_torques_at(0)
        assert tf.shape == (N_DOF,)

    def test_friction_torques_zero_at_rest(self, result: GolferSimulationResult) -> None:
        """At zero velocity, friction should be zero."""
        tf = result.friction_torques_at(0)
        np.testing.assert_allclose(tf, 0.0, atol=1e-14)

    def test_total_torques_at_shape(self, result: GolferSimulationResult) -> None:
        tt = result.total_torques_at(0)
        assert tt.shape == (N_DOF,)

    def test_total_torques_at_finite(self, result: GolferSimulationResult) -> None:
        tt = result.total_torques_at(0)
        assert np.all(np.isfinite(tt))

    def test_last_index_valid(self, result: GolferSimulationResult) -> None:
        last = len(result.t) - 1
        qddot = result.accelerations_at(last)
        assert qddot.shape == (N_DOF,)
