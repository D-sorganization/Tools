# ruff: noqa: E501
"""Integration tests for the golfer simulation engine.

Verifies that the constrained ODE integration produces physically
consistent results over very short time spans.

NOTE: The golfer model uses numerical Jacobians extensively, making each
RHS evaluation expensive.  Tests use very short durations and loose
tolerances so they finish in seconds, not minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.constraint_solver import (
    constraint_violation,
    project_to_constraints,
    project_velocity,
)
from double_pendulum_golf.physics_golfer import (
    N_DOF,
    GolferParams,
)
from double_pendulum_golf.simulation_golfer import (
    GolferSimulationResult,
    make_polynomial_torque,
    run_simulation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GOLFER_PARAMS = GolferParams(
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


def _zero_torque(_t: float) -> tuple:
    """Zero torque for all 7 joints."""
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _consistent_initial_state(params: GolferParams) -> np.ndarray:
    """Create a consistent initial state on the constraint manifold."""
    q = project_to_constraints(np.zeros(N_DOF), params)
    qdot = project_velocity(q, np.zeros(N_DOF), params)
    return np.concatenate([q, qdot])


@pytest.fixture(scope="module")
def sim_result() -> GolferSimulationResult:
    """Run one short simulation shared by all tests in this module.

    Uses a very short t_end (0.01 s) with only 2 output steps and
    loose solver tolerances so the test finishes quickly.
    """
    state0 = _consistent_initial_state(_GOLFER_PARAMS)
    return run_simulation(
        _GOLFER_PARAMS,
        state0,
        t_end=0.01,
        torque_func=_zero_torque,
        dt=0.005,
        rtol=1e-4,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Polynomial torque builder (no simulation needed — fast)
# ---------------------------------------------------------------------------


class TestMakePolynomialTorque:
    """Polynomial torque builder must produce correct output."""

    def test_constant_torque(self) -> None:
        tf = make_polynomial_torque([1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0])
        result = tf(0.0)
        assert len(result) == 7
        assert result == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)

    def test_linear_torque(self) -> None:
        tf = make_polynomial_torque(
            [0.0, 1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
        )
        result = tf(2.0)
        assert abs(result[0] - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Run-simulation output structure
# ---------------------------------------------------------------------------


class TestRunSimulation:
    """Integration test for the full simulation pipeline."""

    def test_runs_without_error(self, sim_result: GolferSimulationResult) -> None:
        assert isinstance(sim_result, GolferSimulationResult)
        assert sim_result.n_steps >= 2

    def test_states_shape(self, sim_result: GolferSimulationResult) -> None:
        assert sim_result.states.shape[1] == 2 * N_DOF

    def test_time_monotonic(self, sim_result: GolferSimulationResult) -> None:
        assert np.all(
            np.diff(sim_result.t) > 0
        ), "Time must be monotonically increasing"

    def test_constraint_bounded(self, sim_result: GolferSimulationResult) -> None:
        for i in range(sim_result.n_steps):
            v = constraint_violation(sim_result.states[i], _GOLFER_PARAMS)
            assert v < 0.1, f"Constraint violation at step {i}: {v}"

    def test_run_with_joint_limits(self) -> None:
        from double_pendulum_golf.physics import JointLimitsNDOF

        limits = JointLimitsNDOF(
            angle_min=np.array([-1.0] * 7),
            angle_max=np.array([1.0] * 7),
            stiffness=100.0,
            damping=10.0,
        )
        state0 = _consistent_initial_state(_GOLFER_PARAMS)
        result = run_simulation(
            _GOLFER_PARAMS,
            state0,
            t_end=0.01,
            torque_func=_zero_torque,
            dt=0.005,
            limits=limits,
            rtol=1e-4,
            atol=1e-6,
        )
        assert result.n_steps >= 2


class TestConstraintViolationPostcondition:
    """Constraint monitoring postcondition: drift must stay within abort threshold."""

    def test_violation_below_abort_threshold(
        self, sim_result: GolferSimulationResult
    ) -> None:
        """All trajectory steps must have constraint violation below abort threshold."""
        abort_tol = 1e-2
        for i in range(sim_result.n_steps):
            v = constraint_violation(sim_result.states[i], _GOLFER_PARAMS)
            assert (
                v < abort_tol
            ), f"Constraint violation {v:.3e} at step {i} exceeds abort threshold {abort_tol:.3e}"

    def test_violation_finite_at_all_steps(
        self, sim_result: GolferSimulationResult
    ) -> None:
        """Constraint violation must be finite at every trajectory step."""
        for i in range(sim_result.n_steps):
            v = constraint_violation(sim_result.states[i], _GOLFER_PARAMS)
            assert np.isfinite(v), f"Non-finite constraint violation at step {i}"


class TestGolferSimulationResultContracts:
    """Trajectory-level DbC validation for GolferSimulationResult."""

    def test_constructor_rejects_non_monotonic_time(self) -> None:
        with pytest.raises((ValueError, TypeError), match="strictly increasing"):
            GolferSimulationResult(
                t=np.array([0.0, 0.0]),
                states=np.zeros((2, 2 * N_DOF)),
                params=_GOLFER_PARAMS,
                torque_func=_zero_torque,
            )

    def test_constructor_rejects_wrong_state_width(self) -> None:
        with pytest.raises((ValueError, TypeError), match="states must have width"):
            GolferSimulationResult(
                t=np.array([0.0, 0.01]),
                states=np.zeros((2, 2 * N_DOF - 1)),
                params=_GOLFER_PARAMS,
                torque_func=_zero_torque,
            )


# ---------------------------------------------------------------------------
# Result accessor methods
# ---------------------------------------------------------------------------


class TestGolferSimulationResult:
    """Accessor methods must return correct shapes."""

    def test_q_at(self, sim_result: GolferSimulationResult) -> None:
        q = sim_result.q_at(0)
        assert q.shape == (N_DOF,)

    def test_positions_at(self, sim_result: GolferSimulationResult) -> None:
        pos = sim_result.positions_at(0)
        assert "hub" in pos
        assert "club_tip" in pos

    def test_energy_at(self, sim_result: GolferSimulationResult) -> None:
        e = sim_result.energy_at(0)
        assert "kinetic" in e
        assert "potential" in e
        assert "total" in e
        assert np.isclose(e["total"], e["kinetic"] + e["potential"])

    def test_mass_matrix_at(self, sim_result: GolferSimulationResult) -> None:
        M = sim_result.mass_matrix_at(0)
        assert M.shape == (N_DOF, N_DOF)

    def test_all_positions_and_energies(
        self, sim_result: GolferSimulationResult
    ) -> None:
        positions = sim_result.all_positions()
        energies = sim_result.all_energies()
        assert len(positions) == sim_result.n_steps
        assert energies["kinetic"].shape == (sim_result.n_steps,)
        assert energies["potential"].shape == (sim_result.n_steps,)
        assert energies["total"].shape == (sim_result.n_steps,)
