"""Wave 4 gap-fill tests covering remaining uncovered lines.

Modules targeted:
- golfer_dynamics.py: TypeError/ValueError raises + state-vector (>N_DOF) truncation
- golfer_constraints.py: TypeError/ValueError raises + native return paths
- constraint_solver.py: torque_limits path (lines 100-102), analytical_constraint_bias (232)
- club_forces.py: state vector truncation (lines 70, 257)
- simulation_triple.py: remaining lines (if any)
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics_golfer import GolferParams, N_DOF


@pytest.fixture(scope="module")
def gp() -> GolferParams:
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
def q8(gp: GolferParams) -> np.ndarray:
    """Valid 8-DOF configuration vector."""
    from double_pendulum_golf.constraint_solver import project_to_constraints

    q0 = np.zeros(N_DOF)
    return project_to_constraints(q0, gp)


@pytest.fixture(scope="module")
def q16(q8: np.ndarray) -> np.ndarray:
    """16-element state vector [q, qdot] — tests >N_DOF truncation."""
    return np.concatenate([q8, np.zeros(N_DOF)])


# ===========================================================================
# golfer_dynamics.py — TypeError/ValueError + state-vector truncation
# ===========================================================================


class TestGolferDynamicsInputValidation:
    def test_potential_energy_wrong_type_p(self, q8: np.ndarray) -> None:
        from double_pendulum_golf.golfer_dynamics import potential_energy

        with pytest.raises(TypeError, match="GolferParams"):
            potential_energy(q8, "not_a_params")  # type: ignore[arg-type]

    def test_potential_energy_state_vector_truncated(
        self, q16: np.ndarray, gp: GolferParams
    ) -> None:
        from double_pendulum_golf.golfer_dynamics import potential_energy

        result = potential_energy(q16, gp)
        assert np.isfinite(result)

    def test_analytical_fk_jacobians_wrong_p_type(self, q8: np.ndarray) -> None:
        from double_pendulum_golf.golfer_dynamics import analytical_fk_jacobians

        with pytest.raises(TypeError, match="GolferParams"):
            analytical_fk_jacobians(q8, "not_a_params")  # type: ignore[arg-type]

    def test_analytical_fk_jacobians_state_vector_truncated(
        self, q16: np.ndarray, gp: GolferParams
    ) -> None:
        from double_pendulum_golf.golfer_dynamics import analytical_fk_jacobians

        result = analytical_fk_jacobians(q16, gp)
        assert isinstance(result, dict)

    def test_analytical_mass_matrix_wrong_p_type(self, q8: np.ndarray) -> None:
        from double_pendulum_golf.golfer_dynamics import analytical_mass_matrix

        with pytest.raises(TypeError, match="GolferParams"):
            analytical_mass_matrix(q8, "not_a_params")  # type: ignore[arg-type]

    def test_analytical_mass_matrix_state_vector_truncated(
        self, q16: np.ndarray, gp: GolferParams
    ) -> None:
        from double_pendulum_golf.golfer_dynamics import analytical_mass_matrix

        M = analytical_mass_matrix(q16, gp)
        assert M.shape == (N_DOF, N_DOF)

    def test_analytical_gravity_vector_wrong_p_type(self, q8: np.ndarray) -> None:
        from double_pendulum_golf.golfer_dynamics import analytical_gravity_vector

        with pytest.raises(TypeError, match="GolferParams"):
            analytical_gravity_vector(q8, "not_a_params")  # type: ignore[arg-type]

    def test_analytical_gravity_vector_state_vector_truncated(
        self, q16: np.ndarray, gp: GolferParams
    ) -> None:
        from double_pendulum_golf.golfer_dynamics import analytical_gravity_vector

        G = analytical_gravity_vector(q16, gp)
        assert G.shape == (N_DOF,)

    def test_analytical_coriolis_wrong_p_type(self, q8: np.ndarray) -> None:
        from double_pendulum_golf.golfer_dynamics import analytical_coriolis

        with pytest.raises(TypeError, match="GolferParams"):
            analytical_coriolis(q8, np.zeros(N_DOF), "not_a_params")  # type: ignore[arg-type]

    def test_analytical_coriolis_state_vector_truncated(
        self, q16: np.ndarray, gp: GolferParams
    ) -> None:
        from double_pendulum_golf.golfer_dynamics import analytical_coriolis

        qdot = np.zeros(N_DOF)
        C = analytical_coriolis(q16, qdot, gp)
        assert C.shape == (N_DOF,)

    def test_kinetic_energy_wrong_p_type(self, q8: np.ndarray) -> None:
        from double_pendulum_golf.golfer_dynamics import kinetic_energy

        with pytest.raises(TypeError, match="GolferParams"):
            kinetic_energy(q8, np.zeros(N_DOF), "not_a_params")  # type: ignore[arg-type]

    def test_kinetic_energy_state_vector_truncated(
        self, q16: np.ndarray, gp: GolferParams
    ) -> None:
        from double_pendulum_golf.golfer_dynamics import kinetic_energy

        qdot = np.zeros(N_DOF)
        T = kinetic_energy(q16, qdot, gp)
        assert np.isfinite(T)
        assert T >= 0.0


# ===========================================================================
# golfer_constraints.py — native return / TypeError raises
# ===========================================================================


class TestGolferConstraintsInputValidation:
    def test_constraint_vector_wrong_param_type(self, q8: np.ndarray) -> None:
        from double_pendulum_golf.golfer_constraints import constraint_vector

        with pytest.raises(TypeError):
            constraint_vector(q8, "not_params")  # type: ignore[arg-type]

    def test_constraint_jacobian_wrong_param_type(self, q8: np.ndarray) -> None:
        from double_pendulum_golf.golfer_constraints import (
            analytical_constraint_jacobian,
        )

        with pytest.raises(TypeError):
            analytical_constraint_jacobian(q8, "not_params")  # type: ignore[arg-type]

    def test_friction_torque_vector_wrong_type(self, q8: np.ndarray) -> None:
        from double_pendulum_golf.golfer_constraints import friction_torque_vector

        with pytest.raises(TypeError):
            friction_torque_vector(q8, "not_params")  # type: ignore[arg-type]

    def test_constraint_vector_state_vector_truncated(
        self, q16: np.ndarray, gp: GolferParams
    ) -> None:
        from double_pendulum_golf.golfer_constraints import constraint_vector

        phi = constraint_vector(q16, gp)
        assert phi.shape == (4,)


# ===========================================================================
# constraint_solver.py — torque_limits path (lines 100-102) +
#                        analytical_constraint_acceleration_bias (line 232)
# ===========================================================================


class TestConstraintSolverGaps:
    def test_solve_with_torque_limits(self, q8: np.ndarray, gp: GolferParams) -> None:
        """torque_limits path — lines 100-102."""
        from double_pendulum_golf.constraint_solver import _solve_constrained_dynamics

        state = np.concatenate([q8, np.zeros(N_DOF)])
        torque_limits = np.full(7, 0.001)
        qddot, lambda_f = _solve_constrained_dynamics(
            state,
            0.0,
            gp,
            lambda t: (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
            torque_limits=torque_limits,
        )
        assert qddot.shape == (N_DOF,)
        assert lambda_f.shape[0] > 0
        assert np.all(np.isfinite(qddot))

    def test_analytical_constraint_acceleration_bias(
        self, q8: np.ndarray, gp: GolferParams
    ) -> None:
        """Line 232 — API-compatible alias."""
        from double_pendulum_golf.constraint_solver import (
            analytical_constraint_acceleration_bias,
        )

        qdot = np.zeros(N_DOF)
        bias = analytical_constraint_acceleration_bias(q8, qdot, gp)
        assert bias.shape[0] > 0
        assert np.all(np.isfinite(bias))


# ===========================================================================
# club_forces.py — state vector truncation (lines 70, 257)
# ===========================================================================


class TestClubForcesStateVectorTruncation:
    def test_club_force_decomposition_state_vector(
        self, q16: np.ndarray, gp: GolferParams
    ) -> None:
        """Line 70: q.shape[0] > N_DOF → q = q[:N_DOF]."""
        from double_pendulum_golf.club_forces import club_force_decomposition

        qdot = np.zeros(N_DOF)
        qddot = np.zeros(N_DOF)
        forces = {
            "rh": (1.0, 0.5),
            "lh": (-0.5, 0.3),
        }

        # Pass q16 (state vector with 16 elements) — tests truncation on line 70
        result = club_force_decomposition(q16, qdot, qddot, gp, forces, alpha=0.5)
        assert isinstance(result, dict)
        assert "net_force" in result

    def test_club_force_decomposition_line257_branch(
        self, q16: np.ndarray, gp: GolferParams
    ) -> None:
        """Line 257: same club_force_decomposition called with state vector."""
        from double_pendulum_golf.club_forces import club_force_decomposition

        # This also exercises the q[:N_DOF] truncation on line 256-257
        forces = {"rh": (0.5, -0.3), "lh": (-0.5, 0.3)}
        result = club_force_decomposition(
            q16, np.zeros(N_DOF), np.zeros(N_DOF), gp, forces, alpha=0.0
        )
        assert np.all(np.isfinite(result["net_force"]))
