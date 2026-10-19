"""The KKT solve uses only the independent closure rows (#5635).

With the corrected club FK (#5626 / #5628) rows 2-3 of ``constraint_vector``
are the rows 0-1 wrist-pin residual rotated into the club frame, so the full
4-row closure Jacobian has rank 2 on the constraint manifold.  Solving the KKT
system with all four rows is therefore (near-)singular and can hand back
large, meaningless multipliers without raising.  The solver must use
``INDEPENDENT_CLOSURE_ROWS`` only (parity with UpstreamDrift #12608 / #12668)
and report zero multipliers on the redundant rows.

Tolerances
----------
``_constraint_acceleration_bias`` is a forward finite difference with step
``1e-7``.  Its truncation error is ~ ``eps * |d2J| * |qdot|^2`` ~ 1e-7 for the
O(1) velocities used here, and it enters rows 0-1 and rows 2-3 differently.
That is the only source of disagreement between the redundant rows, so:

* rows 0-1 are solved exactly: residual < ``ROWS_01_TOL`` (measured <= 3e-14);
* rows 2-3 close to the FD noise level: residual < ``ROWS_23_TOL``
  (measured <= 1e-7);
* the accelerations agree with the legacy 4-row minimum-norm solve to
  ``QDDOT_REL_TOL`` relative (measured <= 4e-9), i.e. the kinematic outputs do
  not move beyond the FD noise floor.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import double_pendulum_golf.constraint_solver as constraint_solver_module
import double_pendulum_golf.golfer_constraints as golfer_constraints_module
from double_pendulum_golf.constraint_solver import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    _constraint_acceleration_bias,
    constrained_accelerations,
    constraint_forces,
    equations_of_motion,
    project_to_constraints,
    project_velocity,
)
from double_pendulum_golf.golfer_constraints import (
    INDEPENDENT_CLOSURE_ROWS,
    analytical_constraint_jacobian,
    constraint_vector,
    friction_torque_vector,
)
from double_pendulum_golf.golfer_dynamics import (
    analytical_coriolis,
    analytical_gravity_vector,
    analytical_mass_matrix,
)
from double_pendulum_golf.physics_golfer import N_CONSTRAINTS, N_DOF, GolferParams

ROWS_01_TOL = 1e-10
ROWS_23_TOL = 1e-6
QDDOT_REL_TOL = 1e-7
GENERALIZED_FORCE_TOL = 1e-6
RANK_TOL = 1e-8
CLOSED_SEEDS = range(8)

_TAU = (0.1, 0.2, -0.1, 0.0, 0.3, 0.1, 0.0)
TorqueFunc = Callable[[float], tuple[float, float, float, float, float, float, float]]


def _torque(t: float) -> tuple[float, float, float, float, float, float, float]:
    return _TAU


@pytest.fixture
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
        L_club=1.1,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
    )


@pytest.fixture(autouse=True)
def _python_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the Python KKT path; the Rust closure is a different formulation."""
    monkeypatch.setattr(
        constraint_solver_module._native_backend,
        "golfer_constrained_dynamics",
        lambda *args, **kwargs: None,
    )


def _closed_state(params: GolferParams, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = project_to_constraints(rng.uniform(-1.0, 1.0, N_DOF), params)
    qdot = project_velocity(q, rng.uniform(-2.0, 2.0, N_DOF), params)
    return np.concatenate([q, qdot])


def _baumgarte_rhs(state: np.ndarray, params: GolferParams) -> np.ndarray:
    q, qdot = state[:N_DOF], state[N_DOF:]
    phi_q = analytical_constraint_jacobian(q, params)
    gamma = _constraint_acceleration_bias(q, qdot, params)
    return (
        -gamma
        - 2.0 * DEFAULT_ALPHA * (phi_q @ qdot)
        - DEFAULT_BETA**2 * constraint_vector(q, params)
    )


def _legacy_full_row_solve(
    state: np.ndarray, params: GolferParams
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-#5635 formulation: minimum-norm KKT over all four rows."""
    q, qdot = state[:N_DOF], state[N_DOF:]
    tau = np.zeros(N_DOF)
    tau[:7] = _TAU
    rhs_dyn = (
        tau
        + friction_torque_vector(qdot, params)
        - analytical_coriolis(q, qdot, params)
        - analytical_gravity_vector(q, params)
    )
    phi_q = analytical_constraint_jacobian(q, params)
    kkt = np.zeros((N_DOF + N_CONSTRAINTS, N_DOF + N_CONSTRAINTS))
    kkt[:N_DOF, :N_DOF] = analytical_mass_matrix(q, params)
    kkt[:N_DOF, N_DOF:] = phi_q.T
    kkt[N_DOF:, :N_DOF] = phi_q
    rhs = np.concatenate([rhs_dyn, _baumgarte_rhs(state, params)])
    sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return sol[:N_DOF], sol[N_DOF:]


class TestIndependentClosureRowsDefinition:
    def test_rows_are_the_world_frame_wrist_pin(self) -> None:
        assert INDEPENDENT_CLOSURE_ROWS == (0, 1)

    def test_solver_reuses_the_single_definition(self) -> None:
        assert (
            constraint_solver_module.INDEPENDENT_CLOSURE_ROWS
            is golfer_constraints_module.INDEPENDENT_CLOSURE_ROWS
        )

    @pytest.mark.parametrize("seed", CLOSED_SEEDS)
    def test_full_closure_has_rank_two_and_independent_rows_are_full_rank(
        self, params: GolferParams, seed: int
    ) -> None:
        q = _closed_state(params, seed)[:N_DOF]
        phi_q = analytical_constraint_jacobian(q, params)
        sv_full = np.linalg.svd(phi_q, compute_uv=False)
        sv_rows = np.linalg.svd(phi_q[list(INDEPENDENT_CLOSURE_ROWS)], compute_uv=False)
        assert np.sum(sv_full > RANK_TOL * sv_full[0]) == 2
        assert sv_rows[-1] > RANK_TOL * sv_rows[0]


class TestMultipliers:
    @pytest.mark.parametrize("seed", CLOSED_SEEDS)
    def test_redundant_rows_carry_zero_multiplier(
        self, params: GolferParams, seed: int
    ) -> None:
        lam = constraint_forces(_closed_state(params, seed), 0.0, params, _torque)
        assert lam.shape == (N_CONSTRAINTS,)
        redundant = [i for i in range(N_CONSTRAINTS) if i not in INDEPENDENT_CLOSURE_ROWS]
        assert np.all(lam[redundant] == 0.0)
        assert np.all(lam[list(INDEPENDENT_CLOSURE_ROWS)] != 0.0)

    @pytest.mark.parametrize("seed", CLOSED_SEEDS)
    def test_generalized_constraint_force_matches_full_row_solve(
        self, params: GolferParams, seed: int
    ) -> None:
        state = _closed_state(params, seed)
        phi_q = analytical_constraint_jacobian(state[:N_DOF], params)
        lam = constraint_forces(state, 0.0, params, _torque)
        _, lam_ref = _legacy_full_row_solve(state, params)
        np.testing.assert_allclose(
            phi_q.T @ lam, phi_q.T @ lam_ref, rtol=0.0, atol=GENERALIZED_FORCE_TOL
        )


class TestAllFourRowsClose:
    @pytest.mark.parametrize("seed", CLOSED_SEEDS)
    def test_acceleration_level_closure_on_every_row(
        self, params: GolferParams, seed: int
    ) -> None:
        state = _closed_state(params, seed)
        qddot = constrained_accelerations(state, 0.0, params, _torque)
        residual = analytical_constraint_jacobian(
            state[:N_DOF], params
        ) @ qddot - _baumgarte_rhs(state, params)
        rows = list(INDEPENDENT_CLOSURE_ROWS)
        redundant = [i for i in range(N_CONSTRAINTS) if i not in rows]
        assert np.max(np.abs(residual[rows])) < ROWS_01_TOL
        assert np.max(np.abs(residual[redundant])) < ROWS_23_TOL

    def test_rk4_rollout_keeps_all_four_rows_closed(self, params: GolferParams) -> None:
        state = _closed_state(params, 0)
        dt = 1e-3
        for step in range(200):
            t = step * dt
            k1 = equations_of_motion(state, t, params, _torque)
            k2 = equations_of_motion(state + 0.5 * dt * k1, t + 0.5 * dt, params, _torque)
            k3 = equations_of_motion(state + 0.5 * dt * k2, t + 0.5 * dt, params, _torque)
            k4 = equations_of_motion(state + dt * k3, t + dt, params, _torque)
            state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        phi = constraint_vector(state[:N_DOF], params)
        assert np.all(np.isfinite(state))
        assert np.max(np.abs(phi)) < ROWS_23_TOL


class TestKinematicOutputsUnchanged:
    @pytest.mark.parametrize("seed", CLOSED_SEEDS)
    def test_accelerations_match_full_row_solve(self, params: GolferParams, seed: int) -> None:
        state = _closed_state(params, seed)
        qddot = constrained_accelerations(state, 0.0, params, _torque)
        qddot_ref, _ = _legacy_full_row_solve(state, params)
        scale = np.max(np.abs(qddot_ref))
        assert np.max(np.abs(qddot - qddot_ref)) < QDDOT_REL_TOL * scale


class TestRankContract:
    def test_raises_when_independent_rows_lose_rank(
        self, params: GolferParams, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _closed_state(params, 0)
        real = analytical_constraint_jacobian(state[:N_DOF], params)
        degenerate = real.copy()
        degenerate[1] = 2.0 * degenerate[0]
        monkeypatch.setattr(
            constraint_solver_module,
            "constraint_jacobian",
            lambda q, p: degenerate.copy(),
        )
        with pytest.raises(ValueError, match="rank"):
            constraint_forces(state, 0.0, params, _torque)

    def test_independent_kkt_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            constraint_solver_module._solve_independent_kkt(
                np.eye(N_DOF),
                np.zeros((3, N_DOF)),
                np.zeros(N_DOF),
                np.zeros(N_CONSTRAINTS),
            )

    def test_independent_kkt_rejects_non_finite(self) -> None:
        phi_q = np.zeros((N_CONSTRAINTS, N_DOF))
        phi_q[0, 0] = phi_q[1, 1] = 1.0
        with pytest.raises(ValueError, match="finite"):
            constraint_solver_module._solve_independent_kkt(
                np.eye(N_DOF),
                phi_q,
                np.full(N_DOF, np.nan),
                np.zeros(N_CONSTRAINTS),
            )
