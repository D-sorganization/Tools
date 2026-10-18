"""Rust vs Python parity for the golfer constrained-dynamics solver (#5638).

After #5635 the Python solver solves the KKT system on
``INDEPENDENT_CLOSURE_ROWS`` only, returns zero multipliers on the redundant
closure rows and raises ``ValueError`` when the independent rows lose rank.
The compiled Rust solver must be the same solver, not a second formulation:

* same closure equations (``Phi`` and ``Phi_q``),
* same independent rows (one definition, read by both backends),
* same multiplier layout (rows 2-3 exactly zero),
* same typed error (``ValueError``) on rank loss,
* same accelerations / joint constraint force on closed configurations.

These tests need the compiled ``pendulum_core`` extension
(``maturin develop`` in ``src/pendulum_simulator/pendulum-core``) and skip
cleanly when it is not installed.  Tolerances are the issue's acceptance
bounds and are intentionally not loosened.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import double_pendulum_golf.constraint_solver as constraint_solver_module
import double_pendulum_golf.native_backend as native_backend
from double_pendulum_golf.golfer_dynamics import analytical_coriolis
from double_pendulum_golf.golfer_constraints import (
    INDEPENDENT_CLOSURE_ROWS,
    analytical_constraint_jacobian,
    constraint_vector,
)
from double_pendulum_golf.physics_golfer import N_CONSTRAINTS, N_DOF, GolferParams

pytestmark = pytest.mark.skipif(
    not native_backend.golfer_native_available(),
    reason="pendulum_core Rust extension not installed (maturin develop)",
)

QDDOT_REL_TOL = 1e-7
JOINT_FORCE_TOL = 1e-6
PHI_TOL = 1e-12
PHI_Q_TOL = 1e-12
N_CONFIGURATIONS = 25
ALPHA = 5.0
BETA = 5.0
_TAU = np.array([0.1, 0.2, -0.1, 0.0, 0.3, 0.1, 0.0, 0.0])


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


@pytest.fixture
def core():  # noqa: ANN201
    return native_backend._pendulum_core


@pytest.fixture(autouse=True)
def _rust_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PENDULUM_GOLFER_BACKEND", "rust")


def _python_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route every golfer kernel through the pure-Python implementation."""
    monkeypatch.setenv("PENDULUM_GOLFER_BACKEND", "python")


def _closed_configuration(
    params: GolferParams, seed: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[np.ndarray, np.ndarray]:
    """Closed ``(q, qdot)`` produced by the pure-Python projections."""
    with monkeypatch.context() as patch:
        patch.setenv("PENDULUM_GOLFER_BACKEND", "python")
        rng = np.random.default_rng(seed)
        q = constraint_solver_module.project_to_constraints(
            rng.uniform(-1.0, 1.0, N_DOF), params
        )
        qdot = constraint_solver_module.project_velocity(
            q, rng.uniform(-2.0, 2.0, N_DOF), params
        )
    return q, qdot


def _python_dynamics(
    q: np.ndarray, qdot: np.ndarray, params: GolferParams, monkeypatch: pytest.MonkeyPatch
) -> tuple[np.ndarray, np.ndarray]:
    with monkeypatch.context() as patch:
        patch.setenv("PENDULUM_GOLFER_BACKEND", "python")
        state = np.concatenate([q, qdot])
        return constraint_solver_module._solve_constrained_dynamics(
            state, 0.0, params, lambda t: tuple(_TAU[:7])
        )


def _rust_dynamics(
    q: np.ndarray, qdot: np.ndarray, params: GolferParams, core
) -> tuple[np.ndarray, np.ndarray]:
    qddot, lam = core.py_golfer_constrained_dynamics(
        q.tolist(),
        qdot.tolist(),
        _TAU.tolist(),
        native_backend._to_rust_golfer_params(params),
        ALPHA,
        BETA,
    )
    return np.asarray(qddot), np.asarray(lam)


class TestSingleClosureDefinition:
    def test_both_backends_read_the_same_independent_rows(self, core) -> None:
        assert tuple(core.golfer_independent_closure_rows()) == INDEPENDENT_CLOSURE_ROWS

    def test_both_backends_use_the_same_rank_tolerance(self, core) -> None:
        assert core.golfer_closure_rank_rtol() == constraint_solver_module.CLOSURE_RANK_RTOL

    @pytest.mark.parametrize("seed", range(N_CONFIGURATIONS))
    def test_closure_vector_matches_python(
        self, params: GolferParams, core, seed: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Not closed on purpose: the equations must agree off the manifold too.
        q = np.random.default_rng(100 + seed).uniform(-1.0, 1.0, N_DOF)
        rust = np.asarray(
            core.py_golfer_constraint_vector(
                q.tolist(), native_backend._to_rust_golfer_params(params)
            )
        )
        _python_solver(monkeypatch)
        python = constraint_vector(q, params)
        assert rust.shape == (N_CONSTRAINTS,)
        np.testing.assert_allclose(rust, python, rtol=0.0, atol=PHI_TOL)

    @pytest.mark.parametrize("seed", range(N_CONFIGURATIONS))
    def test_closure_jacobian_matches_python(
        self, params: GolferParams, core, seed: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        q = np.random.default_rng(200 + seed).uniform(-1.0, 1.0, N_DOF)
        rust = np.asarray(
            core.py_golfer_constraint_jacobian(
                q.tolist(), native_backend._to_rust_golfer_params(params)
            )
        )
        _python_solver(monkeypatch)
        python = analytical_constraint_jacobian(q, params)
        assert rust.shape == (N_CONSTRAINTS, N_DOF)
        np.testing.assert_allclose(rust, python, rtol=0.0, atol=PHI_Q_TOL)


class TestSolverParity:
    @pytest.mark.parametrize("seed", range(N_CONFIGURATIONS))
    def test_accelerations_agree(
        self, params: GolferParams, core, seed: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        q, qdot = _closed_configuration(params, seed, monkeypatch)
        qddot_py, _ = _python_dynamics(q, qdot, params, monkeypatch)
        qddot_rs, _ = _rust_dynamics(q, qdot, params, core)
        scale = np.max(np.abs(qddot_py))
        assert np.max(np.abs(qddot_rs - qddot_py)) <= QDDOT_REL_TOL * scale

    @pytest.mark.parametrize("seed", range(N_CONFIGURATIONS))
    def test_total_joint_constraint_force_agrees(
        self, params: GolferParams, core, seed: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        q, qdot = _closed_configuration(params, seed, monkeypatch)
        _, lam_py = _python_dynamics(q, qdot, params, monkeypatch)
        _, lam_rs = _rust_dynamics(q, qdot, params, core)
        _python_solver(monkeypatch)
        phi_q = analytical_constraint_jacobian(q, params)
        force_py = phi_q.T @ lam_py
        force_rs = phi_q.T @ lam_rs
        assert np.max(np.abs(force_rs - force_py)) <= JOINT_FORCE_TOL

    @pytest.mark.parametrize("seed", range(N_CONFIGURATIONS))
    def test_multiplier_layout_rows_2_3_are_zero(
        self, params: GolferParams, core, seed: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        q, qdot = _closed_configuration(params, seed, monkeypatch)
        _, lam_rs = _rust_dynamics(q, qdot, params, core)
        assert lam_rs.shape == (N_CONSTRAINTS,)
        redundant = [r for r in range(N_CONSTRAINTS) if r not in INDEPENDENT_CLOSURE_ROWS]
        assert redundant == [2, 3]
        assert np.all(lam_rs[redundant] == 0.0)


class TestRankLossContract:
    @staticmethod
    def _degenerate_kkt_inputs() -> tuple[np.ndarray, ...]:
        phi_q = np.zeros((N_CONSTRAINTS, N_DOF))
        phi_q[0, 0] = phi_q[1, 1] = 1.0
        phi_q[1] = 2.0 * phi_q[0]  # rows 0-1 now parallel: rank 1
        return np.eye(N_DOF), phi_q, np.zeros(N_DOF), np.zeros(N_CONSTRAINTS)

    def test_python_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            constraint_solver_module._solve_independent_kkt(*self._degenerate_kkt_inputs())

    def test_rust_raises_the_same_typed_error(self, core) -> None:
        mass, phi_q, rhs_dyn, rhs_con = self._degenerate_kkt_inputs()
        with pytest.raises(ValueError, match="rank"):
            core.py_golfer_solve_independent_kkt(
                mass.tolist(), phi_q.tolist(), rhs_dyn.tolist(), rhs_con.tolist()
            )

    def test_rust_rank_error_is_not_swallowed_as_a_fallback(self, core) -> None:
        # RuntimeError would be caught by native_backend and silently fall back
        # to Python; ValueError must propagate.
        mass, phi_q, rhs_dyn, rhs_con = self._degenerate_kkt_inputs()
        with pytest.raises(Exception) as excinfo:
            core.py_golfer_solve_independent_kkt(
                mass.tolist(), phi_q.tolist(), rhs_dyn.tolist(), rhs_con.tolist()
            )
        assert type(excinfo.value) is ValueError

    def test_rust_rejects_non_finite_like_python(self, core) -> None:
        mass, phi_q, rhs_dyn, rhs_con = self._degenerate_kkt_inputs()
        phi_q[1] = 0.0
        phi_q[1, 1] = 1.0
        rhs_dyn[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            core.py_golfer_solve_independent_kkt(
                mass.tolist(), phi_q.tolist(), rhs_dyn.tolist(), rhs_con.tolist()
            )

    def test_solver_kernels_agree_on_healthy_inputs(self, core) -> None:
        mass = np.diag(np.arange(1.0, N_DOF + 1.0))
        phi_q = np.zeros((N_CONSTRAINTS, N_DOF))
        phi_q[0, 0] = phi_q[1, 1] = 1.0
        phi_q[2, 0] = phi_q[3, 1] = 5.0  # redundant rows must be ignored
        rhs_dyn = np.linspace(-1.0, 1.0, N_DOF)
        rhs_con = np.array([0.3, -0.2, 9.0, 9.0])
        qddot_py, lam_py = constraint_solver_module._solve_independent_kkt(
            mass, phi_q, rhs_dyn, rhs_con
        )
        qddot_rs, lam_rs = core.py_golfer_solve_independent_kkt(
            mass.tolist(), phi_q.tolist(), rhs_dyn.tolist(), rhs_con.tolist()
        )
        np.testing.assert_allclose(qddot_rs, qddot_py, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(lam_rs, lam_py, rtol=1e-12, atol=1e-12)


class TestCoriolisParity:
    """The Coriolis/centrifugal vector is one formula in both backends (#5640).

    Before #5640 both dropped the third Christoffel term (``0.5 * Mdot @ qdot``), so
    the solver-level parity above held while both were wrong.  The independent
    Euler-Lagrange oracle lives in ``test_golfer_coriolis_oracle.py``; this checks
    the compiled kernel against the (oracle-validated) Python one term for term.
    """

    CORIOLIS_RTOL = 1e-9

    @pytest.mark.parametrize("seed", range(N_CONFIGURATIONS))
    def test_coriolis_vector_matches_python(
        self, params: GolferParams, core: Any, seed: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rng = np.random.default_rng(300 + seed)
        q = rng.uniform(-1.0, 1.0, N_DOF)
        qdot = rng.uniform(-2.0, 2.0, N_DOF)
        with monkeypatch.context() as patch:
            patch.setenv("PENDULUM_GOLFER_BACKEND", "python")
            expected = analytical_coriolis(q, qdot, params)
        actual = np.asarray(
            core.py_golfer_coriolis(
                q.tolist(), qdot.tolist(), native_backend._to_rust_golfer_params(params)
            )
        )
        np.testing.assert_allclose(
            actual, expected, rtol=self.CORIOLIS_RTOL, atol=self.CORIOLIS_RTOL
        )

    def test_rust_rejects_wrong_length_inputs(self, params: GolferParams, core: Any) -> None:
        rust_params = native_backend._to_rust_golfer_params(params)
        with pytest.raises(ValueError):
            core.py_golfer_coriolis([0.0] * 7, [0.0] * 8, rust_params)
        with pytest.raises(ValueError):
            core.py_golfer_coriolis([0.0] * 8, [0.0] * 7, rust_params)
