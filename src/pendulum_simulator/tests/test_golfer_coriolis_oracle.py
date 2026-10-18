"""Independent-oracle tests for the golfer Coriolis/centrifugal vector (#5640).

``analytical_coriolis`` used to contract ``0.5 * (dM + dM.T(0,2,1) - dM.T(1,2,0))``.
For a symmetric ``M`` the last two terms are the same tensor, so they cancelled and
the function returned ``0.5 * Mdot @ qdot`` instead of
``Mdot @ qdot - 0.5 * grad_q(qdot^T M qdot)``.

The pre-existing ``test_analytical_coriolis_parity`` could not see it: its
"numerical" reference is ``constraint_solver.coriolis_matrix``, which is an import
alias of ``analytical_coriolis`` itself, so that test compared the function with
itself.  Every oracle below is built from the mass matrix ``M(q)`` and the
Lagrangian alone and never calls the function under test to build the reference.

Energy conservation is checked too, but note it is *not* discriminating for this
defect: ``qdot^T (0.5 * Mdot @ qdot) == 0.5 * qdot^T Mdot qdot`` also holds for the
broken vector, so the broken solver conserves energy on a wrong trajectory.  The
trajectory test against the Euler-Lagrange reference is what pins the dynamics.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from double_pendulum_golf.golfer_dynamics import (
    analytical_coriolis,
    analytical_gravity_vector,
    analytical_mass_matrix,
    christoffel_first_kind,
    kinetic_energy,
    total_energy,
)
from double_pendulum_golf.physics_golfer import N_DOF, GolferParams

# DOFs 3 and 6 (the wrist angles) carry no inertia of their own in the unconstrained
# 8-DOF model: the mass matrix rows/columns are exactly zero, so the free dynamics are
# integrated on the remaining DOFs with those two held at rest.
INERT_DOFS = (3, 6)
ACTIVE_DOFS = tuple(i for i in range(N_DOF) if i not in INERT_DOFS)

FD_STEP = 1e-6
SEEDS = tuple(range(12))
ENERGY_DRIFT_BOUND = 1e-6  # relative, issue #5640 acceptance


@pytest.fixture(scope="module")
def params() -> GolferParams:
    # No scapula segments: with L_rscap/L_lscap > 0 the analytical Jacobians disagree
    # with forward_kinematics (separate defect), which would pollute a V/T oracle.
    return GolferParams(
        m_hub=0.001,
        m_r_upper=3.5,
        m_r_fore=2.0,
        m_l_upper=3.5,
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


def _state(seed: int, qdot_scale: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, N_DOF)
    qdot = rng.uniform(-qdot_scale, qdot_scale, N_DOF)
    qdot[list(INERT_DOFS)] = 0.0
    return q, qdot


def _mass_derivative(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """dM[i, j, k] = dM_ij/dq_k by central differences of M(q) alone."""
    dM = np.zeros((N_DOF, N_DOF, N_DOF))
    for k in range(N_DOF):
        e = np.zeros(N_DOF)
        e[k] = FD_STEP
        dM[:, :, k] = (analytical_mass_matrix(q + e, p) - analytical_mass_matrix(q - e, p)) / (
            2.0 * FD_STEP
        )
    return dM


def euler_lagrange_coriolis(q: np.ndarray, qdot: np.ndarray, p: GolferParams) -> np.ndarray:
    """C(q, qdot) from the Lagrangian, with no Christoffel index algebra.

    With ``T = 0.5 qdot^T M(q) qdot`` the Euler-Lagrange equations give, at zero
    acceleration, ``C = d/dt (M qdot) - dT/dq``.  Both derivatives are central
    differences: ``d/dt (M qdot)`` is the directional derivative of ``M(q) qdot``
    along ``qdot`` and ``dT/dq`` is the gradient of ``kinetic_energy``.
    """
    h = FD_STEP
    momentum_rate = (
        (analytical_mass_matrix(q + h * qdot, p) - analytical_mass_matrix(q - h * qdot, p))
        @ qdot
        / (2.0 * h)
    )
    grad_T = np.zeros(N_DOF)
    for i in range(N_DOF):
        e = np.zeros(N_DOF)
        e[i] = h
        grad_T[i] = (kinetic_energy(q + e, qdot, p) - kinetic_energy(q - e, qdot, p)) / (
            2.0 * h
        )
    result: np.ndarray = momentum_rate - grad_T
    return result


class TestChristoffelHelper:
    """christoffel_first_kind(dM) obeys the defining identities for any symmetric M."""

    @staticmethod
    def _random_dM(seed: int, n: int = 5) -> np.ndarray:
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(n, n, n))
        return 0.5 * (raw + raw.transpose(1, 0, 2))  # symmetric in (i, j)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_brute_force_definition(self, seed: int) -> None:
        dM = self._random_dM(seed)
        n = dM.shape[0]
        expected = np.zeros_like(dM)
        for i, j, k in itertools.product(range(n), repeat=3):
            expected[i, j, k] = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
        np.testing.assert_allclose(christoffel_first_kind(dM), expected, rtol=0, atol=1e-14)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_symmetric_in_last_two_indices(self, seed: int) -> None:
        c = christoffel_first_kind(self._random_dM(seed))
        np.testing.assert_allclose(c, c.transpose(0, 2, 1), rtol=0, atol=1e-14)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_metric_compatibility(self, seed: int) -> None:
        """c_ijk + c_jik == dM_ij/dq_k (the Koszul identity)."""
        dM = self._random_dM(seed)
        c = christoffel_first_kind(dM)
        np.testing.assert_allclose(c + c.transpose(1, 0, 2), dM, rtol=0, atol=1e-14)

    def test_rejects_non_cubic_tensor(self) -> None:
        with pytest.raises(ValueError):
            christoffel_first_kind(np.zeros((3, 3, 4)))

    def test_rejects_tensor_not_symmetric_in_first_two_indices(self) -> None:
        asymmetric = np.zeros((3, 3, 3))
        asymmetric[0, 1, 2] = 1.0
        with pytest.raises(ValueError):
            christoffel_first_kind(asymmetric)


class TestGolferChristoffelFromMassMatrix:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_golfer_dM_gives_metric_compatible_symbols(
        self, params: GolferParams, seed: int
    ) -> None:
        q, _ = _state(seed)
        dM = _mass_derivative(q, params)
        c = christoffel_first_kind(dM)
        np.testing.assert_allclose(c + c.transpose(1, 0, 2), dM, rtol=0, atol=1e-9)


class TestAnalyticalCoriolisAgainstLagrangian:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_euler_lagrange_reference(self, params: GolferParams, seed: int) -> None:
        q, qdot = _state(seed)
        reference = euler_lagrange_coriolis(q, qdot, params)
        np.testing.assert_allclose(
            analytical_coriolis(q, qdot, params), reference, rtol=1e-7, atol=1e-9
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_christoffel_contraction_of_central_dM(
        self, params: GolferParams, seed: int
    ) -> None:
        q, qdot = _state(seed)
        c = christoffel_first_kind(_mass_derivative(q, params))
        reference = np.einsum("ijk,j,k->i", c, qdot, qdot)
        np.testing.assert_allclose(
            analytical_coriolis(q, qdot, params), reference, rtol=1e-7, atol=1e-9
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_not_the_half_mdot_qdot_defect(self, params: GolferParams, seed: int) -> None:
        """Guard the exact #5640 failure mode: C must not collapse to 0.5 * Mdot qdot."""
        q, qdot = _state(seed)
        dM = _mass_derivative(q, params)
        half_mdot_qdot = 0.5 * np.einsum("ijk,k,j->i", dM, qdot, qdot)
        C = analytical_coriolis(q, qdot, params)
        assert np.linalg.norm(C - half_mdot_qdot) > 1e-3 * np.linalg.norm(C)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_skew_symmetry_of_mdot_minus_two_c(self, params: GolferParams, seed: int) -> None:
        """N = Mdot - 2 C(q, qdot) is skew, so qdot^T N qdot = 0 (power balance)."""
        q, qdot = _state(seed)
        dM = _mass_derivative(q, params)
        c = christoffel_first_kind(dM)
        mdot = np.einsum("ijk,k->ij", dM, qdot)
        n_matrix = mdot - 2.0 * np.einsum("ijk,k->ij", c, qdot)
        np.testing.assert_allclose(n_matrix + n_matrix.T, 0.0, atol=1e-9)
        # And the function under test inherits the same power balance.
        C = analytical_coriolis(q, qdot, params)
        scale = max(1.0, float(np.abs(qdot @ mdot @ qdot)))
        assert abs(qdot @ (mdot @ qdot - 2.0 * C)) < 1e-6 * scale

    def test_homogeneous_degree_two_in_qdot(self, params: GolferParams) -> None:
        q, qdot = _state(0)
        np.testing.assert_allclose(
            analytical_coriolis(q, 3.0 * qdot, params),
            9.0 * analytical_coriolis(q, qdot, params),
            rtol=1e-9,
            atol=1e-12,
        )


def _free_acceleration(
    coriolis: Callable[[np.ndarray, np.ndarray], np.ndarray], params: GolferParams
) -> Callable[[np.ndarray], np.ndarray]:
    """x' for the unconstrained, torque-free, undamped golfer on the active DOFs."""
    active = list(ACTIVE_DOFS)

    def rhs(x: np.ndarray) -> np.ndarray:
        q, qdot = x[:N_DOF], x[N_DOF:]
        residual = -coriolis(q, qdot) - analytical_gravity_vector(q, params)
        qddot = np.zeros(N_DOF)
        mass = analytical_mass_matrix(q, params)
        qddot[active] = np.linalg.solve(mass[np.ix_(active, active)], residual[active])
        return np.concatenate([qdot, qddot])

    return rhs


def _rk4(
    rhs: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, dt: float, steps: int
) -> Iterator[np.ndarray]:
    x = x0.copy()
    yield x
    for _ in range(steps):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        yield x


class TestFreeGolferDynamics:
    DT = 1e-3
    STEPS_ONE_SECOND = 1000

    def test_inert_dofs_have_zero_mass_rows(self, params: GolferParams) -> None:
        """Precondition of the reduced integration used below."""
        q, _ = _state(0)
        mass = analytical_mass_matrix(q, params)
        for i in INERT_DOFS:
            assert np.all(mass[i, :] == 0.0) and np.all(mass[:, i] == 0.0)
        eig = np.linalg.eigvalsh(mass[np.ix_(ACTIVE_DOFS, ACTIVE_DOFS)])
        assert eig.min() > 1e-3

    @pytest.mark.parametrize("seed", (0, 1, 2, 3))
    def test_energy_is_conserved_over_one_second(
        self, params: GolferParams, seed: int
    ) -> None:
        """Zero torque, zero damping, RK4 dt=1 ms: relative energy drift < 1e-6.

        Gravity is switched off (``g=0``).  With gravity the unconstrained, torque-free
        chain is a violent floppy pendulum (speeds of 1e2 rad/s within a second and
        near-singular mass matrices), where RK4 at 1 ms cannot hold 1e-6 for reasons
        unrelated to Coriolis.  At ``g=0`` the energy is purely kinetic and its
        conservation is exactly the power balance ``qdot^T (Mdot - 2C) qdot = 0``.

        Regression guard, not a detector for #5640 (see the module docstring).
        """
        free_params = dataclasses.replace(params, g=0.0)
        q, qdot = _state(seed, qdot_scale=1.0)
        x0 = np.concatenate([q, qdot])
        rhs = _free_acceleration(
            lambda a, b: analytical_coriolis(a, b, free_params), free_params
        )
        e0 = total_energy(x0, free_params)
        drift = max(
            abs(total_energy(x, free_params) - e0) / abs(e0)
            for x in _rk4(rhs, x0, self.DT, self.STEPS_ONE_SECOND)
        )
        assert drift < ENERGY_DRIFT_BOUND

    @pytest.mark.parametrize("seed", (1, 2, 3))
    def test_trajectory_matches_euler_lagrange_reference(
        self, params: GolferParams, seed: int
    ) -> None:
        """Same integrator, but Coriolis from the Lagrangian oracle: trajectories agree."""
        q, qdot = _state(seed)
        x0 = np.concatenate([q, qdot])
        steps = 250  # 0.25 s: long enough that a wrong C diverges by O(1) rad
        model = _free_acceleration(lambda a, b: analytical_coriolis(a, b, params), params)
        oracle = _free_acceleration(lambda a, b: euler_lagrange_coriolis(a, b, params), params)
        *_, x_model = _rk4(model, x0, self.DT, steps)
        *_, x_oracle = _rk4(oracle, x0, self.DT, steps)
        np.testing.assert_allclose(x_model, x_oracle, rtol=0, atol=1e-5)
