"""Canonical distributed club convention for the double pendulum (Tools #5431).

Owner ruling: uniform shaft (mass m2, COM L2/2, Icm = m2*L2**2/12) plus a head
point mass ``mClub`` at L2. This module covers the inertia formula, the limiting
cases and the independent sympy Lagrangian oracle; energy, forces and transfer
diagnostics are in ``test_mass_convention_energy_forces.py`` and the shared-fixture
parity tests in ``test_mass_convention_fixtures.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from mass_convention_support import GOLF, REPRODUCER, params_cases, random_states

from double_pendulum_golf.club_inertia import distal_inertia
from double_pendulum_golf.physics import (
    PendulumParams,
    coriolis_vector,
    gravity_vector,
    kinetic_energy,
    mass_matrix,
    total_energy,
)
from double_pendulum_golf.transfer_strategy import (
    _club_kinetic_energy,
)

# --------------------------------------------------------------------- inertia


def test_distal_inertia_formula() -> None:
    inertia = distal_inertia(3.0, 1.0, 1.0)
    assert inertia.mass_kg == pytest.approx(4.0)
    assert inertia.first_moment_kg_m == pytest.approx(2.5)
    assert inertia.inertia_about_pivot_kg_m2 == pytest.approx(2.0)
    assert inertia.com_m == pytest.approx(0.625)
    assert inertia.icm_kg_m2 == pytest.approx(2.0 - 4.0 * 0.625**2)


def test_distal_inertia_shaft_only_is_uniform_rod() -> None:
    inertia = distal_inertia(2.0, 0.0, 1.5)
    assert inertia.com_m == pytest.approx(0.75)
    assert inertia.icm_kg_m2 == pytest.approx(2.0 * 1.5**2 / 12.0)


def test_distal_inertia_head_only_is_point_mass() -> None:
    inertia = distal_inertia(0.0, 0.7, 1.2)
    assert inertia.com_m == pytest.approx(1.2)
    assert inertia.icm_kg_m2 == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    ("shaft", "head", "length"),
    [(-1.0, 1.0, 1.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (float("nan"), 1.0, 1.0)],
)
def test_distal_inertia_rejects_invalid_inputs(
    shaft: float, head: float, length: float
) -> None:
    with pytest.raises(ValueError):
        distal_inertia(shaft, head, length)


# ------------------------------------------------------------- the #5431 defect


def test_issue_reproducer_dynamics_match_transfer_helper() -> None:
    state = np.array([0.0, 0.0, 0.0, 2.0])
    qdot = state[2:]
    dynamics_ke = 0.5 * qdot @ mass_matrix(0.0, REPRODUCER) @ qdot
    helper_ke = float(_club_kinetic_energy(state[None, :], REPRODUCER)[0])
    assert helper_ke == pytest.approx(4.0)
    assert dynamics_ke == pytest.approx(helper_ke)


def test_dynamics_ke_equals_helper_ke_for_moving_proximal_link() -> None:
    # With m1 contribution removed the arm is a point mass at the wrist, so the
    # club-only kinetic energy is KE(total) - KE(arm point mass).
    for params in params_cases():
        for state in random_states(5, 3):
            theta1_rate = state[2]
            arm_ke = 0.5 * params.m1 * (params.L1 * theta1_rate) ** 2
            total_ke = kinetic_energy(state, params)
            club_ke = float(_club_kinetic_energy(state[None, :], params)[0])
            assert total_ke - arm_ke == pytest.approx(club_ke, rel=1e-10, abs=1e-12)


# ---------------------------------------------------------------- limiting cases


def test_head_only_limit_is_uniform_rod_matrix() -> None:
    # mClub = 0: club is a uniform rod. M22 = m2 L2^2 / 3, coupling m2 L1 L2 / 2.
    params = PendulumParams(m1=2.0, m2=3.0, mClub=0.0, L1=0.7, L2=1.2)
    phi = 0.4
    mass = mass_matrix(phi, params)
    j_rod = 3.0 * 1.2**2 / 3.0
    s_rod = 3.0 * 1.2 / 2.0
    assert mass[1, 1] == pytest.approx(j_rod)
    assert mass[0, 1] == pytest.approx(j_rod + s_rod * 0.7 * np.cos(phi))
    assert mass[0, 0] == pytest.approx(
        (2.0 + 3.0) * 0.7**2 + j_rod + 2 * s_rod * 0.7 * np.cos(phi)
    )


def _lumped_reference(params: PendulumParams, state: np.ndarray) -> tuple:
    """The previous point-mass-at-tip model, kept inline as an oracle."""
    me = params.m2 + params.mClub
    theta1, phi, d1, d2 = state
    m11 = (params.m1 + me) * params.L1**2 + me * params.L2**2
    m11 += 2 * me * params.L1 * params.L2 * np.cos(phi)
    m12 = me * params.L2**2 + me * params.L1 * params.L2 * np.cos(phi)
    mass = np.array([[m11, m12], [m12, me * params.L2**2]])
    h = -me * params.L1 * params.L2 * np.sin(phi)
    coriolis = np.array([h * (2 * d1 * d2 + d2**2), -h * d1**2])
    g = params.g
    grav = np.array(
        [
            (params.m1 + me) * g * params.L1 * np.sin(theta1)
            + me * g * params.L2 * np.sin(theta1 + phi),
            me * g * params.L2 * np.sin(theta1 + phi),
        ]
    )
    return mass, coriolis, grav


def test_zero_shaft_mass_limit_recovers_point_mass_model() -> None:
    params = PendulumParams(m1=2.0, m2=1e-12, mClub=1.5, L1=0.7, L2=1.1)
    for state in random_states(6, 11):
        mass, coriolis, grav = _lumped_reference(params, state)
        np.testing.assert_allclose(mass_matrix(state[1], params), mass, rtol=1e-9)
        np.testing.assert_allclose(
            coriolis_vector(state[1], state[2], state[3], params),
            coriolis,
            rtol=1e-9,
            atol=1e-12,
        )
        np.testing.assert_allclose(gravity_vector(state[0], state[1], params), grav, rtol=1e-9)


@pytest.mark.parametrize("params", params_cases())
def test_mass_matrix_is_symmetric_positive_definite(params: PendulumParams) -> None:
    for phi in np.linspace(-np.pi, np.pi, 9):
        mass = mass_matrix(float(phi), params)
        np.testing.assert_allclose(mass, mass.T)
        assert np.all(np.linalg.eigvalsh(mass) > 0.0)


# ------------------------------------------------------ independent derivation


def _sympy_model(params: PendulumParams):  # noqa: ANN202
    sp = pytest.importorskip("sympy")
    t = sp.symbols("t")
    th = sp.Function("th")(t)
    ph = sp.Function("ph")(t)
    wrist = sp.Matrix([params.L1 * sp.sin(th), -params.L1 * sp.cos(th)])
    angle2 = th + ph
    direction = sp.Matrix([sp.sin(angle2), -sp.cos(angle2)])
    com = wrist + params.L2 / 2 * direction
    tip = wrist + params.L2 * direction
    rate = sp.diff
    kinetic = (
        params.m1 * rate(wrist, t).dot(rate(wrist, t)) / 2
        + params.m2 * rate(com, t).dot(rate(com, t)) / 2
        + params.m2 * params.L2**2 / 12 * rate(angle2, t) ** 2 / 2
        + params.mClub * rate(tip, t).dot(rate(tip, t)) / 2
    )
    potential = params.g * (params.m1 * wrist[1] + params.m2 * com[1] + params.mClub * tip[1])
    q = [th, ph]
    qd = [rate(x, t) for x in q]
    th_s, ph_s, d1_s, d2_s = sp.symbols("th_s ph_s d1_s d2_s")
    subs = {rate(th, t): d1_s, rate(ph, t): d2_s}
    mass = sp.Matrix(2, 2, lambda i, j: sp.diff(kinetic, qd[i], qd[j]))
    # Velocity-product term from the Euler-Lagrange equations with qdd = 0:
    # C qd = d/dt(dT/dqd) |_(qdd=0) - dT/dq
    bias = sp.Matrix(
        [
            sum(sp.diff(sp.diff(kinetic, qd[i]), q[j]) * qd[j] for j in range(2))
            - sp.diff(kinetic, q[i])
            for i in range(2)
        ]
    )
    grav = sp.Matrix([sp.diff(potential, x) for x in q])

    def lambdify(expr):  # noqa: ANN001, ANN202
        e = expr.subs(subs).subs({th: th_s, ph: ph_s})
        return sp.lambdify((th_s, ph_s, d1_s, d2_s), e, "numpy")

    return (
        lambdify(mass),
        lambdify(bias),
        lambdify(grav),
        lambdify(sp.Matrix([kinetic + potential])),
    )


def test_dynamics_match_independent_sympy_lagrangian() -> None:
    for params in (REPRODUCER, GOLF):
        mass_fn, bias_fn, grav_fn, energy_fn = _sympy_model(params)
        for state in random_states(10, 5):
            th, ph, d1, d2 = state
            np.testing.assert_allclose(
                mass_matrix(ph, params),
                np.array(mass_fn(th, ph, d1, d2), dtype=float),
                rtol=1e-10,
            )
            np.testing.assert_allclose(
                coriolis_vector(ph, d1, d2, params),
                np.array(bias_fn(th, ph, d1, d2), dtype=float).ravel(),
                rtol=1e-9,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                gravity_vector(th, ph, params),
                np.array(grav_fn(th, ph, d1, d2), dtype=float).ravel(),
                rtol=1e-10,
            )
            assert total_energy(state, params) == pytest.approx(
                float(np.array(energy_fn(th, ph, d1, d2), dtype=float).ravel()[0]), rel=1e-10
            )
