"""Canonical distributed club convention for the double pendulum (Tools #5431).

Owner ruling: uniform shaft (mass m2, COM L2/2, Icm = m2*L2**2/12) plus a head
point mass ``mClub`` at L2. Dynamics, forces, energies and transfer diagnostics
must all agree with it. The oracle is an independent point-by-point Lagrangian
(sympy), finite-difference accelerations, and energy conservation.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from double_pendulum_golf.club_inertia import DistalSegmentInertia, distal_inertia
from double_pendulum_golf.physics import (
    PendulumParams,
    coriolis_vector,
    equations_of_motion,
    gravity_vector,
    kinetic_energy,
    mass_matrix,
    net_joint_forces,
    potential_energy,
    total_energy,
)
from double_pendulum_golf.simulation import run_simulation
from double_pendulum_golf.transfer_strategy import (
    _club_kinetic_energy,
    double_pendulum_force_attribution,
    double_pendulum_transfer_signals,
    summarize_transfer,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "distal_inertia_cases.json"
REPRODUCER = PendulumParams(m1=2.0, m2=3.0, mClub=1.0, L1=0.7, L2=1.0)
GOLF = PendulumParams(m1=5.0, m2=0.108, mClub=0.202, L1=0.65, L2=1.10, b1=0.1, b2=0.05)


def _random_states(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.0, 2.0, size=(count, 4))


def _params_cases() -> list[PendulumParams]:
    rng = np.random.default_rng(7)
    return [
        PendulumParams(
            m1=float(rng.uniform(1, 6)),
            m2=float(rng.uniform(0.05, 3)),
            mClub=float(rng.uniform(0, 1.5)),
            L1=float(rng.uniform(0.3, 1)),
            L2=float(rng.uniform(0.5, 1.5)),
        )
        for _ in range(12)
    ]


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


def test_distal_inertia_matches_shared_fixture() -> None:
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    assert cases
    for case in cases:
        got = distal_inertia(case["shaft_mass_kg"], case["head_mass_kg"], case["length_m"])
        assert isinstance(got, DistalSegmentInertia)
        assert got.mass_kg == pytest.approx(case["mass_kg"], rel=1e-14)
        assert got.first_moment_kg_m == pytest.approx(case["first_moment_kg_m"], rel=1e-14)
        assert got.inertia_about_pivot_kg_m2 == pytest.approx(
            case["inertia_about_pivot_kg_m2"], rel=1e-14
        )
        assert got.centroidal_inertia_kg_m2 == pytest.approx(
            case["centroidal_inertia_kg_m2"], rel=1e-14, abs=1e-300
        )


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
    for params in _params_cases():
        for state in _random_states(5, 3):
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
    for state in _random_states(6, 11):
        mass, coriolis, grav = _lumped_reference(params, state)
        np.testing.assert_allclose(mass_matrix(state[1], params), mass, rtol=1e-9)
        np.testing.assert_allclose(
            coriolis_vector(state[1], state[2], state[3], params),
            coriolis,
            rtol=1e-9,
            atol=1e-12,
        )
        np.testing.assert_allclose(gravity_vector(state[0], state[1], params), grav, rtol=1e-9)


@pytest.mark.parametrize("params", _params_cases())
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
        for state in _random_states(10, 5):
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


# ---------------------------------------------------------------------- energy


def _passive_run(params: PendulumParams, t_end: float, dt: float):  # noqa: ANN202
    return run_simulation(
        params, np.array([1.0, 0.5, 0.3, -0.2]), t_end, lambda _t: (0.0, 0.0), dt=dt
    )


def test_energy_conserved_for_undamped_passive_swing() -> None:
    result = _passive_run(REPRODUCER, 5.0, 0.002)
    energies = np.array([total_energy(s, REPRODUCER) for s in result.states])
    assert np.max(np.abs(energies - energies[0])) < 1e-6
    assert potential_energy(result.states[0], REPRODUCER) + kinetic_energy(
        result.states[0], REPRODUCER
    ) == pytest.approx(energies[0])


def test_energy_balance_with_torque_and_damping() -> None:
    params = GOLF

    def torque(t: float) -> tuple[float, float]:
        return (20.0 * np.cos(3 * t), 2.0 * np.sin(5 * t))

    result = run_simulation(params, np.array([0.3, 0.2, 0.0, 0.0]), 0.6, torque, dt=0.0005)
    time = result.t
    dth, dph = result.states[:, 2], result.states[:, 3]
    tau = np.array([torque(t) for t in time])
    power = tau[:, 0] * dth + tau[:, 1] * dph - params.b1 * dth**2 - params.b2 * dph**2
    work = np.trapezoid(power, time)
    delta = total_energy(result.states[-1], params) - total_energy(result.states[0], params)
    assert delta == pytest.approx(work, abs=2e-4)


def test_energy_balance_residual_converges_with_step() -> None:
    def residual(dt: float) -> float:
        def torque(t: float) -> tuple[float, float]:
            return (15.0 * np.cos(4 * t), 1.5 * np.sin(6 * t))

        result = run_simulation(GOLF, np.array([0.3, 0.2, 0.0, 0.0]), 0.5, torque, dt=dt)
        tau = np.array([torque(t) for t in result.t])
        dth, dph = result.states[:, 2], result.states[:, 3]
        power = tau[:, 0] * dth + tau[:, 1] * dph - GOLF.b1 * dth**2 - GOLF.b2 * dph**2
        delta = total_energy(result.states[-1], GOLF) - total_energy(result.states[0], GOLF)
        return abs(delta - float(np.trapezoid(power, result.t)))

    coarse, fine = residual(0.004), residual(0.002)
    assert fine < coarse
    assert coarse / max(fine, 1e-18) > 2.5  # trapezoid is second order


# ----------------------------------------------------------------------- forces


def _positions(q: np.ndarray, params: PendulumParams) -> dict[str, np.ndarray]:
    theta1, phi = q
    wrist = params.L1 * np.array([np.sin(theta1), -np.cos(theta1)])
    direction = np.array([np.sin(theta1 + phi), -np.cos(theta1 + phi)])
    return {
        "wrist": wrist,
        "shaft_com": wrist + 0.5 * params.L2 * direction,
        "tip": wrist + params.L2 * direction,
    }


def _fd_accelerations(state: np.ndarray, qddot: np.ndarray, params: PendulumParams) -> dict:
    step = 1e-4
    q0, qd0 = state[:2], state[2:]

    def path(time: float) -> np.ndarray:
        return q0 + qd0 * time + 0.5 * qddot * time**2

    plus, mid, minus = (_positions(path(x), params) for x in (step, 0.0, -step))
    return {k: (plus[k] - 2 * mid[k] + minus[k]) / step**2 for k in mid}


@pytest.mark.parametrize("params", [REPRODUCER, GOLF])
def test_net_joint_forces_match_finite_difference_newton_euler(
    params: PendulumParams,
) -> None:
    gravity = np.array([0.0, -params.g])
    for state in _random_states(6, 21):
        qddot = np.array([1.3, -2.1])
        acc = _fd_accelerations(state, qddot, params)
        wrist_force = params.m2 * (acc["shaft_com"] - gravity) + params.mClub * (
            acc["tip"] - gravity
        )
        shoulder_force = wrist_force + params.m1 * (acc["wrist"] - gravity)
        forces = net_joint_forces(state, qddot, params)
        np.testing.assert_allclose(forces["wrist"], wrist_force, rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(forces["shoulder"], shoulder_force, rtol=1e-6, atol=1e-5)


def test_static_hang_forces_carry_the_weight() -> None:
    params = REPRODUCER
    forces = net_joint_forces(np.zeros(4), np.zeros(2), params)
    assert forces["wrist"][1] == pytest.approx((params.m2 + params.mClub) * params.g)
    assert forces["shoulder"][1] == pytest.approx(
        (params.m1 + params.m2 + params.mClub) * params.g
    )


def test_wrist_force_closes_the_club_rotational_equation() -> None:
    # Newton-Euler about the club COM: tau2 + moment of F_wrist at the wrist
    # equals Icm * club angular acceleration. Catches a wrong centroidal inertia.
    no_loss = PendulumParams(m1=5.0, m2=0.108, mClub=0.202, L1=0.65, L2=1.10)
    torque = (4.0, 1.5)
    for state in _random_states(6, 33):
        accel = equations_of_motion(state, 0.0, no_loss, lambda _t: torque)[2:]
        inertia = distal_inertia(no_loss.m2, no_loss.mClub, no_loss.L2)
        forces = net_joint_forces(state, accel, no_loss)
        angle = state[0] + state[1]
        com_offset = inertia.com_m * np.array([np.sin(angle), -np.cos(angle)])
        # ``forces['wrist']`` is the force the arm exerts on the club at the wrist.
        pivot_force = np.array(forces["wrist"])
        moment = (-com_offset[0]) * pivot_force[1] - (-com_offset[1]) * pivot_force[0]
        club_alpha = accel[0] + accel[1]
        assert torque[1] + moment == pytest.approx(inertia.icm_kg_m2 * club_alpha, abs=1e-8)


# --------------------------------------------------------- transfer diagnostics


def _driven_result(dt: float):  # noqa: ANN202
    def torque(t: float) -> tuple[float, float]:
        return (30.0 * np.cos(2 * t) - 5.0, 3.0 * np.sin(7 * t))

    return run_simulation(
        PendulumParams(m1=5.0, m2=0.108, mClub=0.202, L1=0.65, L2=1.10, b1=0.2, b2=0.1),
        np.array([2.0, 1.0, 0.0, 0.0]),
        0.5,
        torque,
        dt=dt,
    )


def test_club_energy_balance_closes_and_converges() -> None:
    def residual(dt: float) -> float:
        signals = double_pendulum_transfer_signals(_driven_result(dt))
        summary = summarize_transfer(signals, start_s=0.0, end_s=0.45)
        assert summary.club_energy_balance_residual_j is not None
        return abs(summary.club_energy_balance_residual_j)

    coarse, fine = residual(0.004), residual(0.002)
    assert fine < 0.05
    assert fine < coarse


def test_additive_closure_is_an_identity_but_energy_balance_is_not() -> None:
    result = _driven_result(0.002)
    signals = double_pendulum_transfer_signals(result)
    good = summarize_transfer(signals, start_s=0.0, end_s=0.45)
    assert abs(good.work_closure_residual_j) < 1e-9
    # Corrupt the gravity power the club sees: additive closure stays zero,
    # the real energy balance must notice.
    from dataclasses import replace

    assert signals.club_gravity_power_w is not None
    bad_signals = replace(signals, club_gravity_power_w=signals.club_gravity_power_w * 1.2)
    bad = summarize_transfer(bad_signals, start_s=0.0, end_s=0.45)
    assert abs(bad.work_closure_residual_j) < 1e-9
    assert abs(bad.club_energy_balance_residual_j) > 10 * abs(
        good.club_energy_balance_residual_j
    )


def test_distal_energy_signal_uses_shared_inertia() -> None:
    result = _driven_result(0.005)
    signals = double_pendulum_transfer_signals(result)
    np.testing.assert_allclose(
        signals.distal_kinetic_energy_j,
        _club_kinetic_energy(np.asarray(result.states), result.params),
    )
    # Club KE from the dynamics' own mass matrix minus the arm point mass.
    for index in (0, 40, 99):
        state = result.states[index]
        arm = 0.5 * result.params.m1 * (result.params.L1 * state[2]) ** 2
        assert signals.distal_kinetic_energy_j[index] == pytest.approx(
            kinetic_energy(state, result.params) - arm, rel=1e-10, abs=1e-12
        )


def test_pair_power_equals_club_plus_arm_interface_power() -> None:
    result = _driven_result(0.005)
    for index in (0, 10, 60):
        state = result.states[index]
        tau2 = result.torques_at(index)[1]
        pair = tau2 * state[3]
        club_side = tau2 * (state[2] + state[3])
        arm_side = -tau2 * state[2]
        assert pair == pytest.approx(club_side + arm_side)


def test_force_attribution_provider_uses_canonical_segment_inertia() -> None:
    result = run_simulation(
        PendulumParams(m1=5.0, m2=0.108, mClub=0.202, L1=0.65, L2=1.10),
        np.array([1.0, 0.5, 0.2, -0.1]),
        0.1,
        lambda _t: (10.0, 1.0),
        dt=0.01,
    )
    attribution = double_pendulum_force_attribution(result)
    assert attribution is not None
    # Generalized accelerations from physics and from the swing_sim kernel agree.
    from shared.python.swing_sim.force_attribution_providers import (
        DoublePendulumAttributionProvider as Provider,
    )

    params = result.params
    inertia = distal_inertia(params.m2, params.mClub, params.L2)
    from shared.python.swing_sim import PendulumParameters

    canonical = PendulumParameters(
        m1=params.m1,
        l1=params.L1,
        lc1=params.L1,
        i1=params.m1 * params.L1**2,
        m2=inertia.mass_kg,
        l2=params.L2,
        lc2=inertia.com_m,
        i2=inertia.inertia_about_pivot_kg_m2,
        d1=params.b1,
        d2=params.b2,
    )
    provider = Provider(canonical, g_inplane=(0.0, -params.g))
    state = result.states[3]
    np.testing.assert_allclose(
        provider.mass_matrix(state[:2]), mass_matrix(state[1], params), rtol=1e-12
    )
    np.testing.assert_allclose(
        provider.gravity(state[:2]), gravity_vector(state[0], state[1], params), rtol=1e-12
    )


# ----------------------------------------------------- numeric stability (Icm)


def test_centroidal_inertia_never_cancels_below_zero() -> None:
    """Icm is a closed form, not ``Jw - s*com`` (catastrophic cancellation).

    Swept over L2 in [0.5, 1.5] and m2 log-uniform in [1e-15, 1] with a head
    mass from 0 to 1 kg; the parallel-axis sum is the reference.
    """
    rng = np.random.default_rng(5431)
    for _ in range(4000):
        length = rng.uniform(0.5, 1.5)
        shaft = 10.0 ** rng.uniform(-15.0, 0.0)
        head = rng.choice([0.0, rng.uniform(0.0, 1.0)])
        if shaft + head <= 0.0:
            continue
        club = distal_inertia(shaft, head, length)
        com = club.com_m
        reference = (
            shaft * length**2 / 12.0
            + shaft * (length / 2.0 - com) ** 2
            + head * (length - com) ** 2
        )
        assert club.icm_kg_m2 >= 0.0
        assert club.icm_kg_m2 == pytest.approx(reference, rel=1e-9, abs=1e-30)


def test_impact_optimality_accepts_the_vanishing_shaft_limit() -> None:
    from double_pendulum_golf.swing_objectives.impact_optimality import (
        impact_hand_speed_coefficient,
    )

    rng = np.random.default_rng(1)
    for length in rng.uniform(0.5, 1.5, 200):
        params = PendulumParams(m1=5.0, m2=1e-15, L1=0.65, L2=float(length), mClub=0.5)
        assert abs(impact_hand_speed_coefficient(params)) < 1e-12


# ------------------------------------------------- shared dynamics golden file


def test_dynamics_match_shared_golden_cases() -> None:
    path = FIXTURE.with_name("distal_dynamics_cases.json")
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 5
    for c in cases:
        params = PendulumParams(
            m1=c["m1"],
            m2=c["shaft_mass_kg"],
            mClub=c["head_mass_kg"],
            L1=c["l1"],
            L2=c["l2"],
            g=c["g"],
        )
        mass = mass_matrix(c["phi"], params)
        assert mass[0, 0] == pytest.approx(c["m11"], rel=1e-12)
        assert mass[0, 1] == pytest.approx(c["m12"], rel=1e-12)
        assert mass[1, 1] == pytest.approx(c["m22"], rel=1e-12)
        coriolis = coriolis_vector(c["phi"], c["dtheta1"], c["dphi"], params)
        assert coriolis[0] == pytest.approx(c["c1"], rel=1e-12)
        assert coriolis[1] == pytest.approx(c["c2"], rel=1e-12)
        gravity = gravity_vector(c["theta1"], c["phi"], params)
        assert gravity[0] == pytest.approx(c["g1"], rel=1e-12)
        assert gravity[1] == pytest.approx(c["g2"], rel=1e-12)
        state = np.array([c["theta1"], c["phi"], c["dtheta1"], c["dphi"]])
        assert potential_energy(state, params) == pytest.approx(
            c["potential_energy_j"], rel=1e-12
        )


def test_driver_preset_fixture_matches_the_derived_split() -> None:
    """The web presets are pinned to this file; it must equal the derivation."""
    from double_pendulum_golf.swing_objectives.club_equivalence import (
        DRIVER_SPEC,
        distributed_club_split,
    )

    fixture = json.loads(FIXTURE.with_name("driver_preset.json").read_text(encoding="utf-8"))
    shaft, head = distributed_club_split(DRIVER_SPEC, fixture["length_m"])
    assert fixture["shaft_mass_kg"] == pytest.approx(shaft, rel=1e-14)
    assert fixture["head_mass_kg"] == pytest.approx(head, rel=1e-14)


def test_gui_golf_presets_use_the_derived_split() -> None:
    pytest.importorskip("PyQt6")
    from double_pendulum_golf.gui.controls_widget import ControlsWidget
    from double_pendulum_golf.swing_objectives.club_equivalence import (
        DRIVER_SPEC,
        distributed_club_split,
    )

    shaft, head = distributed_club_split(DRIVER_SPEC, 1.10)
    for name in ("Golf Swing (passive wrist)", "Golf Swing (active wrist)"):
        preset = ControlsWidget.PRESETS[name]
        assert preset[8] == pytest.approx(shaft, rel=1e-14)
        assert preset[9] == pytest.approx(head, rel=1e-14)
