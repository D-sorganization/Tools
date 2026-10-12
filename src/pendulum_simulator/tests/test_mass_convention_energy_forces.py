"""Energy, force and transfer-diagnostic checks of the distributed club (Tools #5431)."""

from __future__ import annotations

import numpy as np
import pytest
from mass_convention_support import GOLF, REPRODUCER, random_states

from double_pendulum_golf.club_inertia import distal_inertia
from double_pendulum_golf.physics import (
    PendulumParams,
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
    for state in random_states(6, 21):
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
    for state in random_states(6, 33):
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
