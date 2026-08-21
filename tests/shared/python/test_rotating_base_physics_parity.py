"""Numerical parity with the qualified UpstreamDrift rotating-base source."""

from __future__ import annotations

import numpy as np

from shared.python.swing_sim.rotating_base import (
    UPSTREAM_PHYSICS_SOURCE_SHA256,
    RotatingBaseConfig,
    RotatingBaseParams,
    TorsoTwoHandControl,
    initial_state,
    rollout,
    solve_constrained_dynamics,
)

EXPECTED_SOURCE_SHA256 = (
    "a08641054a5ec58eaa9023ff123464c960b10833826f7ac9ba8dea68c26ab0d0"
)


def _control() -> TorsoTwoHandControl:
    return TorsoTwoHandControl(
        torso_nm=55.0,
        lead_arm_nm=7.0,
        trail_arm_nm=7.0,
        lead_wrist_nm=-3.0,
        trail_wrist_nm=-3.0,
    )


def test_exact_source_identity_and_initial_constrained_solution() -> None:
    params = RotatingBaseParams.publication_default()
    state = initial_state(params, torso_rate_rad_s=3.5, club_rate_rad_s=4.5)
    solution = solve_constrained_dynamics(state, _control(), params)

    assert UPSTREAM_PHYSICS_SOURCE_SHA256 == EXPECTED_SOURCE_SHA256
    assert np.allclose(
        state.q,
        [
            0.0,
            -0.20300436066188554,
            0.20300436066188554,
            0.0,
            -0.6072684743999148,
            np.pi / 2.0,
            0.0,
        ],
        atol=1e-14,
        rtol=0.0,
    )
    assert np.allclose(
        solution.qddot,
        [
            5.129635154596906,
            30.41179028255938,
            30.73718178290898,
            21.681987203220224,
            5.413123928832858,
            -53.667453369891916,
            79.96538290173758,
        ],
        atol=1e-11,
        rtol=0.0,
    )
    assert np.allclose(
        solution.force_on_club_n,
        [
            [9.294019966910195, 43.437638894042514],
            [-4.366750725596754, -43.55895184093693],
        ],
        atol=1e-11,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        solution.force_generated_couple_nm, 5.654778397773663, atol=1e-12, rtol=0.0
    )


def test_short_rollout_reproduces_speed_and_closure_oracle() -> None:
    params = RotatingBaseParams.publication_default()
    state = initial_state(params, torso_rate_rad_s=3.5, club_rate_rad_s=4.5)
    control = _control()
    trace = rollout(
        state,
        lambda _time, _state: control,
        params,
        RotatingBaseConfig(duration_s=0.004, step_s=0.002),
    )

    assert np.allclose(
        trace.clubhead_speed_m_s,
        [4.850244273229958, 4.841655924448833, 4.827486889925504],
        atol=1e-11,
        rtol=0.0,
    )
    assert abs(trace.work_energy_closure_j - 0.0011196196284468662) < 1e-12
    assert np.max(trace.position_constraint_norm_m) < 2e-12
    assert np.max(trace.velocity_constraint_norm_m_s) < 1e-14
