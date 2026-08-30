"""Contract tests for vectorized downswing signals.

The optimizer evaluates objectives at every collocation node inside a
finite-difference Jacobian, so the signals they consume are computed with array
operations rather than a per-sample Python loop (AGENTS.md 6a). Correctness is
therefore not self-evident: every quantity here is pinned against the scalar
authority in :mod:`double_pendulum_golf.physics`, exactly as the Python fallback
is pinned against the native Rust backend.

Closes #4768.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics import (
    PendulumParams,
    joint_velocities,
    linear_accelerations,
    net_joint_forces,
)
from double_pendulum_golf.swing_objectives.signals import (
    SwingSignals,
    build_swing_signals,
    generalized_accelerations,
)

_PARAMS = PendulumParams(m1=5.0, m2=0.30, L1=0.65, L2=1.10, mClub=0.20)
_SAMPLE_COUNT = 64
_MATCH_ATOL = 1e-9


def _trajectory(seed: int = 4768) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time, states, torques) spanning a plausible downswing envelope."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 0.28, _SAMPLE_COUNT)
    states = np.column_stack(
        [
            rng.uniform(-np.pi, np.pi, _SAMPLE_COUNT),
            rng.uniform(-0.2, 2.0, _SAMPLE_COUNT),
            rng.uniform(-30.0, 30.0, _SAMPLE_COUNT),
            rng.uniform(-45.0, 45.0, _SAMPLE_COUNT),
        ]
    )
    torques = rng.uniform(-150.0, 150.0, (_SAMPLE_COUNT, 2))
    return time, states, torques


def test_generalized_accelerations_match_the_scalar_equations_of_motion() -> None:
    """Vectorized q̈ must equal solving M q̈ = tau − C − G one sample at a time."""
    from double_pendulum_golf.physics import (
        coriolis_vector,
        gravity_vector,
        mass_matrix,
    )

    _, states, torques = _trajectory()
    vectorized = generalized_accelerations(states, torques, _PARAMS)

    for index, (state, torque) in enumerate(zip(states, torques, strict=True)):
        theta1, phi, dtheta1, dphi = state
        rhs = (
            torque
            - coriolis_vector(phi, dtheta1, dphi, _PARAMS)
            - gravity_vector(theta1, phi, _PARAMS)
        )
        expected = np.linalg.solve(mass_matrix(phi, _PARAMS), rhs)
        assert np.allclose(vectorized[index], expected, atol=_MATCH_ATOL)


def test_grip_force_matches_net_joint_forces() -> None:
    """Vectorized grip force must equal physics.net_joint_forces at the wrist."""
    time, states, torques = _trajectory(seed=7)
    signals = build_swing_signals(time, states, torques, _PARAMS)
    qddot = generalized_accelerations(states, torques, _PARAMS)

    for index, state in enumerate(states):
        expected = net_joint_forces(state, qddot[index], _PARAMS)["wrist"]
        assert np.allclose(signals.grip_force[index], expected, atol=_MATCH_ATOL)


def test_speeds_and_velocities_match_joint_velocities() -> None:
    """Tip speed and grip velocity must equal the scalar Jacobian results."""
    time, states, torques = _trajectory(seed=8)
    signals = build_swing_signals(time, states, torques, _PARAMS)

    for index, state in enumerate(states):
        expected = joint_velocities(state, _PARAMS)
        assert signals.clubhead_speed[index] == pytest.approx(
            expected["tip_speed"], abs=_MATCH_ATOL
        )
        assert np.allclose(
            signals.grip_velocity[index], expected["wrist_vel"], atol=_MATCH_ATOL
        )


def test_linear_accelerations_match_scalar_reference() -> None:
    """Tip acceleration must equal physics.linear_accelerations."""
    time, states, torques = _trajectory(seed=9)
    signals = build_swing_signals(time, states, torques, _PARAMS)
    qddot = generalized_accelerations(states, torques, _PARAMS)

    for index, state in enumerate(states):
        expected = linear_accelerations(state, qddot[index], _PARAMS)["tip"]
        assert np.allclose(signals.clubhead_acceleration[index], expected, atol=1e-8)


def test_signals_expose_consistent_shapes() -> None:
    """Every signal is aligned to the sample axis so objectives can integrate."""
    time, states, torques = _trajectory(seed=10)
    signals = build_swing_signals(time, states, torques, _PARAMS)

    assert isinstance(signals, SwingSignals)
    assert signals.sample_count == _SAMPLE_COUNT
    assert signals.time_s.shape == (_SAMPLE_COUNT,)
    assert signals.clubhead_speed.shape == (_SAMPLE_COUNT,)
    assert signals.grip_force.shape == (_SAMPLE_COUNT, 2)
    assert signals.grip_velocity.shape == (_SAMPLE_COUNT, 2)
    assert signals.grip_force_tangent.shape == (_SAMPLE_COUNT,)
    assert signals.centrifugal_wrist_moment.shape == (_SAMPLE_COUNT,)
    assert signals.coriolis_hub_power.shape == (_SAMPLE_COUNT,)


def test_hand_path_force_is_the_signed_projection_onto_hand_velocity() -> None:
    """The MacKenzie-style force channel excludes radial grip force."""
    time, states, torques = _trajectory(seed=11)
    signals = build_swing_signals(time, states, torques, _PARAMS)
    speed = np.linalg.norm(signals.grip_velocity, axis=1)
    expected = np.divide(
        signals.grip_force_power,
        speed,
        out=np.zeros_like(speed),
        where=speed > 1e-12,
    )

    assert np.allclose(signals.grip_force_tangent, expected)


def test_velocity_term_signals_agree_with_the_scalar_decomposition() -> None:
    """Bulk centrifugal/Coriolis signals must match the per-sample split (#4767)."""
    from double_pendulum_golf.swing_objectives.velocity_terms import (
        decompose_velocity_terms,
    )

    time, states, torques = _trajectory(seed=12)
    signals = build_swing_signals(time, states, torques, _PARAMS)

    for index, (_, phi, dtheta1, dphi) in enumerate(states):
        terms = decompose_velocity_terms(phi, dtheta1, dphi, _PARAMS)
        # Wrist centrifugal moment is reported as the left-hand-side magnitude.
        assert signals.centrifugal_wrist_moment[index] == pytest.approx(
            terms.centrifugal[1], abs=_MATCH_ATOL
        )
        # Coriolis hub power is the right-hand-side generalized force times omega1.
        expected_power = -terms.coriolis[0] * dtheta1
        assert signals.coriolis_hub_power[index] == pytest.approx(expected_power, abs=1e-7)


def test_rejects_misshapen_or_non_finite_inputs() -> None:
    """Contract: signal construction fails closed on malformed trajectories."""
    time, states, torques = _trajectory(seed=13)

    with pytest.raises(ValueError, match="shape"):
        build_swing_signals(time[:-1], states, torques, _PARAMS)
    with pytest.raises(ValueError, match="shape"):
        build_swing_signals(time, states[:, :3], torques, _PARAMS)
    with pytest.raises(ValueError, match="finite"):
        bad_states = states.copy()
        bad_states[0, 0] = np.nan
        build_swing_signals(time, bad_states, torques, _PARAMS)
    with pytest.raises(ValueError, match="increasing"):
        build_swing_signals(time[::-1], states, torques, _PARAMS)


def test_signals_are_immutable() -> None:
    """Reversibility: consumers must not be able to mutate shared signals."""
    time, states, torques = _trajectory(seed=14)
    signals = build_swing_signals(time, states, torques, _PARAMS)
    with pytest.raises((AttributeError, TypeError)):
        signals.clubhead_speed = np.zeros(_SAMPLE_COUNT)  # type: ignore[misc]
