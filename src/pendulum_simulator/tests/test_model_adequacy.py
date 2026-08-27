"""Contract tests for the model-adequacy measurement.

These pin the central negative result of epic #4775: the two-link fixed-hub
model cannot be asked to release the club *and* keep the hands moving at
measured golfer speeds. If that ever stops being true, the model changed and the
conclusions built on it need re-deriving.

Closes #4779.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics import JointLimits, PendulumParams, TorqueClamp
from double_pendulum_golf.swing_objectives.downswing import DownswingConfig
from double_pendulum_golf.swing_objectives.model_adequacy import (
    hand_speed_frontier,
    swing_observables,
)
from double_pendulum_golf.swing_objectives.reference_kinematics import (
    score_against_reference,
)

_PARAMS = PendulumParams(m1=5.0, m2=0.30, L1=0.65, L2=1.10, mClub=0.20)


def _config() -> DownswingConfig:
    """The feasible baseline the frontier is measured from."""
    return DownswingConfig(
        params=_PARAMS,
        node_count=17,
        duration_s=0.36,
        initial_state=np.array([2.618, 1.745, 0.0, 0.0]),
        impact_theta1_rad=0.0,
        torque_clamp=TorqueClamp(max_torque1=250.0, max_torque2=20.0),
        joint_limits=JointLimits(
            phi_min=-0.175, phi_max=2.094, theta1_min=-4.0, theta1_max=4.0
        ),
    )


@pytest.fixture(scope="module")
def frontier():
    """One sweep reused across the module; each point is a full solve."""
    return hand_speed_frontier(_config(), floors_ms=[0.0, 3.0, 6.0])


def test_unconstrained_optimum_stops_the_hands(frontier) -> None:
    """The finding that started the epic, as a regression pin."""
    unconstrained = frontier.points[0]
    assert unconstrained.reachable
    assert unconstrained.hand_speed_ms < 1.0
    assert unconstrained.club_arm_rate_ratio > 20.0


def test_the_unconstrained_optimum_brakes_the_arms(frontier) -> None:
    """Hub torque reverses against the arms — the mechanism, not a side effect."""
    unconstrained = frontier.points[0]
    assert unconstrained.braking_fraction > 0.20
    assert unconstrained.peak_braking_torque_nm > 100.0


def test_raising_the_floor_costs_clubhead_speed(frontier) -> None:
    """Realism is not free: every m/s of hand speed is paid for in clubhead speed."""
    reachable = frontier.reachable_points
    assert len(reachable) >= 2
    speeds = [point.clubhead_speed_ms for point in reachable]
    assert all(b <= a + 1e-6 for a, b in zip(speeds[:-1], speeds[1:], strict=True))


def test_the_model_cannot_reach_measured_golfer_hand_speed(frontier) -> None:
    """The central negative result.

    Real golfers arrive at 6-9 m/s. Asked for that, the two-link fixed-hub model
    has no dynamically feasible answer, because releasing the club here *requires*
    reversing the hub torque and so decelerating the arms.
    """
    assert not frontier.reaches_measured_hand_speed
    assert frontier.max_reachable_hand_speed_ms < 6.0


def test_frontier_reports_one_point_per_floor_in_order() -> None:
    """Callers index the frontier positionally; order is part of the contract."""
    floors = [0.0, 2.0, 4.0]
    result = hand_speed_frontier(_config(), floors_ms=floors)
    assert [point.hand_speed_floor_ms for point in result.points] == floors


def test_frontier_rejects_an_empty_or_negative_sweep() -> None:
    """Contract: the sweep must be usable."""
    with pytest.raises(ValueError, match="at least one floor"):
        hand_speed_frontier(_config(), floors_ms=[])
    with pytest.raises(ValueError, match="non-negative"):
        hand_speed_frontier(_config(), floors_ms=[-1.0])


def test_swing_observables_match_the_reference_keys() -> None:
    """Every observable produced must be one the reference bands can score."""
    states = np.column_stack(
        [
            np.linspace(2.6, 0.0, 20),
            np.linspace(1.7, 0.0, 20),
            np.linspace(0.0, -12.0, 20),
            np.linspace(0.0, -25.0, 20),
        ]
    )
    speed = np.linspace(0.0, 48.0, 20)
    observables = swing_observables(states, speed, duration_s=0.27, arm_length_m=0.65)

    score = score_against_reference(observables)
    assert score.missing == ()
    assert observables["clubhead_speed_ms"] == pytest.approx(48.0)
    assert observables["downswing_time_s"] == pytest.approx(0.27)


def test_swing_observables_handle_a_stopped_arm_without_infinities() -> None:
    """A dead-stopped arm has no meaningful ratio; it must still be scoreable."""
    states = np.column_stack(
        [np.linspace(2.6, 0.0, 5), np.linspace(1.7, 0.0, 5), np.zeros(5), np.full(5, -20.0)]
    )
    observables = swing_observables(
        states, np.linspace(0.0, 40.0, 5), duration_s=0.3, arm_length_m=0.65
    )
    assert np.isfinite(observables["club_arm_rate_ratio"])
    score = score_against_reference(observables)
    assert not score.inside["club_arm_rate_ratio"]


def test_swing_observables_rejects_a_malformed_trajectory() -> None:
    """Contract: the reducer fails closed on the wrong shape."""
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        swing_observables(np.zeros((1, 4)), np.zeros(1), 0.3, 0.65)
