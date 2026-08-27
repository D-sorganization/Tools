"""Contract tests for the Hill-type joint actuation model.

Epic #4775 established that the optimizer stops the hands because a
point-mass-clubhead model makes that optimal. The fix has to make the *golfer*
unable to do it, which means two things the shipped symmetric torque clamp does
not express:

* torque capacity falls as the joint speeds up (Hill 1938), so the arms cannot
  be driven indefinitely; and
* the muscles that decelerate the arms are not the ones that drive them, and are
  far weaker, so braking is not free.

Closes #4777.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.swing_objectives.actuation import (
    JointActuation,
    SwingActuation,
    tour_hub_actuation,
    tour_wrist_actuation,
)

_HUB = tour_hub_actuation()


def test_isometric_capacity_is_the_peak_torque() -> None:
    """At zero joint rate the driving limit is the isometric peak."""
    assert _HUB.driving_limit(0.0) == pytest.approx(_HUB.peak_torque_nm)


def test_driving_capacity_falls_monotonically_with_joint_rate() -> None:
    """Hill's hyperbola: the faster the joint moves, the less torque is available."""
    rates = np.linspace(0.0, _HUB.max_rate_rad_s, 40)
    limits = [_HUB.driving_limit(r) for r in rates]
    assert all(b <= a + 1e-12 for a, b in zip(limits[:-1], limits[1:], strict=True))
    assert limits[0] > limits[-1]


def test_driving_capacity_vanishes_at_the_maximum_rate() -> None:
    """The zero-torque velocity is where the joint can no longer contribute."""
    assert _HUB.driving_limit(_HUB.max_rate_rad_s) == pytest.approx(0.0, abs=1e-9)
    assert _HUB.driving_limit(_HUB.max_rate_rad_s * 2.0) == pytest.approx(0.0)


def test_driving_capacity_is_never_negative() -> None:
    """Past the zero-torque velocity the joint contributes nothing, not thrust."""
    for rate in (0.0, 10.0, 29.9, 30.0, 60.0, 500.0):
        assert _HUB.driving_limit(rate) >= 0.0


def test_braking_capacity_is_a_small_fraction_of_driving() -> None:
    """The antagonists that stop the arms are much weaker than the prime movers.

    This asymmetry is the whole point: with the shipped symmetric clamp,
    braking the arms costs exactly what driving them costs, and the optimizer
    spends 32% of the downswing braking.
    """
    assert _HUB.braking_limit(0.0) < 0.5 * _HUB.peak_torque_nm
    assert _HUB.braking_limit(0.0) > 0.0


def test_eccentric_gain_raises_the_braking_limit_above_its_isometric_value() -> None:
    """Lengthening muscle is stronger than isometric, so braking gets a boost."""
    ungained = JointActuation(
        peak_torque_nm=_HUB.peak_torque_nm,
        max_rate_rad_s=_HUB.max_rate_rad_s,
        curvature=_HUB.curvature,
        brake_fraction=_HUB.brake_fraction,
        eccentric_gain=1.0,
    )
    assert _HUB.braking_limit(5.0) > ungained.braking_limit(5.0)


def test_torque_bounds_follow_the_direction_of_motion() -> None:
    """Driving capacity applies with the motion, braking capacity against it."""
    low, high = _HUB.torque_bounds(-12.0)  # downswing: arm rate is negative
    assert low == pytest.approx(-_HUB.driving_limit(12.0))
    assert high == pytest.approx(_HUB.braking_limit(12.0))

    low, high = _HUB.torque_bounds(+12.0)
    assert low == pytest.approx(-_HUB.braking_limit(12.0))
    assert high == pytest.approx(_HUB.driving_limit(12.0))


def test_bounds_are_symmetric_at_rest() -> None:
    """At zero rate there is no direction of motion to be with or against."""
    low, high = _HUB.torque_bounds(0.0)
    assert low == pytest.approx(-high)


def test_margins_are_positive_exactly_when_a_torque_is_admissible() -> None:
    """The optimizer consumes these as inequality constraints."""
    rate = -12.0
    low, high = _HUB.torque_bounds(rate)
    inside = 0.5 * (low + high)
    assert min(_HUB.margins(rate, inside)) > 0.0
    assert min(_HUB.margins(rate, low - 1.0)) < 0.0
    assert min(_HUB.margins(rate, high + 1.0)) < 0.0


def test_vectorized_margins_match_the_scalar_path() -> None:
    """Batched evaluation must agree with the per-sample contract."""
    rates = np.linspace(-25.0, 25.0, 21)
    torques = np.linspace(-200.0, 200.0, 21)
    batched = _HUB.batch_margins(rates, torques)
    assert batched.shape == (21, 2)
    for i, (rate, torque) in enumerate(zip(rates, torques, strict=True)):
        assert np.allclose(batched[i], _HUB.margins(float(rate), float(torque)))


def test_swing_actuation_bundles_both_joints() -> None:
    """A swing needs a hub and a wrist limit, and reports both margins."""
    actuation = SwingActuation(hub=tour_hub_actuation(), wrist=tour_wrist_actuation())
    rates = np.zeros((5, 2))
    torques = np.zeros((5, 2))
    margins = actuation.batch_margins(rates, torques)
    assert margins.shape == (5, 4)
    assert np.all(margins > 0.0)


def test_wrist_capacity_is_far_below_the_hub() -> None:
    """The actuation asymmetry that makes the release passive rather than driven."""
    assert tour_wrist_actuation().peak_torque_nm < 0.2 * _HUB.peak_torque_nm


def test_rejects_unphysical_configuration() -> None:
    """Contract: every parameter has a physically meaningful range."""
    good = dict(
        peak_torque_nm=200.0,
        max_rate_rad_s=30.0,
        curvature=4.0,
        brake_fraction=0.3,
        eccentric_gain=1.3,
    )
    for field, bad in (
        ("peak_torque_nm", 0.0),
        ("max_rate_rad_s", 0.0),
        ("curvature", -1.0),
        ("brake_fraction", 0.0),
        ("brake_fraction", 1.5),
        ("eccentric_gain", 0.5),
    ):
        with pytest.raises(ValueError, match=field):
            JointActuation(**{**good, field: bad})


def test_rejects_non_finite_queries() -> None:
    """Non-finite rates must fail closed rather than produce a silent bound."""
    with pytest.raises(ValueError, match="finite"):
        _HUB.driving_limit(np.nan)
    with pytest.raises(ValueError, match="finite"):
        _HUB.torque_bounds(np.inf)


def test_actuation_is_immutable() -> None:
    """Reversibility: a configuration handed to the optimizer cannot be mutated."""
    with pytest.raises((AttributeError, TypeError)):
        _HUB.peak_torque_nm = 1.0  # type: ignore[misc]
