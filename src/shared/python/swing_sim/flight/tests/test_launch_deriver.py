"""Launch-deriver round-trip tests (post-impact vectors -> LaunchConditions)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.swing_sim.flight import (
    DEFAULT_BACKSPIN_AXIS,
    derive_launch_conditions,
)
from shared.python.swing_sim.flight._constants import RPM_TO_RAD_S


@pytest.mark.unit
def test_derive_recovers_speed_angles_and_spin() -> None:
    vel = np.array([60.0, 5.0, 15.0])
    spin = 300.0 * np.array([0.0, -1.0, 0.0])  # pure backspin, 300 rad/s

    launch = derive_launch_conditions(vel, spin)

    assert launch.ball_speed == pytest.approx(float(np.linalg.norm(vel)))
    horiz = math.hypot(vel[0], vel[1])
    assert launch.launch_angle == pytest.approx(math.atan2(vel[2], horiz))
    assert launch.azimuth_angle == pytest.approx(math.atan2(vel[1], vel[0]))
    assert launch.spin_rate == pytest.approx(300.0 / RPM_TO_RAD_S)
    assert launch.spin_axis == pytest.approx((0.0, -1.0, 0.0))


@pytest.mark.unit
def test_derived_conditions_round_trip_through_launch_vectors() -> None:
    """get_initial_velocity / get_spin_vector reproduce the input vectors."""
    vel = np.array([55.0, -8.0, 12.0])
    spin = np.array([30.0, -280.0, 45.0])

    launch = derive_launch_conditions(vel, spin)

    np.testing.assert_allclose(launch.get_initial_velocity(), vel, atol=1e-9)
    np.testing.assert_allclose(launch.get_spin_vector(), spin, atol=1e-9)


@pytest.mark.unit
def test_vertical_launch_pins_angle_to_pi_over_two() -> None:
    launch = derive_launch_conditions(np.array([0.0, 0.0, 30.0]), np.zeros(3))
    assert launch.launch_angle == pytest.approx(math.pi / 2.0)
    assert launch.azimuth_angle == 0.0


@pytest.mark.unit
def test_zero_spin_defaults_to_backspin_axis() -> None:
    launch = derive_launch_conditions(np.array([50.0, 0.0, 10.0]), np.zeros(3))
    assert launch.spin_rate == 0.0
    assert launch.spin_axis == pytest.approx(DEFAULT_BACKSPIN_AXIS)


@pytest.mark.unit
def test_rejects_malformed_vectors() -> None:
    with pytest.raises(ValueError, match="ball_velocity"):
        derive_launch_conditions(np.array([1.0, 2.0]), np.zeros(3))
    with pytest.raises(ValueError, match="ball_angular_velocity"):
        derive_launch_conditions(np.zeros(3), np.array([np.nan, 0.0, 0.0]))
