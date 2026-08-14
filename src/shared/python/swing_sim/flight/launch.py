"""Launch-condition derivation from post-impact ball kinematics.

Public, tested port of ``_LaunchConditionsDeriver`` from UpstreamDrift
``src/shared/python/physics/swing_ball_flight_pipeline.py`` (epic #4103,
flight port #4107): post-impact ball velocity/spin vectors in the flight
frame (x forward / y left / z up) → speed, launch angle, azimuth, spin
rate [RPM], and unit spin axis, packed as
:class:`~shared.python.swing_sim.flight.types.LaunchConditions`.
"""

from __future__ import annotations

import math

import numpy as np

from ._constants import RPM_TO_RAD_S
from .types import DEFAULT_BACKSPIN_AXIS, LaunchConditions

_EPS = 1e-12


def derive_launch_conditions(
    ball_velocity: np.ndarray,
    ball_angular_velocity: np.ndarray,
) -> LaunchConditions:
    """Return :class:`LaunchConditions` from post-impact ball kinematics.

    Args:
        ball_velocity: Post-impact ball velocity [m/s], shape (3,), flight
            frame (x forward, y left, z up).
        ball_angular_velocity: Post-impact ball angular velocity [rad/s],
            shape (3,), flight frame.

    Returns:
        LaunchConditions ready for any :class:`BallFlightModel` — speed,
        launch angle from the horizontal plane [rad], azimuth (bearing from
        +x, positive toward +y) [rad], spin rate [RPM], and unit spin axis.

    Raises:
        ValueError: If either vector is not a finite 3-vector.
    """
    vel = np.asarray(ball_velocity, dtype=float)
    spin_w = np.asarray(ball_angular_velocity, dtype=float)
    for name, vec in (
        ("ball_velocity", vel),
        ("ball_angular_velocity", spin_w),
    ):
        if vec.shape != (3,):
            raise ValueError(f"{name} must be shape (3,); got {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} must be finite; got {vec!r}")

    speed = float(math.hypot(vel[0], vel[1], vel[2]))

    # Launch angle from the horizontal plane, radians.
    horiz_speed = float(math.hypot(vel[0], vel[1]))
    if horiz_speed < _EPS:
        launch_angle_rad = math.pi / 2.0
    else:
        launch_angle_rad = float(math.atan2(vel[2], horiz_speed))

    # Azimuth angle (compass bearing, 0 = forward = +x), radians.
    azimuth_rad = 0.0
    if horiz_speed > _EPS:
        azimuth_rad = float(math.atan2(vel[1], vel[0]))

    spin_rate_rad_s = float(math.hypot(spin_w[0], spin_w[1], spin_w[2]))
    spin_rate_rpm = spin_rate_rad_s / RPM_TO_RAD_S
    if spin_rate_rad_s > _EPS:
        spin_axis = spin_w / spin_rate_rad_s
    else:
        spin_axis = np.array(DEFAULT_BACKSPIN_AXIS)

    return LaunchConditions(
        ball_speed=speed,
        launch_angle=launch_angle_rad,
        azimuth_angle=azimuth_rad,
        spin_rate=spin_rate_rpm,
        spin_axis=(float(spin_axis[0]), float(spin_axis[1]), float(spin_axis[2])),
    )


__all__ = ["derive_launch_conditions"]
