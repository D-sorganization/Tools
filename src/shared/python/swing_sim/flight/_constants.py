"""Vendored physical constants for the ball-flight package (with citations).

Vendored from UpstreamDrift ``src/shared/python/core/physics_constants.py``
so :mod:`shared.python.swing_sim.flight` stays self-contained (epic #4103,
flight port #4107). Kept in a flight-private module (rather than a shared
``swing_sim/constants.py``) so parallel domain ports (impact, #4106) cannot
conflict on one shared file; promote to a shared module when a second
subpackage needs the same values.

Every constant records units and source in its docstring.
"""

from __future__ import annotations

import math

GRAVITY_M_S2 = 9.80665
"""Standard gravity [m/s^2] — NIST CODATA 2018."""

AIR_DENSITY_SEA_LEVEL_KG_M3 = 1.225
"""Air density at sea level, 15 C [kg/m^3] — ISA Standard Atmosphere."""

GOLF_BALL_MASS_KG = 0.04593
"""Maximum golf ball mass (1.620 oz) [kg] — USGA Rule 5-1."""

GOLF_BALL_DIAMETER_M = 0.04267
"""Minimum golf ball diameter (1.680 in) [m] — USGA Rule 5-2."""

GOLF_BALL_RADIUS_M = GOLF_BALL_DIAMETER_M / 2.0
"""Minimum golf ball radius [m] — derived from USGA Rule 5-2."""

MAX_GOLF_BALL_LIFT_COEFFICIENT = 0.155
"""Physical cap on the golf-ball lift coefficient [-] — Penner (2003) fit
ceiling used by the UpstreamDrift flight models."""

MIN_SPEED_THRESHOLD_M_S = 0.1
"""Minimum speed for aerodynamic force evaluation [m/s] — numerical guard."""

NUMERICAL_EPSILON = 1e-10
"""Epsilon for vector normalisation [-] — numerical guard."""

MPH_TO_MPS = 0.44704
"""Miles per hour to metres per second [(m/s)/mph] — NIST (exact)."""

RPM_TO_RAD_S = 2.0 * math.pi / 60.0
"""Revolutions per minute to radians per second [(rad/s)/RPM] — exact."""

__all__ = [
    "AIR_DENSITY_SEA_LEVEL_KG_M3",
    "GOLF_BALL_DIAMETER_M",
    "GOLF_BALL_MASS_KG",
    "GOLF_BALL_RADIUS_M",
    "GRAVITY_M_S2",
    "MAX_GOLF_BALL_LIFT_COEFFICIENT",
    "MIN_SPEED_THRESHOLD_M_S",
    "MPH_TO_MPS",
    "NUMERICAL_EPSILON",
    "RPM_TO_RAD_S",
]
