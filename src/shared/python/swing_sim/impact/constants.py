"""Vendored physical constants for the impact package (SI units).

Vendored from UpstreamDrift ``src/shared/python/core/physics_constants.py``
so that swing_sim stays self-contained (Tools must NOT import
upstream-drift; the dependency arrow points the other way — UD vendors
Tools). Each constant carries its original citation.

Epic #4103 / issue #4106.
"""

from __future__ import annotations

# --- Golf ball (USGA equipment rules) ------------------------------------
GOLF_BALL_MASS_KG = 0.04593
"""Maximum golf ball mass (1.620 oz). Source: USGA Rule 5-1."""

GOLF_BALL_DIAMETER_M = 0.04267
"""Minimum golf ball diameter (1.680 in). Source: USGA Rule 5-2."""

GOLF_BALL_RADIUS_M = GOLF_BALL_DIAMETER_M / 2.0
"""Minimum golf ball radius (derived from USGA Rule 5-2)."""

GOLF_BALL_MOMENT_OF_INERTIA_KG_M2 = (
    (2.0 / 5.0) * GOLF_BALL_MASS_KG * GOLF_BALL_RADIUS_M**2
)
"""Golf ball MOI, uniform solid-sphere approximation (2/5 m r^2)."""

# --- Driver clubhead ------------------------------------------------------
DRIVER_COR = 0.83
"""Driver coefficient of restitution. Source: USGA/R&A CT limit
(~0.83 equivalent COR)."""

DRIVER_MOI_KG_M2 = 4.5e-4
"""Driver clubhead MOI about CG (perpendicular to face).
Source: typical driver specs."""

DRIVER_MASS_KG = 0.200
"""Typical driver clubhead mass. Source: industry standard specs."""

DRIVER_LOFT_RAD = 0.18325957145940461  # math.radians(10.5)
"""Typical driver loft (10.5 deg). Source: modern trade average."""

DRIVER_LIE_RAD = 1.0471975511965976  # math.radians(60.0)
"""Typical driver lie angle (60 deg). Source: industry standard specs."""

DRIVER_CG_DEPTH_M = 0.035
"""Typical driver CG depth behind the face plane [m]. Modern drivers place
the CG 30-40 mm rearward of the face; the front-to-back lever arm is what
produces gear-effect spin on off-center hits. Source: published driver CG
measurements (e.g. Golf Laboratories / manufacturer spec sheets, ~1.2-1.6
in). New in Tools; not present in the UpstreamDrift constant set."""

# --- Contact --------------------------------------------------------------
TYPICAL_CONTACT_DURATION_S = 0.0005
"""Typical ball-clubface contact time. Source: high-speed video studies."""

__all__ = [
    "DRIVER_CG_DEPTH_M",
    "DRIVER_COR",
    "DRIVER_LIE_RAD",
    "DRIVER_LOFT_RAD",
    "DRIVER_MASS_KG",
    "DRIVER_MOI_KG_M2",
    "GOLF_BALL_DIAMETER_M",
    "GOLF_BALL_MASS_KG",
    "GOLF_BALL_MOMENT_OF_INERTIA_KG_M2",
    "GOLF_BALL_RADIUS_M",
    "TYPICAL_CONTACT_DURATION_S",
]
