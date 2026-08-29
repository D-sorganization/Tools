"""Hole-capture physics: lip bound and effective radius (#4800 P2).

Lip capture (first principles, #4125 H3)
----------------------------------------
The ball is supported by turf until its contact point — directly
below its center — crosses the hole rim, i.e. until its center is
within the hole radius ``R`` (USGA hole: 4.25 in diameter,
``R = 0.054 m``). It is then in free fall, and is captured if it
drops far enough to strike the far wall below its equator before
crossing. Assumptions (documented): center-line pass, drop of half a
ball diameter ``r`` needed, horizontal travel budget of one hole
radius ``R``. Free fall covers ``r`` in ``t = sqrt(2 r / g)``, so::

    v_capture = R * sqrt(g / (2 r)) ~= 0.82 m/s

This is the conservative end of the published range: Holmes,
"Putting: How a golf ball and hole interact", Am. J. Phys. 59,
129-136 (1991) derives up to ~1.6 m/s for a perfectly centered pass
(a travel budget of the full diameter). Off-center passes reduce the
budget, so the radius-budget bound is used here as a representative
capture proxy. A ball that crosses the hole faster than the bound
rolls on (simplification: no lip-out deflection is modeled).

Effective capture radius (P2 upgrade)
-------------------------------------
The published capture picture (Holmes 1991, above; A. R. Penner, "The
physics of putting", Can. J. Phys. 80, 83-96, 2002) is an effective
hole that shrinks with approach speed: an off-center pass at
perpendicular offset ``b`` has a shorter travel budget across the
mouth, so the drop-budget condition generalizes (with the same
half-chord budget posture as the center-line ``R`` above) to::

    captured  iff  b <= R_eff(v) = R * sqrt(1 - (v / v_capture)^2)

``R_eff`` runs from the full hole radius at a dying pace to zero at
``v_capture`` — the #4125 H3 constant is pinned as the exact limiting
case (capture is possible iff ``v < v_capture``), and the window is
strictly monotone in approach speed (test-gated). The legacy
:func:`~.green.simulate_putt` keeps the historic speed-threshold
capture so its trajectories stay bit-identical; the surface API
defaults to the effective-radius model.
"""

from __future__ import annotations

import math

from shared.python.contracts import ensure, require, require_finite
from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M

from .roll import GRAVITY_M_S2

__all__ = [
    "HOLE_RADIUS_M",
    "capture_speed_mps",
    "effective_hole_radius_m",
]

#: USGA hole radius [m] (4.25 in diameter).
HOLE_RADIUS_M = 0.054


def capture_speed_mps() -> float:
    """Geometric lip-capture speed bound (module derivation).

    Returns:
        ``R * sqrt(g / (2 r))`` [m/s], ~0.82.
    """
    return HOLE_RADIUS_M * math.sqrt(GRAVITY_M_S2 / (2.0 * GOLF_BALL_RADIUS_M))


def effective_hole_radius_m(speed_mps: float) -> float:
    """Effective capture radius at an approach speed (module derivation).

    ``R_eff(v) = R sqrt(1 - (v / v_capture)^2)`` — the published
    shrinking-hole capture picture (Holmes 1991; Penner 2002, module
    docstring) with :func:`capture_speed_mps` pinned as the limiting
    case: the full hole radius at a dying pace, zero at and above
    ``v_capture``, strictly decreasing in between.

    Args:
        speed_mps: Approach speed at the hole mouth [m/s].

    Returns:
        Effective capture radius [m].

    Raises:
        ValueError: If the speed is negative or non-finite.
    """
    require_finite(speed_mps, "speed_mps")
    require(speed_mps >= 0.0, "speed must be non-negative", speed_mps)
    ratio = speed_mps / capture_speed_mps()
    if ratio >= 1.0:
        return 0.0
    result = HOLE_RADIUS_M * math.sqrt(1.0 - ratio * ratio)
    ensure(0.0 <= result <= HOLE_RADIUS_M, "R_eff must be within the hole", result)
    return result
