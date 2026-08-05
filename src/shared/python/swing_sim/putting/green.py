"""Planar sloped green: putt trajectory, break, capture (#4125, H3).

Green model
-----------
A plane of uniform slope described by a grade (percent rise) and an
aspect (the compass direction the green falls toward, measured
counter-clockwise from the initial putt line; 0 = downhill straight
ahead, +90 = downhill to the putt's left). Frame: x = initial putt
line, y = left of the putt line, both in the green plane. Small-angle
(grades are a few percent), so the in-plane gravity component is::

    g_par = g * (grade / 100) * (cos(aspect), sin(aspect))

ODE and integration
-------------------
State ``(x, y, vx, vy, s)`` where ``s = omega r`` is the ball's
contact-surface speed along its direction of travel (documented
simplification: the spin axis stays perpendicular to the velocity —
exact for straight putts, first-order accurate for the small break
angles of real putts). Two modes, following the skid -> roll
derivation in :mod:`.roll`:

* **Skid** (``s < |v|``): sliding friction opposes the slip;
  ``dv/dt = -mu_k g v_hat + g_par``, ``ds/dt = +(5/2) mu_k g``.
* **Pure roll** (``s >= |v|``): rolling resistance;
  ``dv/dt = -mu_r g v_hat + g_par`` with ``s`` pinned to ``|v|``.

Classic RK4 with a fixed step, mode held constant within a step and
transitions applied between steps (the mode functions are smooth
within a phase; at ``dt = 2 ms`` the transition error is sub-mm).
The integrator is fully deterministic — the TypeScript mirror pins
its output value-for-value.

Lip capture (first principles)
------------------------------
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
"Putting: How a golf ball and hole interact", Am. J. Phys. 59 (1991)
derives up to ~1.6 m/s for a perfectly centered pass (a travel budget
of the full diameter). Off-center passes reduce the budget, so the
radius-budget bound is used here as a representative capture proxy.
A ball that crosses the hole faster than the bound rolls on
(simplification: no lip-out deflection is modeled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shared.python.contracts import ensure, require, require_finite
from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M

from .impact import PuttLaunch
from .roll import DEFAULT_SLIDING_MU, GRAVITY_M_S2, stimp_to_rolling_mu

__all__ = [
    "HOLE_RADIUS_M",
    "GreenConditions",
    "PuttResult",
    "capture_speed_mps",
    "simulate_putt",
]

#: USGA hole radius [m] (4.25 in diameter).
HOLE_RADIUS_M = 0.054

#: Integration step [s]; pinned by the TS parity tests.
_DT_S = 0.002

#: Rest threshold [m/s].
_STOP_SPEED_MPS = 0.005

#: Integration time cap [s].
_MAX_TIME_S = 60.0


def capture_speed_mps() -> float:
    """Geometric lip-capture speed bound (module derivation).

    Returns:
        ``R * sqrt(g / (2 r))`` [m/s], ~0.82.
    """
    return HOLE_RADIUS_M * math.sqrt(GRAVITY_M_S2 / (2.0 * GOLF_BALL_RADIUS_M))


@dataclass(frozen=True)
class GreenConditions:
    """Uniform planar green.

    Attributes:
        stimp_ft: Stimpmeter reading [feet] (green speed).
        grade_percent: Uniform slope grade [%]; 0-8 covers greens.
        aspect_deg: Downhill direction, CCW from the putt line [deg];
            0 = downhill ahead, +90 = downhill to the left.
        mu_slide: Sliding friction for the skid phase.
    """

    stimp_ft: float
    grade_percent: float = 0.0
    aspect_deg: float = 0.0
    mu_slide: float = DEFAULT_SLIDING_MU

    def __post_init__(self) -> None:
        # stimp range is validated by stimp_to_rolling_mu.
        stimp_to_rolling_mu(self.stimp_ft)
        require_finite(self.grade_percent, "grade_percent")
        require(
            0.0 <= self.grade_percent <= 10.0,
            "grade must be in [0, 10] percent",
            self.grade_percent,
        )
        require_finite(self.aspect_deg, "aspect_deg")
        require(
            -360.0 <= self.aspect_deg <= 360.0,
            "aspect must be in [-360, 360] deg",
            self.aspect_deg,
        )
        require_finite(self.mu_slide, "mu_slide")
        require(
            0.0 < self.mu_slide <= 1.5,
            "mu_slide must be in (0, 1.5]",
            self.mu_slide,
        )


@dataclass(frozen=True)
class PuttResult:
    """One integrated putt.

    Attributes:
        path_x_m: Sampled x positions [m] (putt-line axis).
        path_y_m: Sampled y positions [m] (left positive).
        speeds_mps: Sampled speeds [m/s].
        times_s: Sample times [s].
        skid_end_index: First sample index in pure roll.
        skid_distance_m: Ground covered while skidding [m].
        total_distance_m: Total ground covered [m].
        time_s: Time to rest or capture [s].
        break_m: Final lateral displacement [m], left positive.
        holed: Whether the ball was captured.
        speed_at_hole_mps: Speed when first crossing the hole mouth
            [m/s]; None when the ball never crossed it.
        margin_mps: When holed, capture-bound minus crossing speed
            [m/s]; None otherwise.
        miss_distance_m: When missed, rest-to-hole distance [m];
            None when holed.
    """

    path_x_m: tuple[float, ...]
    path_y_m: tuple[float, ...]
    speeds_mps: tuple[float, ...]
    times_s: tuple[float, ...]
    skid_end_index: int
    skid_distance_m: float
    total_distance_m: float
    time_s: float
    break_m: float
    holed: bool
    speed_at_hole_mps: float | None
    margin_mps: float | None
    miss_distance_m: float | None

    @property
    def skid_fraction(self) -> float:
        """Skid distance as a fraction of the total distance."""
        if self.total_distance_m <= 0.0:
            return 0.0
        return self.skid_distance_m / self.total_distance_m


def _derivative(
    state: tuple[float, float, float, float, float],
    sliding: bool,
    mu_slide: float,
    mu_roll: float,
    g_par: tuple[float, float],
) -> tuple[float, float, float, float, float]:
    """Right-hand side of the putt ODE (see module docstring)."""
    _x, _y, vx, vy, _s = state
    speed = math.hypot(vx, vy)
    if speed <= 0.0:
        return (0.0, 0.0, g_par[0], g_par[1], 0.0)
    mu = mu_slide if sliding else mu_roll
    ax = -mu * GRAVITY_M_S2 * vx / speed + g_par[0]
    ay = -mu * GRAVITY_M_S2 * vy / speed + g_par[1]
    ds = 2.5 * mu_slide * GRAVITY_M_S2 if sliding else 0.0
    return (vx, vy, ax, ay, ds)


def _rk4_step(
    state: tuple[float, float, float, float, float],
    sliding: bool,
    mu_slide: float,
    mu_roll: float,
    g_par: tuple[float, float],
) -> tuple[float, float, float, float, float]:
    """One classic RK4 step with the mode held constant."""
    k1 = _derivative(state, sliding, mu_slide, mu_roll, g_par)
    mid1 = tuple(s + 0.5 * _DT_S * k for s, k in zip(state, k1, strict=True))
    k2 = _derivative(mid1, sliding, mu_slide, mu_roll, g_par)  # type: ignore[arg-type]
    mid2 = tuple(s + 0.5 * _DT_S * k for s, k in zip(state, k2, strict=True))
    k3 = _derivative(mid2, sliding, mu_slide, mu_roll, g_par)  # type: ignore[arg-type]
    end = tuple(s + _DT_S * k for s, k in zip(state, k3, strict=True))
    k4 = _derivative(end, sliding, mu_slide, mu_roll, g_par)  # type: ignore[arg-type]
    return tuple(  # type: ignore[return-value]
        s + (_DT_S / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        for s, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
    )


def simulate_putt(
    launch: PuttLaunch,
    green: GreenConditions,
    hole_distance_m: float,
) -> PuttResult:
    """Integrate one putt on the planar green.

    The ball starts at the origin aimed along +x with the hole at
    ``(hole_distance_m, 0)``. Vertical launch motion is folded into
    the ground speed (documented simplification: at ~3 deg effective
    loft the airborne hop is a few millimetres and carries < 0.5 % of
    the energy).

    Args:
        launch: Post-impact ball state from :func:`~.impact.strike`.
        green: Green conditions.
        hole_distance_m: Distance to the hole center [m].

    Returns:
        The integrated :class:`PuttResult`.

    Raises:
        ValueError: If inputs are out of range.
    """
    require_finite(hole_distance_m, "hole_distance_m")
    require(
        0.1 <= hole_distance_m <= 40.0,
        "hole distance must be in [0.1, 40] m",
        hole_distance_m,
    )
    require(
        launch.horizontal_speed_mps > 0.0,
        "putt must start moving",
        launch.horizontal_speed_mps,
    )

    mu_roll = stimp_to_rolling_mu(green.stimp_ft)
    aspect = math.radians(green.aspect_deg)
    grade = green.grade_percent / 100.0
    g_par = (
        GRAVITY_M_S2 * grade * math.cos(aspect),
        GRAVITY_M_S2 * grade * math.sin(aspect),
    )
    v_capture = capture_speed_mps()

    state = (
        0.0,
        0.0,
        launch.horizontal_speed_mps,
        0.0,
        launch.spin_rad_s * GOLF_BALL_RADIUS_M,
    )
    sliding = state[4] < state[2]
    xs = [0.0]
    ys = [0.0]
    speeds = [launch.horizontal_speed_mps]
    times = [0.0]
    distance = 0.0
    skid_distance = 0.0
    skid_end_index = 0 if not sliding else -1
    holed = False
    speed_at_hole: float | None = None
    time = 0.0

    while time < _MAX_TIME_S:
        prev = state
        state = _rk4_step(state, sliding, green.mu_slide, mu_roll, g_par)
        time += _DT_S
        step = math.hypot(state[0] - prev[0], state[1] - prev[1])
        distance += step
        speed = math.hypot(state[2], state[3])
        if sliding:
            skid_distance += step
            if state[4] >= speed:
                sliding = False
                skid_end_index = len(xs)
        xs.append(state[0])
        ys.append(state[1])
        speeds.append(speed)
        times.append(time)
        to_hole = math.hypot(state[0] - hole_distance_m, state[1])
        if to_hole <= HOLE_RADIUS_M:
            if speed_at_hole is None:
                speed_at_hole = speed
            if speed <= v_capture:
                holed = True
                break
        if speed <= _STOP_SPEED_MPS:
            break

    if skid_end_index < 0:  # never transitioned (stopped while sliding)
        skid_end_index = len(xs) - 1
    miss_distance = None
    margin = None
    if holed and speed_at_hole is not None:
        margin = v_capture - speed_at_hole
    else:
        miss_distance = math.hypot(xs[-1] - hole_distance_m, ys[-1])

    ensure(distance >= 0.0, "distance must be non-negative", distance)
    ensure(
        skid_distance <= distance + 1e-9,
        "skid cannot exceed the total roll",
        (skid_distance, distance),
    )
    return PuttResult(
        path_x_m=tuple(xs),
        path_y_m=tuple(ys),
        speeds_mps=tuple(speeds),
        times_s=tuple(times),
        skid_end_index=skid_end_index,
        skid_distance_m=skid_distance,
        total_distance_m=distance,
        time_s=time,
        break_m=ys[-1],
        holed=holed,
        speed_at_hole_mps=speed_at_hole,
        margin_mps=margin,
        miss_distance_m=miss_distance,
    )
