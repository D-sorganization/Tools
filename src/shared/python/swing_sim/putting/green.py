"""Green roll integration, break, and hole capture (#4125 H3, #4800 P2).

Green model
-----------
The surface geometry lives in :mod:`.surface`: a parametric plane
(:class:`~.surface.PlanarGreenSurface` — grade + aspect, the model
this module carried since #4125 H3) or a grid heightfield
(:class:`~.surface.GridGreenSurface`). Frame: x = initial putt line,
y = left of the putt line. Small-slope (grades are a few percent), so
the in-plane gravity component at ``(x, y)`` is ``-g * grad h``; for
the uniform plane that reduces to the historic::

    g_par = g * (grade / 100) * (cos(aspect), sin(aspect))

:class:`GreenConditions` + :func:`simulate_putt` remain the planar
API and now delegate to the surface integrator with the legacy
speed-threshold capture — bit-identical on any planar surface
(regression-gated: the #4125 reference pins are unchanged).

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

Hole capture
------------
The capture physics — the geometric lip bound ``v_capture ~= 0.82
m/s`` and the published effective radius ``R_eff(v)`` that shrinks
with approach speed (Holmes 1991; Penner 2002) — lives in
:mod:`.capture`, with full derivations and citations. The legacy
:func:`simulate_putt` keeps the historic speed-threshold capture so
its trajectories stay bit-identical; the surface API defaults to the
effective-radius model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from shared.python.contracts import ensure, require, require_finite
from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M

from .capture import HOLE_RADIUS_M, capture_speed_mps, effective_hole_radius_m
from .impact import PuttLaunch
from .roll import DEFAULT_SLIDING_MU, GRAVITY_M_S2, stimp_to_rolling_mu
from .surface import GreenSurface, GridGreenSurface, PlanarGreenSurface

__all__ = [
    "HOLE_RADIUS_M",
    "CaptureModel",
    "GreenConditions",
    "PuttResult",
    "capture_speed_mps",
    "effective_hole_radius_m",
    "simulate_putt",
    "simulate_putt_on_surface",
]

#: Hole-capture models: the published effective-radius model (Holmes
#: 1991 / Penner 2002, module docstring) or the historic #4125 H3
#: speed threshold that :func:`simulate_putt` keeps for bit-identical
#: planar trajectories.
CaptureModel = Literal["effective_radius", "speed_threshold"]

#: In-plane gravity field ``(x, y) -> (gx, gy)`` [m/s^2].
_GravityField = Callable[[float, float], tuple[float, float]]

#: Capture predicate ``(distance_to_hole_m, speed_mps) -> captured``,
#: evaluated only while the ball is over the hole mouth.
_CaptureFn = Callable[[float, float], bool]

#: Integration step [s]; pinned by the TS parity tests.
_DT_S = 0.002

#: Rest threshold [m/s].
_STOP_SPEED_MPS = 0.005

#: Integration time cap [s].
_MAX_TIME_S = 60.0


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
    state: tuple[float, ...],
    sliding: bool,
    mu_slide: float,
    mu_roll: float,
    gravity_at: _GravityField,
) -> tuple[float, float, float, float, float]:
    """Right-hand side of the putt ODE (see module docstring).

    Gravity is sampled at the sub-step position, so a heightfield's
    varying gradient enters every RK4 stage; a planar surface returns
    a precomputed constant, keeping the historic arithmetic
    bit-identical.
    """
    x, y, vx, vy, _s = state
    gx, gy = gravity_at(x, y)
    speed = math.hypot(vx, vy)
    if speed <= 0.0:
        return (0.0, 0.0, gx, gy, 0.0)
    mu = mu_slide if sliding else mu_roll
    ax = -mu * GRAVITY_M_S2 * vx / speed + gx
    ay = -mu * GRAVITY_M_S2 * vy / speed + gy
    ds = 2.5 * mu_slide * GRAVITY_M_S2 if sliding else 0.0
    return (vx, vy, ax, ay, ds)


def _rk4_step(
    state: tuple[float, float, float, float, float],
    sliding: bool,
    mu_slide: float,
    mu_roll: float,
    gravity_at: _GravityField,
) -> tuple[float, float, float, float, float]:
    """One classic RK4 step with the mode held constant."""
    k1 = _derivative(state, sliding, mu_slide, mu_roll, gravity_at)
    mid1 = tuple(s + 0.5 * _DT_S * k for s, k in zip(state, k1, strict=True))
    k2 = _derivative(mid1, sliding, mu_slide, mu_roll, gravity_at)
    mid2 = tuple(s + 0.5 * _DT_S * k for s, k in zip(state, k2, strict=True))
    k3 = _derivative(mid2, sliding, mu_slide, mu_roll, gravity_at)
    end = tuple(s + _DT_S * k for s, k in zip(state, k3, strict=True))
    k4 = _derivative(end, sliding, mu_slide, mu_roll, gravity_at)
    return tuple(  # type: ignore[return-value]
        s + (_DT_S / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        for s, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
    )


def _integrate(
    launch: PuttLaunch,
    gravity_at: _GravityField,
    mu_slide: float,
    mu_roll: float,
    hole_distance_m: float,
    captured: _CaptureFn,
) -> PuttResult:
    """Core RK4 loop shared by the planar and surface entry points."""
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
        state = _rk4_step(state, sliding, mu_slide, mu_roll, gravity_at)
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
            if captured(to_hole, speed):
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


def _capture_predicate(capture_model: CaptureModel) -> _CaptureFn:
    """Capture predicate for a model name (see :data:`CaptureModel`)."""
    if capture_model == "speed_threshold":
        v_capture = capture_speed_mps()

        def threshold(_to_hole: float, speed: float) -> bool:
            return speed <= v_capture

        return threshold
    if capture_model == "effective_radius":

        def effective(to_hole: float, speed: float) -> bool:
            return to_hole <= effective_hole_radius_m(speed)

        return effective
    raise ValueError(f"unknown capture model: {capture_model!r}")


def _require_putt_inputs(launch: PuttLaunch, hole_distance_m: float) -> None:
    """Shared launch / hole-distance preconditions."""
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


def simulate_putt_on_surface(
    launch: PuttLaunch,
    surface: GreenSurface,
    *,
    stimp_ft: float,
    hole_distance_m: float,
    mu_slide: float = DEFAULT_SLIDING_MU,
    capture_model: CaptureModel = "effective_radius",
) -> PuttResult:
    """Integrate one putt on a green surface (planar or heightfield).

    The ball starts at the origin aimed along +x with the hole at
    ``(hole_distance_m, 0)``. In-plane gravity comes from the local
    surface gradient at every RK4 stage; rolling resistance comes from
    the stimp reading via :func:`~.roll.stimp_to_rolling_mu`. Vertical
    launch motion is folded into the ground speed (documented
    simplification, as in :func:`simulate_putt`).

    Args:
        launch: Post-impact ball state from :func:`~.impact.strike`.
        surface: Green geometry (:class:`~.surface.PlanarGreenSurface`
            or :class:`~.surface.GridGreenSurface`).
        stimp_ft: Stimpmeter reading [feet].
        hole_distance_m: Distance to the hole center [m].
        mu_slide: Sliding friction for the skid phase.
        capture_model: Hole-capture model (default: the published
            effective-radius model; ``"speed_threshold"`` is the
            historic #4125 H3 behaviour).

    Returns:
        The integrated :class:`PuttResult`.

    Raises:
        ValueError: If inputs are out of range.
        TypeError: If the surface is not a green surface.
    """
    require(
        isinstance(surface, (PlanarGreenSurface, GridGreenSurface)),
        "surface must be a GreenSurface",
    )
    _require_putt_inputs(launch, hole_distance_m)
    require_finite(mu_slide, "mu_slide")
    require(0.0 < mu_slide <= 1.5, "mu_slide must be in (0, 1.5]", mu_slide)
    mu_roll = stimp_to_rolling_mu(stimp_ft)
    return _integrate(
        launch,
        surface.gravity_inplane_mps2,
        mu_slide,
        mu_roll,
        hole_distance_m,
        _capture_predicate(capture_model),
    )


def simulate_putt(
    launch: PuttLaunch,
    green: GreenConditions,
    hole_distance_m: float,
) -> PuttResult:
    """Integrate one putt on the planar green (legacy #4125 H3 API).

    Delegates to the surface integrator with a
    :class:`~.surface.PlanarGreenSurface` and the historic
    speed-threshold capture — trajectories and results are
    bit-identical to the pre-#4800 planar implementation
    (regression-gated by the reference pins).

    Args:
        launch: Post-impact ball state from :func:`~.impact.strike`.
        green: Green conditions.
        hole_distance_m: Distance to the hole center [m].

    Returns:
        The integrated :class:`PuttResult`.

    Raises:
        ValueError: If inputs are out of range.
    """
    surface = PlanarGreenSurface(
        grade_percent=green.grade_percent,
        aspect_deg=green.aspect_deg,
    )
    return simulate_putt_on_surface(
        launch,
        surface,
        stimp_ft=green.stimp_ft,
        hole_distance_m=hole_distance_m,
        mu_slide=green.mu_slide,
        capture_model="speed_threshold",
    )
