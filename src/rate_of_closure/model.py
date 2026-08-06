"""Twist-based impact-point velocity model for a rotating clubhead.

A clubhead at impact is a rigid body with a full 6-DOF motion state (a
twist): the velocity of any point P on the body is

    v(P) = v(R) + omega x (P - R)

where R is the tracked reference point (center of mass or geometric
center). Launch monitors report the path of R; the ball only experiences
the path of the contact point. This module quantifies the difference.

Angular velocity is decomposed the way 3-D motion-capture golf studies
report it (Cheetham 2014; AffineDrift closure-rate literature dossier):

* ``omega_plane`` — rotation about the swing-plane normal (Cheetham's
  swing-plane velocity, SPV): the arc rotation carrying the head.
* ``omega_shaft`` — rotation about the shaft axis (Cheetham's horizontal
  turning velocity, HTV): the closing/release component. Positive
  closes the face for a right-handed golfer.

The vertical component of the resulting omega vector is exactly the
dossier's global closure-rate reconciliation,

    CCV = HTV * sin(lie) + SPV * cos(lie)

so the model's closure rate and the literature's club closure velocity
(CCV, ~2,100 deg/s tour mean) are the same quantity by construction.

Frame convention — the AffineDrift house convention (Launch Monitor
Technology Review, ``sections/02-parameters.tex``, following the
standard launch-monitor definitions):
    x along the target line, y vertical (up), z to the right of the
    target line looking down it. Angles positive right and up; club
    path + = in-to-out (right of target for a right-handed golfer).

The reference point — the launch monitor's geometric center (GC; the CG
lies within ~6 mm of it) — travels dead at the target with zero attack
angle by construction, so every output is a *deviation from the
reported delivery* at the instant of maximum compression. That is
exactly the reference-point question: openly published launch-monitor material puts the
GC-path vs face-center-path gap at roughly 3 degrees for a driver.
Equivalently the gap is d / R_ISA (offset over distance to the
instantaneous screw axis), which is why the speed-invariant closure
unit is degrees per foot of travel (omega / v = 1 / R_ISA). Defaults
are dossier-sourced values, and every number is an input, not an
assumption.

Design by Contract: preconditions validate physical ranges at
construction; postconditions guarantee finite outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from ._contracts import ensure, require, require_finite

__all__ = [
    "MPH_PER_MPS",
    "ClosureMetrics",
    "ImpactResult",
    "ImpactScenario",
    "closure_metrics",
    "solve",
    "sweep",
]

#: Exact conversion factor between meters/second and miles/hour.
MPH_PER_MPS = 1.0 / 0.44704

#: Inclusive physical bounds per scenario field: (low, high).
_BOUNDS: dict[str, tuple[float, float]] = {
    "clubhead_speed_mph": (1.0, 250.0),
    "omega_plane_dps": (-20_000.0, 20_000.0),
    "omega_shaft_dps": (-20_000.0, 20_000.0),
    "lie_angle_deg": (10.0, 90.0),
    "com_to_face_mm": (0.0, 150.0),
    "impact_offset_toe_mm": (-80.0, 80.0),
    "impact_offset_high_mm": (-40.0, 40.0),
    "contact_duration_us": (0.0, 2_000.0),
}


@dataclass(frozen=True)
class ImpactScenario:
    """One delivery: reference-point speed plus the clubhead's rotation.

    Args:
        clubhead_speed_mph: Speed of the tracked reference point (COM or
            geometric center), miles per hour.
        omega_plane_dps: Angular velocity about the swing-plane normal
            (Cheetham's SPV), degrees per second. Positive swings the
            head toward the target. Default 1,870 makes the default
            CCV equal the dossier's ~2,100 deg/s tour mean at lie 58.
        omega_shaft_dps: Angular velocity about the shaft axis
            (Cheetham's HTV), degrees per second. Positive closes the
            face (right-handed golfer). Tour driver mean 1,307 +/- 304,
            range 652-2,432 (n = 94).
        lie_angle_deg: Shaft angle from horizontal at impact, degrees.
            90 puts the shaft vertical (isolates pure horizontal closure).
        com_to_face_mm: Distance from the reference point (GC) forward
            to the face center, millimetres. published head data cites 25-50 for
            drivers; 40 is the AffineDrift worked-example value.
        impact_offset_toe_mm: Impact-point offset from face center toward
            the toe (positive) or heel (negative), millimetres.
        impact_offset_high_mm: Impact-point offset above (positive) or
            below (negative) face center, millimetres.
        contact_duration_us: Ball contact time, microseconds (~450 for a
            driver). Used for face rotation *during* contact.
    """

    clubhead_speed_mph: float
    omega_plane_dps: float = 1870.0
    omega_shaft_dps: float = 1307.0
    lie_angle_deg: float = 58.0
    com_to_face_mm: float = 40.0
    impact_offset_toe_mm: float = 0.0
    impact_offset_high_mm: float = 0.0
    contact_duration_us: float = 450.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"{field.name} must be a number, got {type(value).__name__}"
                )
            require_finite(float(value), name=field.name)
            low, high = _BOUNDS[field.name]
            require(
                low <= float(value) <= high,
                f"{field.name} must be within [{low}, {high}]",
                value,
            )


@dataclass(frozen=True)
class ImpactResult:
    """Deviation of the impact point's delivery from the reference point's.

    Attributes:
        reference_speed_mph: Speed of the tracked reference point.
        point_speed_mph: Speed of the impact point.
        speed_delta_mph: ``point - reference`` speed difference.
        tangential_speed_mph: Magnitude of ``omega x r`` at the impact
            point — the rotation-induced velocity.
        path_deviation_deg: Horizontal angle between the impact point's
            path and the reference path. Positive = right (in-to-out);
            negative = left, per the AffineDrift launch-monitor convention.
        aoa_deviation_deg: Vertical angle difference. Positive = the
            impact point is travelling more upward (shallower).
        closure_during_contact_deg: Face closure accumulated over the
            contact duration.
        loft_gain_during_contact_deg: Dynamic loft gained over contact.
        closure_rate_dps: Instantaneous horizontal face-closure rate —
            the literature's CCV (= HTV sin lie + SPV cos lie).
        normalized_closure_deg_per_ft: Closure per foot of travel,
            omega / v — the speed-invariant unit preferred in the
            AffineDrift closure-rate derivation (equals 1 / R_ISA).
        point_velocity_mps: Impact-point velocity vector (x, y, z), m/s.
        omega_dps: Angular velocity vector (x, y, z), degrees per second.
        shaft_axis: Unit vector from clubhead toward the hands.
        plane_normal: Unit normal of the swing plane.
    """

    reference_speed_mph: float
    point_speed_mph: float
    speed_delta_mph: float
    tangential_speed_mph: float
    path_deviation_deg: float
    aoa_deviation_deg: float
    closure_during_contact_deg: float
    loft_gain_during_contact_deg: float
    closure_rate_dps: float
    normalized_closure_deg_per_ft: float
    point_velocity_mps: tuple[float, float, float]
    omega_dps: tuple[float, float, float]
    shaft_axis: tuple[float, float, float]
    plane_normal: tuple[float, float, float]


def impact_frame(lie_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Shaft axis and swing-plane normal for a given impact lie angle.

    The shaft points from the head up toward the hands: up (+y) and
    toward the golfer (-z, left of the target line for a right-handed
    golfer). The swing plane contains the shaft and the target line;
    its normal is ``unit(v_hat x shaft)``.
    """
    lie = math.radians(lie_angle_deg)
    shaft = np.array([0.0, math.sin(lie), -math.cos(lie)])
    v_hat = np.array([1.0, 0.0, 0.0])
    normal = np.cross(v_hat, shaft)
    normal /= np.linalg.norm(normal)
    return shaft, normal


def _omega_rad(scenario: ImpactScenario) -> np.ndarray:
    """Angular velocity vector in rad/s from the two reported components."""
    shaft, normal = impact_frame(scenario.lie_angle_deg)
    return np.asarray(
        math.radians(scenario.omega_plane_dps) * normal
        + math.radians(scenario.omega_shaft_dps) * shaft
    )


def impact_lever_m(scenario: ImpactScenario) -> np.ndarray:
    """Vector from the reference point to the impact point, metres."""
    return (
        np.array(
            [
                scenario.com_to_face_mm,
                scenario.impact_offset_high_mm,
                scenario.impact_offset_toe_mm,
            ]
        )
        / 1000.0
    )


def solve(scenario: ImpactScenario) -> ImpactResult:
    """Solve one scenario for the impact point's delivery deviation.

    Args:
        scenario: The delivery to evaluate.

    Returns:
        The full set of deviations and geometry vectors.

    Raises:
        TypeError: If the scenario is not an :class:`ImpactScenario`.
    """
    if not isinstance(scenario, ImpactScenario):
        raise TypeError(
            f"scenario must be ImpactScenario, got {type(scenario).__name__}"
        )

    v_ref = np.array([scenario.clubhead_speed_mph / MPH_PER_MPS, 0.0, 0.0])
    omega = _omega_rad(scenario)
    tangential = np.cross(omega, impact_lever_m(scenario))
    v_point = v_ref + tangential

    ref_speed = float(np.linalg.norm(v_ref))
    point_speed = float(np.linalg.norm(v_point))
    # Club path + = in-to-out (+z, right of target); AoA + = up (+y).
    path_dev = math.degrees(math.atan2(v_point[2], v_point[0]))
    aoa_dev = math.degrees(math.atan2(v_point[1], math.hypot(v_point[0], v_point[2])))

    # The face normal is +x at square impact. Its horizontal (closing)
    # swing rate is the vertical omega component — the literature's CCV —
    # and its loft-gain rate is the heel-toe (+z) component.
    contact_s = scenario.contact_duration_us * 1e-6
    closure_rate = math.degrees(omega[1])
    closure = closure_rate * contact_s
    loft_gain = math.degrees(omega[2]) * contact_s
    speed_fts = ref_speed / 0.3048
    normalized_closure = closure_rate / speed_fts if speed_fts else 0.0

    shaft, normal = impact_frame(scenario.lie_angle_deg)
    result = ImpactResult(
        reference_speed_mph=ref_speed * MPH_PER_MPS,
        point_speed_mph=point_speed * MPH_PER_MPS,
        speed_delta_mph=(point_speed - ref_speed) * MPH_PER_MPS,
        tangential_speed_mph=float(np.linalg.norm(tangential)) * MPH_PER_MPS,
        path_deviation_deg=path_dev,
        aoa_deviation_deg=aoa_dev,
        closure_during_contact_deg=closure,
        loft_gain_during_contact_deg=loft_gain,
        closure_rate_dps=closure_rate,
        normalized_closure_deg_per_ft=normalized_closure,
        point_velocity_mps=(
            float(v_point[0]),
            float(v_point[1]),
            float(v_point[2]),
        ),
        omega_dps=(
            float(math.degrees(omega[0])),
            float(math.degrees(omega[1])),
            float(math.degrees(omega[2])),
        ),
        shaft_axis=(float(shaft[0]), float(shaft[1]), float(shaft[2])),
        plane_normal=(float(normal[0]), float(normal[1]), float(normal[2])),
    )
    ensure(
        all(math.isfinite(getattr(result, f.name)) for f in fields(result)[:10]),
        "all scalar outputs must be finite",
    )
    return result


#: Heel-to-toe face length of a modern driver, metres — used only for
#: the toe-vs-heel speed differential metric (about 4.6 in is typical
#: of published head geometries).
_FACE_LENGTH_M = 0.117


@dataclass(frozen=True)
class ClosureMetrics:
    """Common closure parameters reported across the golf literature.

    Every value is an algebraic restatement of the solved delivery —
    no additional empirical inputs — so each stays traceable to the
    same twist model. Rates appear per second, per millisecond, per
    foot, and per inch because different sources prefer different
    normalizations; R_ISA and the time-to-square figures restate the
    same rotation as a geometry and as a timing.

    Attributes:
        ccv_dps: Club closure velocity (the model's closure rate).
        closure_deg_per_ft: Closure per foot of travel (omega / v).
        closure_deg_per_inch: Closure per inch of travel.
        closure_deg_per_ms: Closure per millisecond.
        r_isa_m: Distance to the instantaneous screw axis, metres
            (v / omega). ``inf`` when the face is not closing.
        r_isa_ft: The same distance in feet.
        time_to_square_from_1deg_open_ms: How long before impact the
            face was one degree open, milliseconds. ``inf`` when the
            face is not closing.
        toe_heel_speed_delta_mph: Speed difference between the toe and
            heel ends of the face due to rotation (117 mm face length).
    """

    ccv_dps: float
    closure_deg_per_ft: float
    closure_deg_per_inch: float
    closure_deg_per_ms: float
    r_isa_m: float
    r_isa_ft: float
    time_to_square_from_1deg_open_ms: float
    toe_heel_speed_delta_mph: float


def closure_metrics(scenario: ImpactScenario) -> ClosureMetrics:
    """Restate one delivery as the closure parameters the literature uses.

    Args:
        scenario: The delivery to evaluate.

    Returns:
        The derived metric set. Ratio metrics are ``inf`` when the
        closure rate is zero (a non-closing face never squares).
    """
    result = solve(scenario)
    ccv = result.closure_rate_dps
    speed_mps = result.reference_speed_mph / MPH_PER_MPS
    omega = np.radians(np.array(result.omega_dps))
    # Toe and heel sit +/- half the face length along +/-z; their speed
    # difference is |omega x (L z_hat)|.
    toe_heel = float(
        np.linalg.norm(np.cross(omega, np.array([0.0, 0.0, _FACE_LENGTH_M])))
    )
    closing = abs(ccv) > 1e-12
    r_isa_m = speed_mps / abs(math.radians(ccv)) if closing else math.inf
    return ClosureMetrics(
        ccv_dps=ccv,
        closure_deg_per_ft=result.normalized_closure_deg_per_ft,
        closure_deg_per_inch=result.normalized_closure_deg_per_ft / 12.0,
        closure_deg_per_ms=ccv / 1000.0,
        r_isa_m=r_isa_m,
        r_isa_ft=r_isa_m / 0.3048,
        time_to_square_from_1deg_open_ms=(1000.0 / abs(ccv) if closing else math.inf),
        toe_heel_speed_delta_mph=toe_heel * MPH_PER_MPS,
    )


def sweep(scenario: ImpactScenario, field_name: str, values: np.ndarray) -> np.ndarray:
    """Path deviation across a range of one scenario field, vectorized.

    Args:
        scenario: Base scenario; every field except ``field_name`` is held.
        field_name: Scenario field to vary.
        values: Values to evaluate, one output per element.

    Returns:
        Array of ``path_deviation_deg``, same shape as ``values``.

    Raises:
        ValueError: If ``field_name`` is not a scenario field.
    """
    names = {f.name for f in fields(ImpactScenario)}
    require(field_name in names, f"unknown scenario field {field_name!r}")
    array = np.asarray(values, dtype=float)
    flat = [
        solve(
            ImpactScenario(
                **{
                    **{
                        f.name: getattr(scenario, f.name)
                        for f in fields(ImpactScenario)
                    },
                    field_name: float(value),
                }
            )
        ).path_deviation_deg
        for value in array.ravel()
    ]
    return np.asarray(flat).reshape(array.shape)
