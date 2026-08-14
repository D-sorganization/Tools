"""Reference static response for station-based golf-shaft profiles."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import require_finite_float
from .shaft_profile import ShaftProfile

_MODEL_NAME = "euler_bernoulli_cantilever_static/1"
_GAUSS_NODES = (
    -0.906179845938664,
    -0.538469310105683,
    0.0,
    0.538469310105683,
    0.906179845938664,
)
_GAUSS_WEIGHTS = (
    0.236926885056189,
    0.478628670499366,
    0.568888888888889,
    0.478628670499366,
    0.236926885056189,
)


@dataclass(frozen=True)
class ShaftTipLoad:
    """Tip wrench in the shaft frame.

    The shaft axis points from fixed butt toward the tip. Positive transverse
    forces produce positive deflection on their named axes. Right-handed
    moments give positive rotations.
    """

    force_x_n: float = 0.0
    force_y_n: float = 0.0
    torque_about_shaft_nm: float = 0.0

    def __post_init__(self) -> None:
        for name in ("force_x_n", "force_y_n", "torque_about_shaft_nm"):
            object.__setattr__(
                self,
                name,
                require_finite_float(getattr(self, name), name),
            )


@dataclass(frozen=True)
class ShaftTipResponse:
    """Small-deflection static response at the exposed shaft tip."""

    deflection_x_m: float
    deflection_y_m: float
    rotation_about_x_rad: float
    rotation_about_y_rad: float
    twist_about_shaft_rad: float
    flexible_length_m: float
    model_name: str = _MODEL_NAME


def solve_cantilever_tip_response(
    profile: ShaftProfile,
    load: ShaftTipLoad,
) -> ShaftTipResponse:
    """Solve Euler-Bernoulli bending and Saint-Venant torsion statics.

    The fixed boundary is the trimmed butt. The exposed tip is the trimmed
    tip less insertion depth. Shear deformation, rotary inertia, head-joint
    compliance, and geometric nonlinearity are intentionally outside this
    reference model.
    """
    if not isinstance(profile, ShaftProfile):
        raise TypeError("profile must be ShaftProfile")
    if not isinstance(load, ShaftTipLoad):
        raise TypeError("load must be ShaftTipLoad")
    deflection_x, rotation_y = _bending_response(
        profile,
        load.force_x_n,
        "ei_about_y_n_m2",
    )
    deflection_y, rotation_x_magnitude = _bending_response(
        profile,
        load.force_y_n,
        "ei_about_x_n_m2",
    )
    twist = load.torque_about_shaft_nm * _compliance_integral(
        profile,
        "gj_n_m2",
        power=0,
    )
    return ShaftTipResponse(
        deflection_x_m=deflection_x,
        deflection_y_m=deflection_y,
        rotation_about_x_rad=-rotation_x_magnitude,
        rotation_about_y_rad=rotation_y,
        twist_about_shaft_rad=twist,
        flexible_length_m=profile.flexible_length_m,
    )


def _bending_response(
    profile: ShaftProfile,
    force_n: float,
    stiffness_name: str,
) -> tuple[float, float]:
    rotation = force_n * _compliance_integral(
        profile,
        stiffness_name,
        power=1,
    )
    deflection = force_n * _compliance_integral(
        profile,
        stiffness_name,
        power=2,
    )
    return deflection, rotation


def _compliance_integral(
    profile: ShaftProfile,
    stiffness_name: str,
    *,
    power: int,
) -> float:
    start = profile.butt_trim_m
    end = profile.raw_length_m - profile.tip_trim_m - profile.insertion_depth_m
    boundaries = [start, end]
    boundaries.extend(
        station.position_m
        for station in profile.stations
        if start < station.position_m < end
    )
    ordered = sorted(boundaries)
    total = 0.0
    for left, right in zip(ordered, ordered[1:], strict=False):
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        for node, weight in zip(_GAUSS_NODES, _GAUSS_WEIGHTS, strict=True):
            raw_position = midpoint + half_width * node
            exposed_position = raw_position - start
            lever = profile.flexible_length_m - exposed_position
            stiffness = float(getattr(profile.station_at(raw_position), stiffness_name))
            total += half_width * weight * lever**power / stiffness
    return total


__all__ = [
    "ShaftTipLoad",
    "ShaftTipResponse",
    "solve_cantilever_tip_response",
]
