"""Arbitrary-plane contact and landing-target geometry for ground studies."""

import math

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_float
from shared.python.swing_sim.solver.spatial_targets import (
    SpatialTarget,
    SurfaceCircleTolerance,
    SurfaceCorridorTolerance,
    TargetMiss,
)

from .contract_records import GroundSimulationRequest
from .contract_types import GroundSurfaceProfile, Vector3
from .study_types import (
    GroundEndpointKind,
    GroundTargetEvaluation,
)

_GEOMETRY_TOLERANCE_M = 1e-8
_BASIS_TOLERANCE = 1e-12


def _dot(left: Vector3, right: Vector3) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (
        float(left[0] - right[0]),
        float(left[1] - right[1]),
        float(left[2] - right[2]),
    )


def _add(left: Vector3, right: Vector3) -> Vector3:
    return (
        float(left[0] + right[0]),
        float(left[1] + right[1]),
        float(left[2] + right[2]),
    )


def _scale(value: Vector3, factor: float) -> Vector3:
    return (
        float(value[0] * factor),
        float(value[1] * factor),
        float(value[2] * factor),
    )


def _norm(value: Vector3) -> float:
    return math.hypot(*value)


def _canonical_vector(value: Vector3) -> Vector3:
    return (
        canonical_numeric_float(value[0]),
        canonical_numeric_float(value[1]),
        canonical_numeric_float(value[2]),
    )


def _unit(value: Vector3, name: str) -> Vector3:
    magnitude = _norm(value)
    if magnitude <= _BASIS_TOLERANCE:
        raise ValueError(f"{name} is degenerate")
    return _scale(value, 1.0 / magnitude)


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def sphere_contact_point(
    center_position_m: Vector3,
    normal_unit: Vector3,
    ball_radius_m: float,
) -> Vector3:
    """Return the sphere point one radius opposite an upward plane normal."""
    return _subtract(center_position_m, _scale(normal_unit, ball_radius_m))


def _plane_origin(surface: GroundSurfaceProfile) -> Vector3:
    return (0.0, surface.height_m, 0.0)


def surface_contact_point(
    center_position_m: Vector3,
    surface: GroundSurfaceProfile,
    ball_radius_m: float,
) -> Vector3:
    """Return a proven sphere/plane contact point with residual correction."""
    raw = sphere_contact_point(center_position_m, surface.normal_unit, ball_radius_m)
    gap = _dot(
        _subtract(center_position_m, _plane_origin(surface)), surface.normal_unit
    )
    gap -= ball_radius_m
    if abs(gap) > _GEOMETRY_TOLERANCE_M:
        raise ValueError("ground endpoint ball center does not contact the bound plane")
    return _subtract(raw, _scale(surface.normal_unit, gap))


def endpoint_contacts_surface(
    center_position_m: Vector3,
    surface: GroundSurfaceProfile,
    ball_radius_m: float,
) -> bool:
    """Return whether a ball centre represents contact with the bound plane."""
    return (
        abs(endpoint_signed_gap_m(center_position_m, surface, ball_radius_m))
        <= _GEOMETRY_TOLERANCE_M
    )


def endpoint_signed_gap_m(
    center_position_m: Vector3,
    surface: GroundSurfaceProfile,
    ball_radius_m: float,
) -> float:
    """Return sphere clearance above a plane; negative values are penetration."""
    center_height = _dot(
        _subtract(center_position_m, _plane_origin(surface)), surface.normal_unit
    )
    return float(center_height - ball_radius_m)


def endpoint_is_airborne(
    center_position_m: Vector3,
    surface: GroundSurfaceProfile,
    ball_radius_m: float,
) -> bool:
    """Return whether a sphere endpoint is positively clear of the plane."""
    return (
        endpoint_signed_gap_m(center_position_m, surface, ball_radius_m)
        > _GEOMETRY_TOLERANCE_M
    )


def validate_nonpenetrating_endpoint(
    center_position_m: Vector3,
    surface: GroundSurfaceProfile,
    ball_radius_m: float,
) -> None:
    """Reject an endpoint that penetrates the bound plane beyond tolerance."""
    if (
        endpoint_signed_gap_m(center_position_m, surface, ball_radius_m)
        < -_GEOMETRY_TOLERANCE_M
    ):
        raise ValueError("ground endpoint ball center penetrates the bound plane")


def _surface_basis(normal_unit: Vector3) -> tuple[Vector3, Vector3]:
    downrange_axis: Vector3 = (1.0, 0.0, 0.0)
    projected = _subtract(
        downrange_axis, _scale(normal_unit, _dot(downrange_axis, normal_unit))
    )
    if _norm(projected) > _BASIS_TOLERANCE:
        downrange = _unit(projected, "surface downrange tangent")
        right = _unit(_cross(downrange, normal_unit), "surface right tangent")
        return downrange, right
    right_axis: Vector3 = (0.0, 0.0, 1.0)
    projected_right = _subtract(
        right_axis,
        _scale(normal_unit, _dot(right_axis, normal_unit)),
    )
    right = _unit(projected_right, "surface right tangent")
    if _dot(right, right_axis) < 0.0:
        right = _scale(right, -1.0)
    downrange = _unit(_cross(normal_unit, right), "surface downrange tangent")
    return downrange, right


def validate_target(target: SpatialTarget, surface: GroundSurfaceProfile) -> None:
    """Require a landing tolerance bound to the exact target surface."""
    if target.kind != "landing_area":
        raise ValueError("ground study target must be a landing_area")
    if target.ground_source != surface.surface_id:
        raise ValueError("target ground_source must match the result surface_id")
    if not isinstance(
        target.tolerance,
        (SurfaceCircleTolerance, SurfaceCorridorTolerance),
    ):
        raise ValueError("ground study target requires a surface tolerance")
    plane_offset = _dot(
        _subtract(target.point.app_coordinates_m, _plane_origin(surface)),
        surface.normal_unit,
    )
    if abs(plane_offset) > _GEOMETRY_TOLERANCE_M:
        raise ValueError("target center must lie on the bound surface plane")


def _closest_target_point(
    target: SpatialTarget,
    contact_point_m: Vector3,
    surface: GroundSurfaceProfile,
) -> Vector3:
    center = target.point.app_coordinates_m
    downrange, right = _surface_basis(surface.normal_unit)
    delta = _subtract(contact_point_m, center)
    along = _dot(delta, downrange)
    across = _dot(delta, right)
    tolerance = target.tolerance
    if isinstance(tolerance, SurfaceCircleTolerance):
        radial = math.hypot(along, across)
        if radial <= tolerance.radius_m:
            return contact_point_m
        factor = tolerance.radius_m / radial
        return _add(
            center,
            _add(_scale(downrange, along * factor), _scale(right, across * factor)),
        )
    if isinstance(tolerance, SurfaceCorridorTolerance):
        along = min(max(along, -tolerance.half_length_m), tolerance.half_length_m)
        across = min(max(across, -tolerance.half_width_m), tolerance.half_width_m)
        return _add(center, _add(_scale(downrange, along), _scale(right, across)))
    raise TypeError("unsupported target tolerance")


def intrinsic_target_miss(
    target: SpatialTarget,
    contact_point_m: Vector3,
    surface: GroundSurfaceProfile,
) -> TargetMiss:
    closest = _closest_target_point(target, contact_point_m, surface)
    vector = _subtract(contact_point_m, closest)
    distance = _norm(vector)
    accepted = distance <= _BASIS_TOLERANCE
    if accepted:
        closest = contact_point_m
        vector = (0.0, 0.0, 0.0)
        distance = 0.0
    return TargetMiss(
        _canonical_vector(closest),
        _canonical_vector(vector),
        canonical_numeric_float(distance),
        accepted,
    )


def target_evaluation(
    target: SpatialTarget,
    center_position_m: Vector3,
    endpoint_kind: GroundEndpointKind,
    request: GroundSimulationRequest,
) -> GroundTargetEvaluation:
    """Evaluate one observed ball endpoint against an intrinsic surface target."""
    contact = surface_contact_point(
        center_position_m,
        request.surface,
        request.ball_radius_m,
    )
    target_center = target.point.app_coordinates_m
    residual = _subtract(contact, target_center)
    return GroundTargetEvaluation(
        target.label,
        endpoint_kind,
        center_position_m,
        contact,
        target_center,
        residual,
        _norm(residual),
        intrinsic_target_miss(target, contact, request.surface),
    )


__all__ = [
    "sphere_contact_point",
    "intrinsic_target_miss",
    "endpoint_contacts_surface",
    "surface_contact_point",
    "target_evaluation",
    "validate_target",
]
