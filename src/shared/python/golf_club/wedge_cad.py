"""Exact OpenCascade solid generation for the generic wedge family."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._head_cad import hollow_tube, matching_planar_face
from .wedge_geometry import wedge_body_profile_m
from .wedge_parameters import Handedness, WedgeHeadParameters

_MM_PER_M = 1_000.0
_M3_PER_MM3 = 1.0e-9
_HOSEL_BODY_OVERLAP_MM = 2.0


@dataclass(frozen=True)
class WedgeMeasuredMetrics:
    """Dimensions recovered from the generated B-Rep and its mass model."""

    loft_deg: float
    lie_deg: float
    bounce_deg: float
    face_length_m: float
    volume_m3: float
    mass_kg: float
    target_mass_residual_kg: float


@dataclass(frozen=True)
class WedgeSolidResult:
    """Canonical exact solid plus independently measured engineering values."""

    solid: Any
    measured: WedgeMeasuredMetrics


def build_wedge_solid(parameters: WedgeHeadParameters) -> WedgeSolidResult:
    """Build one closed, deterministic wedge head and hollow hosel solid."""
    if not isinstance(parameters, WedgeHeadParameters):
        raise TypeError("parameters must be WedgeHeadParameters")
    body = _build_body(parameters)
    hosel, shaft_axis = _build_hosel(parameters)
    combined = body.fuse(hosel)
    if not combined.is_valid or len(combined.solids()) != 1:
        raise RuntimeError("wedge body and hosel did not form one valid solid")
    measured = _measure_solid(combined, parameters, shaft_axis)
    return WedgeSolidResult(solid=combined, measured=measured)


def _build_body(parameters: WedgeHeadParameters) -> Any:
    from build123d import Solid, fillet

    points = _body_profile_mm(parameters)
    half_width = 0.5 * parameters.face_length_m * _MM_PER_M
    heel_wire = _profile_wire(points, -half_width)
    toe_wire = _profile_wire(points, half_width)
    body = Solid.make_loft([heel_wire, toe_wire], ruled=True)
    leading_x, leading_y = points[0]
    leading_edge = min(
        body.edges(),
        key=lambda edge: _leading_edge_score(edge, leading_x, leading_y),
    )
    return fillet(
        leading_edge,
        parameters.leading_edge_radius_m * _MM_PER_M,
    )


def _profile_wire(
    points: tuple[tuple[float, float], ...],
    z_position: float,
) -> Any:
    from build123d import Edge, Wire

    vertices = [(x_value, y_value, z_position) for x_value, y_value in points]
    edges = (
        Edge.make_line(vertices[0], vertices[1]),
        Edge.make_line(vertices[1], vertices[2]),
        Edge.make_bezier(vertices[2], vertices[3], vertices[4], vertices[5]),
        Edge.make_line(vertices[5], vertices[0]),
    )
    return Wire(edges)


def _body_profile_mm(
    parameters: WedgeHeadParameters,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (x_value * _MM_PER_M, y_value * _MM_PER_M)
        for x_value, y_value in wedge_body_profile_m(parameters)
    )


def _leading_edge_score(edge: Any, leading_x: float, leading_y: float) -> float:
    center = edge.center()
    distance = math.hypot(center.X - leading_x, center.Y - leading_y)
    return float(distance + abs(edge.length - 2.0 * abs(center.Z)) * 1.0e-6)


def _build_hosel(parameters: WedgeHeadParameters) -> tuple[Any, np.ndarray]:
    direction_z = -1.0 if parameters.handedness is Handedness.RIGHT else 1.0
    lie = math.radians(parameters.lie_deg)
    shaft_axis = np.array([0.0, math.sin(lie), direction_z * math.cos(lie)])
    half_width = 0.5 * parameters.face_length_m * _MM_PER_M
    heel_z = direction_z * (
        half_width - 0.45 * parameters.hosel_outer_diameter_m * _MM_PER_M
    )
    along_face = 0.62 * parameters.face_height_m * _MM_PER_M
    base = np.array(
        [
            parameters.face_progression_m * _MM_PER_M
            - along_face * math.sin(math.radians(parameters.loft_deg)),
            parameters.leading_edge_radius_m * _MM_PER_M
            + along_face * math.cos(math.radians(parameters.loft_deg)),
            heel_z,
        ]
    )
    origin = base - _HOSEL_BODY_OVERLAP_MM * shaft_axis
    total_length = parameters.hosel_length_m * _MM_PER_M + _HOSEL_BODY_OVERLAP_MM
    radii = (
        0.5 * parameters.hosel_outer_diameter_m * _MM_PER_M,
        0.5 * parameters.hosel_bore_diameter_m * _MM_PER_M,
    )
    return hollow_tube(origin, shaft_axis, radii, total_length), shaft_axis


def _measure_solid(
    solid: Any,
    parameters: WedgeHeadParameters,
    shaft_axis: np.ndarray,
) -> WedgeMeasuredMetrics:
    loft_normal = np.array(
        [
            math.cos(math.radians(parameters.loft_deg)),
            math.sin(math.radians(parameters.loft_deg)),
            0.0,
        ]
    )
    bounce_normal = np.array(
        [
            -math.sin(math.radians(parameters.bounce_deg)),
            -math.cos(math.radians(parameters.bounce_deg)),
            0.0,
        ]
    )
    face, measured_face_normal = matching_planar_face(solid, loft_normal)
    _, measured_sole_normal = matching_planar_face(solid, bounce_normal)
    _, measured_hosel_axis = matching_planar_face(solid, shaft_axis)
    loft = math.degrees(math.atan2(measured_face_normal[1], measured_face_normal[0]))
    bounce = math.degrees(
        math.atan2(-measured_sole_normal[0], -measured_sole_normal[1])
    )
    lie = math.degrees(math.atan2(measured_hosel_axis[1], abs(measured_hosel_axis[2])))
    volume_m3 = float(solid.volume) * _M3_PER_MM3
    mass = volume_m3 * parameters.material_density_kg_m3
    return WedgeMeasuredMetrics(
        loft_deg=loft,
        lie_deg=lie,
        bounce_deg=bounce,
        face_length_m=float(face.bounding_box().size.Z) / _MM_PER_M,
        volume_m3=volume_m3,
        mass_kg=mass,
        target_mass_residual_kg=mass - parameters.target_mass_kg,
    )


__all__ = [
    "WedgeMeasuredMetrics",
    "WedgeSolidResult",
    "build_wedge_solid",
]
