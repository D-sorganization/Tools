"""Exact OpenCascade solid generation for the generic iron family (Tools #4149).

Every reported quantity is *recovered from the finished B-Rep*: loft, bounce,
and lie from the normals of its planar datum faces, blade length from the
face extent, and volume, mass, and centre of gravity from OpenCascade's
exact-solid integration (``recover_solid_mass_properties``). None is copied
from the input parameters. The golden tests check the recovered values
against independent references (closed form and exported-mesh divergence
theorem); see ``tests/shared/python/golf_club/test_iron_cad.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._head_cad import MM_PER_M, hollow_tube, matching_planar_face
from ._validation import Vector3, require_finite_float
from .iron_geometry import iron_body_sections_m
from .iron_parameters import IronHeadParameters
from .wedge_parameters import Handedness

_M3_PER_MM3 = 1.0e-9
# The hosel base sits this far above the highest sole point, so its tilted
# bottom cap stays inside the body instead of piercing the sole.
_HOSEL_SOLE_CLEARANCE_M = 0.001


@dataclass(frozen=True)
class SolidMassProperties:
    """Uniform-density volume, mass, and centre of gravity of one solid."""

    volume_m3: float
    mass_kg: float
    cg_m: Vector3


@dataclass(frozen=True)
class IronMeasuredMetrics:
    """Datums and mass properties recovered from the generated head solid."""

    loft_deg: float
    lie_deg: float
    bounce_deg: float
    blade_length_m: float
    volume_m3: float
    mass_kg: float
    cg_m: Vector3
    target_mass_residual_kg: float


@dataclass(frozen=True)
class IronSolidResult:
    """Canonical head solid, its body before the hosel, and measured values."""

    solid: Any
    body: Any
    measured: IronMeasuredMetrics


def build_iron_solid(parameters: IronHeadParameters) -> IronSolidResult:
    """Build one closed, deterministic iron head with a hollow hosel.

    Precondition: ``parameters`` is an ``IronHeadParameters`` (whose
    constructor already enforced every range and cross-field contract).
    Postcondition: ``solid`` is one valid OpenCascade solid and every field
    of ``measured`` was recovered from it.
    """
    if not isinstance(parameters, IronHeadParameters):
        raise TypeError("parameters must be IronHeadParameters")
    body = _build_body(parameters)
    hosel, shaft_axis = _build_hosel(parameters)
    combined = body.fuse(hosel)
    if not combined.is_valid or len(combined.solids()) != 1:
        raise RuntimeError("iron body and hosel did not form one valid solid")
    measured = _measure_solid(combined, parameters, shaft_axis)
    return IronSolidResult(solid=combined, body=body, measured=measured)


def recover_solid_mass_properties(
    solid: object, density_kg_m3: float
) -> SolidMassProperties:
    """Integrate volume and centroid of an exact solid at uniform density.

    Uses OpenCascade's B-Rep surface integration (``GProp``), not any builder
    input. Raises ``ValueError`` for a non-positive density and ``TypeError``
    for an object that is not a build123d solid.
    """
    density = require_finite_float(density_kg_m3, "density_kg_m3", positive=True)
    from build123d import CenterOf

    try:
        volume_mm3 = float(solid.volume)  # type: ignore[attr-defined]
        center = solid.center(CenterOf.MASS)  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError("solid must be a build123d solid") from error
    volume_m3 = volume_mm3 * _M3_PER_MM3
    if not math.isfinite(volume_m3) or volume_m3 <= 0.0:
        raise RuntimeError("solid volume must be finite and positive")
    cg_m = tuple(float(value) / MM_PER_M for value in center)
    return SolidMassProperties(
        volume_m3=volume_m3,
        mass_kg=volume_m3 * density,
        cg_m=(cg_m[0], cg_m[1], cg_m[2]),
    )


def _build_body(parameters: IronHeadParameters) -> Any:
    from build123d import Solid, Wire

    sections = iron_body_sections_m(parameters)
    wires = [
        Wire.make_polygon(
            [
                (x_value * MM_PER_M, y_value * MM_PER_M, z_value * MM_PER_M)
                for x_value, y_value in profile
            ],
            close=True,
        )
        for profile, z_value in (
            (sections.heel_profile_m, sections.heel_z_m),
            (sections.toe_profile_m, sections.toe_z_m),
        )
    ]
    return Solid.make_loft(wires, ruled=True)


def _heel_sign(parameters: IronHeadParameters) -> float:
    return -1.0 if parameters.handedness is Handedness.RIGHT else 1.0


def _build_hosel(parameters: IronHeadParameters) -> tuple[Any, np.ndarray]:
    heel_sign = _heel_sign(parameters)
    lie = math.radians(parameters.lie_deg)
    shaft_axis = np.array([0.0, math.sin(lie), heel_sign * math.cos(lie)])
    outer_radius = 0.5 * parameters.hosel_outer_diameter_m
    highest_sole_y = parameters.sole_width_m * math.sin(
        math.radians(parameters.bounce_deg)
    )
    base_m = np.array(
        [
            parameters.offset_m - outer_radius,
            highest_sole_y + outer_radius * math.cos(lie) + _HOSEL_SOLE_CLEARANCE_M,
            heel_sign * (0.5 * parameters.blade_length_m - outer_radius),
        ]
    )
    radii_mm = (
        outer_radius * MM_PER_M,
        0.5 * parameters.hosel_bore_diameter_m * MM_PER_M,
    )
    tube = hollow_tube(
        base_m * MM_PER_M,
        shaft_axis,
        radii_mm,
        parameters.hosel_length_m * MM_PER_M,
    )
    return tube, shaft_axis


def _measure_solid(
    solid: Any,
    parameters: IronHeadParameters,
    shaft_axis: np.ndarray,
) -> IronMeasuredMetrics:
    loft = math.radians(parameters.loft_deg)
    bounce = math.radians(parameters.bounce_deg)
    face_normal = np.array([math.cos(loft), math.sin(loft), 0.0])
    sole_normal = np.array([math.sin(bounce), -math.cos(bounce), 0.0])
    face, measured_face = matching_planar_face(solid, face_normal)
    _, measured_sole = matching_planar_face(solid, sole_normal)
    _, measured_axis = matching_planar_face(solid, shaft_axis)
    mass = recover_solid_mass_properties(solid, parameters.material_density_kg_m3)
    face_z = [float(vertex.Z) for vertex in face.vertices()]
    return IronMeasuredMetrics(
        loft_deg=math.degrees(math.atan2(measured_face[1], measured_face[0])),
        lie_deg=math.degrees(math.atan2(measured_axis[1], abs(measured_axis[2]))),
        bounce_deg=math.degrees(math.atan2(measured_sole[0], -measured_sole[1])),
        blade_length_m=(max(face_z) - min(face_z)) / MM_PER_M,
        volume_m3=mass.volume_m3,
        mass_kg=mass.mass_kg,
        cg_m=mass.cg_m,
        target_mass_residual_kg=mass.mass_kg - parameters.target_mass_kg,
    )


__all__ = [
    "IronMeasuredMetrics",
    "IronSolidResult",
    "SolidMassProperties",
    "build_iron_solid",
    "recover_solid_mass_properties",
]
