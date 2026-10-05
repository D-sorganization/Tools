"""Immutable SI schema for a generic single-piece iron head (Tools #4149).

Schema identity
    ``golf_club.iron_parameters/1`` (see ``iron_serialization``). Any change to
    a field's name, unit, meaning, or bound is a new schema version.

Head frame (shared with the wedge family)
    ``x`` points toward the target (horizontal component of the face normal),
    ``y`` is up from the ground plane, and ``z`` runs heel to toe for a
    right-handed head (heel at ``-z``). Lengths are metres, angles degrees.

Topology
    The body is a ruled loft between a heel section and a toe section. Each
    section is the polygon leading edge -> face top -> topline back -> back
    wall top -> trailing edge. The face, topline, and sole are planes; only
    the upper back surface twists between heel and toe. A hollow hosel tube is
    fused at the heel. Leading-edge radius, sole camber, cavity back, and
    weight ports are deliberately absent from version 1: they would make the
    body non-polyhedral and remove the closed-form mass-property check that
    the golden test relies on.

Parameter choices and ranges (design rationale, for owner review)
    The ranges are *envelopes* that admit common modern iron archetypes,
    not statistics. Issue #4149 cites no dimensional source, so they are
    engineering judgement against the spread of publicly listed iron
    specifications; no number is copied from, or claimed to match, a specific
    commercial head, and each bound is open to owner revision. The only
    external rule relied on is the USGA/R&A Equipment Rules requirement that
    a non-putter head be longer heel-to-toe than face-to-back, which the
    section builder asserts as a postcondition.

    * ``loft_deg`` [16, 50]: from a 2-iron to a set-matched pitching or
      gap wedge. Higher lofts belong to the wedge family.
    * ``lie_deg`` [56, 66]: standard long-to-short iron lie with fitting
      adjustment headroom on both sides.
    * ``bounce_deg`` [0, 10]: sole angle with the trailing edge on the
      ground and the leading edge raised (standard bounce sense).
    * ``blade_length_m`` [0.065, 0.090]: heel-to-toe body length.
    * ``heel_face_height_m`` [0.028, 0.050] and ``toe_face_height_m``
      [0.038, 0.065]: face height measured along the face plane; the toe is
      never lower than the heel.
    * ``sole_width_m`` [0.012, 0.032]: blade (narrow) to game-improvement
      (wide) soles.
    * ``topline_thickness_m`` [0.003, 0.010]: normal to the face.
    * ``back_wall_height_m`` [0.004, 0.030]: height of the vertical rear
      wall above the trailing edge; sets the muscle / mass position.
    * ``offset_m`` [0, 0.008]: distance the leading edge sits behind the
      front of the hosel.
    * ``hosel_outer_diameter_m`` [0.012, 0.016], ``hosel_bore_diameter_m``
      [0.0088, 0.0100]: covers 0.355 in taper-tip and 0.370 in parallel-tip
      bores with a minimum 1.5 mm wall (same floor as the wedge family).
    * ``hosel_length_m`` [0.040, 0.075]: along the shaft axis.
    * ``material_density_kg_m3`` [2000, 20000]: same envelope as the wedge
      family (aluminium to tungsten alloys); steel is about 7800.
    * ``target_mass_kg`` [0.230, 0.300]: long-iron to short-iron head mass.
      The builder reports the residual; it does not solve to it.

    Cross-field contracts (rejected at construction): toe face height not
    below the heel's; hosel wall of at least 1.5 mm; the back-wall top at
    least 1 mm below the heel topline; and the back-wall top at least 2 mm
    behind the face plane, which keeps every section a simple polygon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ._validation import require_finite_float, require_identifier
from .wedge_parameters import Handedness, WedgeGeometryProvenance

MIN_HOSEL_WALL_M = 0.0015
MIN_TOPLINE_CLEARANCE_M = 0.001
MIN_BACK_WALL_SETBACK_M = 0.002
_BOUNDS: dict[str, tuple[float, float]] = {
    "loft_deg": (16.0, 50.0),
    "lie_deg": (56.0, 66.0),
    "bounce_deg": (0.0, 10.0),
    "blade_length_m": (0.065, 0.090),
    "heel_face_height_m": (0.028, 0.050),
    "toe_face_height_m": (0.038, 0.065),
    "sole_width_m": (0.012, 0.032),
    "topline_thickness_m": (0.003, 0.010),
    "back_wall_height_m": (0.004, 0.030),
    "offset_m": (0.0, 0.008),
    "hosel_outer_diameter_m": (0.012, 0.016),
    "hosel_bore_diameter_m": (0.0088, 0.0100),
    "hosel_length_m": (0.040, 0.075),
    "material_density_kg_m3": (2_000.0, 20_000.0),
    "target_mass_kg": (0.230, 0.300),
}


class IronPreset(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Illustrative, non-vendor positions in a generic iron set."""

    LONG_IRON = "long_iron"
    MID_IRON = "mid_iron"
    SHORT_IRON = "short_iron"


@dataclass(frozen=True)
class IronHeadParameters:
    """Supported domain for the version-1 exact iron solid.

    Preconditions are enforced in ``__post_init__``; an instance that exists
    always describes a buildable, simple-polygon body (see module docstring).
    ``provenance`` uses the family-neutral wedge provenance record.
    """

    head_id: str
    handedness: Handedness
    loft_deg: float
    lie_deg: float
    bounce_deg: float
    blade_length_m: float
    heel_face_height_m: float
    toe_face_height_m: float
    sole_width_m: float
    topline_thickness_m: float
    back_wall_height_m: float
    offset_m: float
    hosel_outer_diameter_m: float
    hosel_bore_diameter_m: float
    hosel_length_m: float
    material_density_kg_m3: float
    target_mass_kg: float
    provenance: WedgeGeometryProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_id", require_identifier(self.head_id, "head_id"))
        if not isinstance(self.handedness, Handedness):
            raise TypeError("handedness must be Handedness")
        if not isinstance(self.provenance, WedgeGeometryProvenance):
            raise TypeError("provenance must be WedgeGeometryProvenance")
        for name, (lower, upper) in _BOUNDS.items():
            value = require_finite_float(getattr(self, name), name)
            if value < lower or value > upper:
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
            object.__setattr__(self, name, value)
        _require_cross_field_geometry(self)


def _require_cross_field_geometry(parameters: IronHeadParameters) -> None:
    if parameters.toe_face_height_m < parameters.heel_face_height_m:
        raise ValueError("toe_face_height_m must be at least heel_face_height_m")
    wall = 0.5 * (parameters.hosel_outer_diameter_m - parameters.hosel_bore_diameter_m)
    if wall < MIN_HOSEL_WALL_M:
        raise ValueError(f"hosel wall must be at least {MIN_HOSEL_WALL_M} m")
    loft = math.radians(parameters.loft_deg)
    bounce = math.radians(parameters.bounce_deg)
    leading_y = parameters.sole_width_m * math.sin(bounce)
    heel_topline_back_y = (
        leading_y
        + parameters.heel_face_height_m * math.cos(loft)
        - parameters.topline_thickness_m * math.sin(loft)
    )
    if parameters.back_wall_height_m > heel_topline_back_y - MIN_TOPLINE_CLEARANCE_M:
        raise ValueError(
            "back_wall_height_m leaves less than "
            f"{MIN_TOPLINE_CLEARANCE_M} m below the heel topline"
        )
    back_x = -parameters.sole_width_m * math.cos(bounce)
    setback = -(
        back_x * math.cos(loft)
        + (parameters.back_wall_height_m - leading_y) * math.sin(loft)
    )
    if setback < MIN_BACK_WALL_SETBACK_M:
        raise ValueError(
            f"back wall must sit at least {MIN_BACK_WALL_SETBACK_M} m "
            "behind the face plane"
        )


_PRESET_VALUES: dict[IronPreset, dict[str, float]] = {
    IronPreset.LONG_IRON: {
        "loft_deg": 21.0,
        "lie_deg": 60.5,
        "heel_face_height_m": 0.036,
        "toe_face_height_m": 0.048,
        "sole_width_m": 0.014,
        "back_wall_height_m": 0.010,
        "offset_m": 0.0045,
        "target_mass_kg": 0.248,
    },
    IronPreset.MID_IRON: {
        "loft_deg": 33.0,
        "lie_deg": 62.5,
        "heel_face_height_m": 0.038,
        "toe_face_height_m": 0.051,
        "sole_width_m": 0.018,
        "back_wall_height_m": 0.012,
        "offset_m": 0.0030,
        "target_mass_kg": 0.264,
    },
    IronPreset.SHORT_IRON: {
        "loft_deg": 41.0,
        "lie_deg": 64.0,
        "heel_face_height_m": 0.040,
        "toe_face_height_m": 0.054,
        "sole_width_m": 0.022,
        "back_wall_height_m": 0.016,
        "offset_m": 0.0020,
        "target_mass_kg": 0.278,
    },
}


def iron_preset(preset: IronPreset) -> IronHeadParameters:
    """Return one generic single-piece iron starting point."""
    if not isinstance(preset, IronPreset):
        raise TypeError("preset must be IronPreset")
    return IronHeadParameters(
        head_id=f"generic-forged-iron-{preset.value.replace('_', '-')}",
        handedness=Handedness.RIGHT,
        bounce_deg=3.0,
        blade_length_m=0.078,
        topline_thickness_m=0.0060,
        hosel_outer_diameter_m=0.0135,
        hosel_bore_diameter_m=0.0094,
        hosel_length_m=0.058,
        material_density_kg_m3=7_800.0,
        provenance=WedgeGeometryProvenance(
            source_name="illustrative generic archetype",
            geometry_basis="general single-piece iron proportions and datums",
            uncertainty_note=(
                "Illustrative engineering geometry; not proprietary and not a "
                "validated copy of a commercial head."
            ),
            data_license="MIT",
        ),
        **_PRESET_VALUES[preset],
    )


__all__ = [
    "IronHeadParameters",
    "IronPreset",
    "iron_preset",
]
