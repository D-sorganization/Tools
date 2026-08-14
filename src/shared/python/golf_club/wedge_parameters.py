"""Immutable SI contracts for a generic modern-wedge head family."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._validation import require_finite_float, require_identifier

_MIN_HOSEL_WALL_M = 0.0015
_BOUNDS: dict[str, tuple[float, float]] = {
    "loft_deg": (40.0, 65.0),
    "lie_deg": (55.0, 70.0),
    "bounce_deg": (0.0, 18.0),
    "face_length_m": (0.070, 0.100),
    "face_height_m": (0.035, 0.065),
    "sole_width_m": (0.010, 0.032),
    "topline_thickness_m": (0.003, 0.010),
    "leading_edge_radius_m": (0.0005, 0.005),
    "rear_curve_depth_fraction": (0.20, 0.85),
    "face_progression_m": (-0.005, 0.015),
    "hosel_outer_diameter_m": (0.012, 0.020),
    "hosel_bore_diameter_m": (0.007, 0.013),
    "hosel_length_m": (0.040, 0.090),
    "material_density_kg_m3": (2_000.0, 20_000.0),
    "target_mass_kg": (0.240, 0.360),
}


class Handedness(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Mirror policy for heel/hosel placement in the head frame."""

    RIGHT = "right"
    LEFT = "left"


class WedgePreset(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Illustrative, non-vendor wedge archetypes."""

    LOW_BOUNCE = "low_bounce"
    MID_BOUNCE = "mid_bounce"
    HIGH_BOUNCE = "high_bounce"


@dataclass(frozen=True)
class WedgeGeometryProvenance:
    """Source basis and claim boundary for one parameter set."""

    source_name: str
    geometry_basis: str
    uncertainty_note: str
    source_uri: str | None = None
    data_license: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_name", "geometry_basis", "uncertainty_note"):
            object.__setattr__(
                self, name, require_identifier(getattr(self, name), name)
            )
        for name in ("source_uri", "data_license"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_identifier(value, name))


@dataclass(frozen=True)
class WedgeHeadParameters:
    """Supported topology domain for the first exact wedge solid.

    Face height is measured along the face plane. Face length is the
    heel-to-toe span. Bounce is the central sole tangent angle above the
    ground plane from leading edge toward trailing edge.
    """

    head_id: str
    handedness: Handedness
    loft_deg: float
    lie_deg: float
    bounce_deg: float
    face_length_m: float
    face_height_m: float
    sole_width_m: float
    topline_thickness_m: float
    leading_edge_radius_m: float
    rear_curve_depth_fraction: float
    face_progression_m: float
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
        wall = 0.5 * (self.hosel_outer_diameter_m - self.hosel_bore_diameter_m)
        if wall < _MIN_HOSEL_WALL_M:
            raise ValueError(f"hosel wall must be at least {_MIN_HOSEL_WALL_M} m")


def wedge_preset(preset: WedgePreset) -> WedgeHeadParameters:
    """Return one generic modern-wedge starting point."""
    if not isinstance(preset, WedgePreset):
        raise TypeError("preset must be WedgePreset")
    bounce_by_preset = {
        WedgePreset.LOW_BOUNCE: 6.0,
        WedgePreset.MID_BOUNCE: 10.0,
        WedgePreset.HIGH_BOUNCE: 14.0,
    }
    sole_width_by_preset = {
        WedgePreset.LOW_BOUNCE: 0.016,
        WedgePreset.MID_BOUNCE: 0.020,
        WedgePreset.HIGH_BOUNCE: 0.024,
    }
    return WedgeHeadParameters(
        head_id=f"generic-modern-wedge-{preset.value.replace('_', '-')}",
        handedness=Handedness.RIGHT,
        loft_deg=56.0,
        lie_deg=64.0,
        bounce_deg=bounce_by_preset[preset],
        face_length_m=0.085,
        face_height_m=0.052,
        sole_width_m=sole_width_by_preset[preset],
        topline_thickness_m=0.0055,
        leading_edge_radius_m=0.0022,
        rear_curve_depth_fraction=0.85,
        face_progression_m=0.003,
        hosel_outer_diameter_m=0.0145,
        hosel_bore_diameter_m=0.0094,
        hosel_length_m=0.062,
        material_density_kg_m3=7_800.0,
        target_mass_kg=0.296,
        provenance=WedgeGeometryProvenance(
            source_name="illustrative generic archetype",
            geometry_basis="general modern-wedge proportions and declared datums",
            uncertainty_note=(
                "Illustrative engineering geometry; not proprietary and not a "
                "validated copy of a commercial head."
            ),
            data_license="MIT",
        ),
    )


__all__ = [
    "Handedness",
    "WedgeHeadParameters",
    "WedgeGeometryProvenance",
    "WedgePreset",
    "wedge_preset",
]
