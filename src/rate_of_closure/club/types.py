"""Club specification types — SI units throughout.

A :class:`ClubSpec` is the single source of truth for one club: static
geometry (length, loft, lie), head inertial properties (mass, scalar MOI
about the shaft axis, CG location), and optional face curvature (bulge
and roll radii). Everything is SI — meters, kilograms, degrees for
angles (the one deliberate non-SI concession, matching how every golf
spec sheet publishes loft and lie).

Design by Contract: construction validates every field against the
physical bounds in :data:`SPEC_BOUNDS`, so a ``ClubSpec`` that exists is
a ``ClubSpec`` that is physically plausible.

The TypeScript twin is ``web/src/model/club.ts``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, fields

from rate_of_closure._contracts import require, require_finite

__all__ = ["SPEC_BOUNDS", "ClubSpec", "ClubType", "HeadStyle"]


class ClubType(enum.Enum):
    """Broad club categories, matching manufacturer spec-sheet groupings."""

    DRIVER = "Driver"
    WOOD = "Wood"
    HYBRID = "Hybrid"
    IRON = "Iron"
    WEDGE = "Wedge"
    PUTTER = "Putter"


class HeadStyle(enum.Enum):
    """Head-shape refinement within a club type (putters, H1 #4125).

    ``AUTO`` means "the canonical shape for the club type"; putters may
    pick ``MALLET`` (deep rounded body) or ``BLADE`` (anser-style
    shallow rectangle with a plumber's-neck hosel — a generic form, no
    brand geometry).
    """

    AUTO = "Auto"
    MALLET = "Mallet"
    BLADE = "Blade"


#: Inclusive physical bounds per numeric spec field: (low, high).
#: Sourced from the span of typical published manufacturer specs with
#: generous margins (e.g. USGA limits club length to 48 in = 1.22 m).
SPEC_BOUNDS: dict[str, tuple[float, float]] = {
    "length_m": (0.6, 1.3),
    "head_mass_kg": (0.1, 0.5),
    "loft_deg": (0.0, 70.0),
    "lie_deg": (45.0, 80.0),
    "moi_about_shaft_kg_m2": (5.0e-5, 2.0e-3),
    "cg_depth_m": (0.0, 0.08),
    "cg_height_m": (0.0, 0.06),
    "face_bulge_radius_m": (0.1, 2.0),
    "face_roll_radius_m": (0.1, 2.0),
}


@dataclass(frozen=True)
class ClubSpec:
    """One club's static specification, SI units.

    Args:
        name: Display name (e.g. ``"Driver 10.5°"``). Must be non-empty.
        club_type: Broad category (:class:`ClubType`).
        length_m: Overall club length, meters (grip butt to sole).
        head_mass_kg: Clubhead mass, kilograms.
        loft_deg: Static face loft, degrees from vertical.
        lie_deg: Static lie angle, degrees from horizontal.
        moi_about_shaft_kg_m2: Scalar head moment of inertia about the
            shaft axis, kg·m² (spec sheets publish this in g·cm²;
            1 g·cm² = 1e-7 kg·m²).
        cg_depth_m: Head CG distance measured back from the face,
            meters.
        cg_height_m: Head CG height above the sole plane, meters.
        face_bulge_radius_m: Horizontal (heel-toe) face curvature
            radius, meters, or ``None`` for a flat face. Typical driver
            bulge is around 0.25-0.33 m (10-13 in) per published
            fitting references.
        face_roll_radius_m: Vertical (crown-sole) face curvature
            radius, meters, or ``None`` for a flat face. Typically
            similar to bulge on drivers.
        head_style: Head-shape refinement (:class:`HeadStyle`);
            ``AUTO`` selects the canonical shape for the club type
            (putters resolve to the blade form).
    """

    name: str
    club_type: ClubType
    length_m: float
    head_mass_kg: float
    loft_deg: float
    lie_deg: float
    moi_about_shaft_kg_m2: float
    cg_depth_m: float
    cg_height_m: float
    face_bulge_radius_m: float | None = None
    face_roll_radius_m: float | None = None
    head_style: HeadStyle = HeadStyle.AUTO

    def __post_init__(self) -> None:
        require(
            isinstance(self.head_style, HeadStyle),
            "head_style must be a HeadStyle",
            self.head_style,
        )
        require(
            isinstance(self.name, str) and len(self.name) > 0,
            "name must be a non-empty string",
            self.name,
        )
        require(
            isinstance(self.club_type, ClubType),
            "club_type must be a ClubType",
            self.club_type,
        )
        for field in fields(self):
            if field.name in ("name", "club_type", "head_style"):
                continue
            value = getattr(self, field.name)
            optional = field.name in ("face_bulge_radius_m", "face_roll_radius_m")
            if optional and value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"{field.name} must be a number, got {type(value).__name__}"
                )
            require_finite(float(value), name=field.name)
            low, high = SPEC_BOUNDS[field.name]
            require(
                low <= float(value) <= high,
                f"{field.name} must be within [{low}, {high}]",
                value,
            )

    @property
    def has_curved_face(self) -> bool:
        """Whether either face-curvature radius is specified."""
        return self.face_bulge_radius_m is not None or (
            self.face_roll_radius_m is not None
        )
