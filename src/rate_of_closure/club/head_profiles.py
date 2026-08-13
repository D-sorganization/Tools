"""Type-specific head profiles: cross-sections, hosel, face center (H1, #4125).

One :class:`HeadProfile` per broad club shape, expressed as superellipse
loft cross-sections ``(x, half_height, half_width, y_center)`` in the
AffineDrift head frame (x face-forward, y up, z toe) at the profile's
reference head mass. :func:`profile_for` dispatches on the spec's
:class:`~rate_of_closure.club.types.ClubType` (and, for putters, its
:class:`~rate_of_closure.club.types.HeadStyle`), and everything scales
by the constant-density factor ``(head_mass / reference_mass)^(1/3)``.

Shape rationale (proportions follow the span of typical published
head dimensions; no brand geometry is reproduced):

* **Woods/drivers** — rounded crown and curved sole: the deep
  superellipse loft the parametric head has always used (~110 mm deep
  at 200 g).
* **Hybrids** — intermediate: a wood silhouette at ~70% depth.
* **Irons** — blade profile: tall thin face, shallow face-to-back depth
  (~22 mm vs ~110 mm for a wood), rear sections dropping toward the
  sole so the topline stays thin; an optional cavity-back recess pushes
  the back cap inward.
* **Wedges** — iron-like with the rear sections biased further toward
  the sole (a bounce/muscle hint) and no cavity.
* **Mallet putter** — deep rounded body: half-widths taper along a
  semicircular plan ~100 mm back.
* **Blade putter** — anser-style form: shallow rectangular head with a
  lower flange back and a plumber's-neck hosel offset behind the face.

The TypeScript twin is ``web/src/model/clubHeads.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass

from rate_of_closure._contracts import ensure, require

from .types import ClubSpec, ClubType, HeadStyle

__all__ = [
    "PLUMBER_NECK_OFFSET_M",
    "HeadProfile",
    "face_center_point",
    "hosel_point",
    "mass_scale",
    "profile_for",
    "resolved_style",
]

#: Plumber's-neck shaft offset [m] on the blade putter — roughly one
#: shaft diameter ("full-shaft offset" in typical published putter
#: fitting references).
PLUMBER_NECK_OFFSET_M = 0.0095

#: Cross-section row: (x, half_height, half_width, y_center), meters.
Section = tuple[float, float, float, float]


@dataclass(frozen=True)
class HeadProfile:
    """A club-type head shape at its reference mass.

    Attributes:
        reference_mass_kg: Head mass the section proportions represent.
        sections: Loft cross-sections, face (+x) first, tail last.
        hosel_anchor: Hosel location on/near the envelope (heel side,
            z < 0) at reference mass.
        rear_recess_m: Inward (+x) offset of the tail-cap fan center —
            a cavity-back recess when positive.
    """

    reference_mass_kg: float
    sections: tuple[Section, ...]
    hosel_anchor: tuple[float, float, float]
    rear_recess_m: float = 0.0

    def __post_init__(self) -> None:
        require(self.reference_mass_kg > 0.0, "reference mass positive")
        require(len(self.sections) >= 3, "a profile needs >= 3 sections")
        xs = [s[0] for s in self.sections]
        require(xs == sorted(xs, reverse=True), "sections must run face to tail")
        require(all(s[1] > 0.0 and s[2] > 0.0 for s in self.sections), "extents > 0")
        require(self.hosel_anchor[2] < 0.0, "hosel must sit on the heel side (z < 0)")
        require(self.rear_recess_m >= 0.0, "recess must be non-negative")


#: Woods & drivers — the historical parametric-head envelope (200 g).
WOOD_PROFILE = HeadProfile(
    reference_mass_kg=0.200,
    sections=(
        (0.055, 0.028, 0.058, 0.0),  # face plate
        (0.010, 0.031, 0.062, 0.0),  # crown bulge
        (-0.035, 0.024, 0.048, 0.0),  # rear taper
        (-0.055, 0.010, 0.020, 0.0),  # tail
    ),
    hosel_anchor=(0.030, 0.030, -0.052),  # heel-crown transition
)

#: Hybrids — intermediate: wood silhouette at ~70% depth (230 g).
HYBRID_PROFILE = HeadProfile(
    reference_mass_kg=0.230,
    sections=(
        (0.038, 0.024, 0.050, 0.0),
        (0.008, 0.026, 0.052, 0.0),
        (-0.022, 0.020, 0.040, 0.0),
        (-0.037, 0.008, 0.016, 0.0),
    ),
    hosel_anchor=(0.022, 0.025, -0.044),  # heel-crown transition
)

#: Irons — blade: thin topline, ~22 mm deep, cavity-back recess (250 g).
IRON_PROFILE = HeadProfile(
    reference_mass_kg=0.250,
    sections=(
        (0.011, 0.025, 0.040, 0.0),
        (0.005, 0.024, 0.039, 0.0),
        (-0.005, 0.020, 0.037, -0.002),
        (-0.011, 0.011, 0.032, -0.007),
    ),
    hosel_anchor=(0.008, 0.024, -0.038),  # heel-top of the face
    rear_recess_m=0.006,
)

#: Wedges — iron-like, rear mass biased toward the sole (300 g).
WEDGE_PROFILE = HeadProfile(
    reference_mass_kg=0.300,
    sections=(
        (0.012, 0.026, 0.040, 0.0),
        (0.006, 0.025, 0.039, -0.001),
        (-0.005, 0.021, 0.037, -0.004),
        (-0.012, 0.012, 0.032, -0.009),
    ),
    hosel_anchor=(0.009, 0.025, -0.038),  # heel-top of the face
)

#: Mallet putter — deep semicircular-plan rounded body (360 g).
MALLET_PROFILE = HeadProfile(
    reference_mass_kg=0.360,
    sections=(
        (0.020, 0.0140, 0.055, 0.0),
        (-0.005, 0.0145, 0.054, 0.0),
        (-0.035, 0.0140, 0.047, 0.0),
        (-0.060, 0.0120, 0.032, 0.0),
        (-0.080, 0.0070, 0.013, 0.0),
    ),
    hosel_anchor=(0.014, 0.014, -0.048),  # heel-top near the face
)

#: Blade putter — anser-style: shallow rectangle, lower flange back,
#: plumber's-neck hosel set behind the face (350 g).
BLADE_PUTTER_PROFILE = HeadProfile(
    reference_mass_kg=0.350,
    sections=(
        (0.012, 0.0125, 0.050, 0.0),
        (0.004, 0.0125, 0.050, 0.0),
        (-0.004, 0.0090, 0.048, -0.0035),
        (-0.014, 0.0055, 0.043, -0.0070),
    ),
    hosel_anchor=(0.012 - PLUMBER_NECK_OFFSET_M, 0.0125, -0.046),
)


def resolved_style(spec: ClubSpec) -> HeadStyle:
    """The concrete head style for a spec (putter AUTO resolves to BLADE)."""
    if spec.head_style is not HeadStyle.AUTO:
        return spec.head_style
    return HeadStyle.BLADE if spec.club_type is ClubType.PUTTER else HeadStyle.AUTO


def profile_for(spec: ClubSpec) -> HeadProfile:
    """The head profile a spec's type (and putter style) selects."""
    if spec.club_type in (ClubType.DRIVER, ClubType.WOOD):
        return WOOD_PROFILE
    if spec.club_type is ClubType.HYBRID:
        return HYBRID_PROFILE
    if spec.club_type is ClubType.IRON:
        return IRON_PROFILE
    if spec.club_type is ClubType.WEDGE:
        return WEDGE_PROFILE
    require(spec.club_type is ClubType.PUTTER, "unknown club type", spec.club_type)
    if resolved_style(spec) is HeadStyle.MALLET:
        return MALLET_PROFILE
    return BLADE_PUTTER_PROFILE


def mass_scale(spec: ClubSpec) -> float:
    """Uniform envelope scale: constant-density mass scaling per type."""
    profile = profile_for(spec)
    scale = float((spec.head_mass_kg / profile.reference_mass_kg) ** (1.0 / 3.0))
    ensure(scale > 0.0, "mass scale positive")
    return scale


def face_center_point(spec: ClubSpec) -> tuple[float, float, float]:
    """Face-plate center in the head frame, meters (pre-loft tilt)."""
    profile = profile_for(spec)
    scale = mass_scale(spec)
    x, _hh, _hw, yc = profile.sections[0]
    return (x * scale, yc * scale, 0.0)


def hosel_point(spec: ClubSpec) -> tuple[float, float, float]:
    """Hosel (shaft attachment) point on the head, meters, head frame.

    Heel-top for irons, wedges, and putters (with the plumber's-neck
    set-back on the blade putter), heel-crown transition for woods and
    hybrids. Deterministic per spec; both renderers attach the shaft
    line here.
    """
    profile = profile_for(spec)
    scale = mass_scale(spec)
    point = tuple(c * scale for c in profile.hosel_anchor)
    ensure(point[2] < 0.0, "hosel point must be on the heel side")
    return (point[0], point[1], point[2])
