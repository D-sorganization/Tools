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
* **Irons** — blade profile: tall thin face, a flat ~21 mm sole at
  reference (typical published iron sole widths span ~18-24 mm), rear
  sections dropping toward the sole so the topline stays thin; the
  cavity-back recess pushes the back cap inward (#4803).
* **Wedges** — muscle-back: a ~29 mm sole at reference (typical
  published wedge sole widths span ~26-32 mm) with a sub-millimeter
  bounce dip behind the leading edge, rear-sole mass bias, and no
  cavity (#4803).
* **Mallet putter** — deep rounded body: half-widths taper along a
  semicircular plan ~100 mm back.
* **Blade putter** — anser-style form: shallow rectangular head with a
  lower flange back and a plumber's-neck hosel offset behind the face.

Loft is realized as a **leading-edge lean** (#4799): :func:`lean_point`
shears head-frame points about the ``y = y_le`` leading-edge line, the
same affine map the mesh generator applies, so :func:`face_center_point`
and :func:`hosel_point` stay coincident with the generated geometry.

The TypeScript twin is ``web/src/model/clubHeads.ts``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rate_of_closure._contracts import ensure, require, require_finite

from .types import ClubSpec, ClubType, HeadStyle

__all__ = [
    "BLADE_HOSEL_HEIGHT_FRACTION",
    "PLUMBER_NECK_OFFSET_M",
    "HeadProfile",
    "face_center_point",
    "hosel_point",
    "lean_point",
    "leading_edge_height",
    "mass_scale",
    "profile_for",
    "resolved_style",
]

#: Plumber's-neck shaft offset [m] on the blade putter — roughly one
#: shaft diameter ("full-shaft offset" in typical published putter
#: fitting references).
PLUMBER_NECK_OFFSET_M = 0.0095

#: Where the blade (iron/wedge) hosel meets the head, as a fraction of
#: the face slant height above the leading edge (#4799 G2): the hosel
#: enters at the heel, where the face is shorter than at center, so the
#: anchor sits a bit above mid-face rather than at the topline.
BLADE_HOSEL_HEIGHT_FRACTION = 0.58

#: Cross-section row: (x, half_height, half_width, y_center), meters.
Section = tuple[float, float, float, float]


@dataclass(frozen=True)
class HeadProfile:
    """A club-type head shape at its reference mass.

    Attributes:
        reference_mass_kg: Head mass the section proportions represent.
        sections: Loft cross-sections, face (+x) first, tail last.
        hosel_anchor: Hosel location on/near the envelope (heel side,
            z < 0) at reference mass. Blades (irons/wedges) use only its
            z component — their x/y come from the loft-aware leading-edge
            rule (#4799 G2).
        rear_recess_m: Inward (+x) offset of the tail-cap fan center —
            a cavity-back recess when positive.
        hosel_offset_m: Blade hosel offset behind the leading edge at
            reference mass (#4799 G2) — a touch of real offset, never
            onset. Unused by non-blade profiles.
    """

    reference_mass_kg: float
    sections: tuple[Section, ...]
    hosel_anchor: tuple[float, float, float]
    rear_recess_m: float = 0.0
    hosel_offset_m: float = 0.0

    def __post_init__(self) -> None:
        require(self.reference_mass_kg > 0.0, "reference mass positive")
        require(len(self.sections) >= 3, "a profile needs >= 3 sections")
        xs = [s[0] for s in self.sections]
        require(xs == sorted(xs, reverse=True), "sections must run face to tail")
        require(all(s[1] > 0.0 and s[2] > 0.0 for s in self.sections), "extents > 0")
        require(self.hosel_anchor[2] < 0.0, "hosel must sit on the heel side (z < 0)")
        require(self.rear_recess_m >= 0.0, "recess must be non-negative")
        require(self.hosel_offset_m >= 0.0, "hosel offset must be non-negative")


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

#: Irons — blade: thin topline, cavity-back recess, and a real sole
#: (#4803): every station's bottom sits on the ``y = y_le`` sole line,
#: so the sole runs flat ~21 mm front-to-back at reference — inside the
#: typical published iron sole-width span of ~18-24 mm (players through
#: game-improvement irons; no brand geometry is reproduced) (250 g).
IRON_PROFILE = HeadProfile(
    reference_mass_kg=0.250,
    sections=(
        (0.011, 0.025, 0.040, 0.0),  # face plate (strike-view extents)
        (0.005, 0.023, 0.039, -0.002),  # bottom -0.025 = y_le
        (-0.004, 0.018, 0.037, -0.007),  # bottom -0.025 = y_le
        (-0.010, 0.010, 0.032, -0.015),  # sole tail; bottom -0.025 = y_le
    ),
    hosel_anchor=(0.008, 0.024, -0.038),  # heel side; z only (#4799 G2)
    rear_recess_m=0.006,
    hosel_offset_m=0.005,  # mid-iron offset, typical published range
)

#: Wedges — muscle-back blade with a deep sole (#4803): the sole runs
#: ~29 mm front-to-back at reference, inside the typical published
#: wedge sole-width span of ~26-32 mm (sand/lob soles are the widest in
#: a set; no brand geometry is reproduced). The station bottoms dip
#: 0.6-0.8 mm below the ``y = y_le`` leading edge mid-sole and relieve
#: to 0.3 mm at the trailing edge — a bounce hint (the leading edge
#: rides above the sole's low point) that also biases the sole-slab
#: mass toward the rear, like a muscle/bounce sole. No cavity (300 g).
WEDGE_PROFILE = HeadProfile(
    reference_mass_kg=0.300,
    sections=(
        (0.012, 0.026, 0.040, 0.0),  # face plate; bottom -0.026 = y_le
        (0.004, 0.0242, 0.039, -0.0024),  # bottom -0.0266 (0.6 mm dip)
        (-0.008, 0.0184, 0.037, -0.0084),  # bottom -0.0268 (bounce apex)
        (-0.0165, 0.0100, 0.032, -0.0163),  # sole tail; bottom -0.0263
    ),
    hosel_anchor=(0.009, 0.025, -0.038),  # heel side; z only (#4799 G2)
    hosel_offset_m=0.0035,  # wedges carry less offset than irons
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


def leading_edge_height(spec: ClubSpec) -> float:
    """Leading-edge height ``y_le`` [m]: the authored face-section bottom,
    mass-scaled (#4799 G1). The loft lean's fixed line."""
    profile = profile_for(spec)
    scale = mass_scale(spec)
    _x, hh, _hw, yc = profile.sections[0]
    return (yc - hh) * scale


def lean_point(
    spec: ClubSpec, point: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Leading-edge loft lean of one head-frame point [m] (#4799 G1).

    The affine map applied to every generated-head vertex::

        x' = x - (y - y_le) * sin(loft)
        y' = y_le + (y - y_le) * cos(loft)
        z' = z

    The ``y = y_le`` fiber (leading edge / sole line) is fixed, so
    lofting a face never throws the leading edge forward of the authored
    station; the vertical extent compresses by ``cos(loft)`` — the
    authored face height becomes slant height, as on a real wedge.
    """
    for value, name in zip(point, ("x", "y", "z"), strict=True):
        require_finite(value, name=name)
    lam = math.radians(spec.loft_deg)
    y_le = leading_edge_height(spec)
    dy = point[1] - y_le
    return (
        point[0] - dy * math.sin(lam),
        y_le + dy * math.cos(lam),
        point[2],
    )


def face_center_point(spec: ClubSpec) -> tuple[float, float, float]:
    """Face-plate center in the head frame, meters (leaned; #4799 G1)."""
    profile = profile_for(spec)
    scale = mass_scale(spec)
    x, _hh, _hw, yc = profile.sections[0]
    return lean_point(spec, (x * scale, yc * scale, 0.0))


def hosel_point(spec: ClubSpec) -> tuple[float, float, float]:
    """Hosel (shaft attachment) point on the head, meters, head frame.

    Loft-aware per type (#4799 G2), so the shaft lands even with the
    leading edge instead of far behind it:

    * **Irons / wedges** — ``x = x_le - offset`` (authored offset,
      never onset), ``y = y_le + f * H * cos(loft)`` with
      ``f = BLADE_HOSEL_HEIGHT_FRACTION`` and ``H`` the authored face
      height, z from the authored anchor.
    * **Woods / hybrids / putters** — the authored anchor under the same
      leading-edge lean the mesh gets (heel-crown transition for woods
      and hybrids, plumber's-neck set-back on the blade putter).

    Deterministic per spec; both renderers attach the shaft line here.
    """
    profile = profile_for(spec)
    scale = mass_scale(spec)
    if spec.club_type in (ClubType.IRON, ClubType.WEDGE):
        x_le, hh, _hw, _yc = profile.sections[0]
        height = 2.0 * hh * scale
        lam = math.radians(spec.loft_deg)
        point = (
            (x_le - profile.hosel_offset_m) * scale,
            leading_edge_height(spec)
            + BLADE_HOSEL_HEIGHT_FRACTION * height * math.cos(lam),
            profile.hosel_anchor[2] * scale,
        )
    else:
        anchor = profile.hosel_anchor
        point = lean_point(
            spec, (anchor[0] * scale, anchor[1] * scale, anchor[2] * scale)
        )
    ensure(point[2] < 0.0, "hosel point must be on the heel side")
    return (point[0], point[1], point[2])
