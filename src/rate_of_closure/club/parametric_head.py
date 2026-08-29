"""Parametric clubhead meshes from a :class:`ClubSpec` — bulge & roll.

Extends the superellipse-loft generator behind the bundled example head
(:mod:`rate_of_closure.scripts.generate_example_head`) into a spec-driven
one. Given a :class:`~rate_of_closure.club.types.ClubSpec`:

* The **envelope** comes from the club type's cross-section profile
  (:mod:`~rate_of_closure.club.head_profiles` — woods, hybrids,
  iron/wedge blades, mallet and blade putters), scaled uniformly by
  ``(head_mass / reference_mass)^(1/3)`` — constant-density mass
  scaling around the reference head each profile represents.
* The **face** is a patch of concentric superellipse rings. When bulge
  (horizontal curvature) and/or roll (vertical curvature) radii are
  specified, each face vertex is set back by the circular sagitta
  ``s(t) = R - sqrt(R² - t²)`` in its offset coordinate, producing a
  convex spherical-ish face; with both off the face is flat.
* **Loft** is a leading-edge lean (#4799 G1): the whole head is built
  unlofted, then every vertex is sheared about the ``y = y_le``
  leading-edge line (:func:`~.head_profiles.lean_point`), so the
  leading edge stays at the authored face station — no onset — the
  authored face height becomes slant height (vertical extent compresses
  by ``cos loft``), and the flat-face normal is exactly
  ``(cos loft, sin loft, 0)``. The map's Jacobian determinant is
  ``cos loft > 0``, preserving orientation and watertightness.

Everything is a pure function of the spec — no RNG — so meshes are
bit-for-bit deterministic and the vitest twin (``web/src/model/
club.ts``) can pin identical numbers.

:func:`face_normal_at_offset` exposes the face-curvature normal for the
future impact package: for the pre-loft surface
``x = x_face - s_bulge(z) - s_roll(y)`` the outward normal is

    n ∝ (1,  y / sqrt(R_roll² - y²),  z / sqrt(R_bulge² - z²))

(the gradient of the surface function), rotated by the loft tilt. This
analytic rotated form is the contract: the leaned mesh realizes it
exactly for flat faces (all blades), and to first order in the sagitta
slope for curved wood faces (a shear and a rotation of a curved surface
agree only to first order). With curvature off the corresponding
component is zero — a flat lofted face.
"""

from __future__ import annotations

import math

import numpy as np

from rate_of_closure._contracts import ensure, require, require_finite
from rate_of_closure.mesh import HeadMesh, triangle_normals

from .geometry import RING_POINTS, cap_fan, loft_band, superellipse_ring
from .head_profiles import leading_edge_height, mass_scale, profile_for
from .types import ClubSpec

__all__ = [
    "BASE_SECTIONS",
    "REFERENCE_HEAD_MASS_KG",
    "build_parametric_head",
    "face_normal_at_offset",
    "face_sagitta",
    "parametric_head_mesh",
]

#: Head mass the wood envelope proportions represent (a 200 g driver).
REFERENCE_HEAD_MASS_KG = 0.200

#: Wood loft cross-sections at reference mass: (x, half-height,
#: half-width). Kept for the strike view's face-extent derivation;
#: type-specific profiles live in :mod:`.head_profiles`.
BASE_SECTIONS: tuple[tuple[float, float, float], ...] = (
    (0.055, 0.028, 0.058),  # face plate
    (0.010, 0.031, 0.062),  # crown bulge
    (-0.035, 0.024, 0.048),  # rear taper
    (-0.055, 0.010, 0.020),  # tail
)

#: Concentric face-patch rings as fractions of the face boundary.
_FACE_FRACTIONS: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2)
#: Longitudinal subdivisions between authored profile stations.
_BODY_SUBDIVISIONS = 3


def _refined_sections(
    sections: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Linearly subdivide profile stations without changing the envelope."""
    refined: list[tuple[float, float, float, float]] = []
    for first, second in zip(sections[:-1], sections[1:], strict=True):
        for step in range(_BODY_SUBDIVISIONS):
            fraction = step / _BODY_SUBDIVISIONS
            refined.append(
                (
                    first[0] + fraction * (second[0] - first[0]),
                    first[1] + fraction * (second[1] - first[1]),
                    first[2] + fraction * (second[2] - first[2]),
                    first[3] + fraction * (second[3] - first[3]),
                )
            )
    refined.append(sections[-1])
    return refined


def _sagitta(radius_m: float | None, offset_m: float) -> float:
    """Circular sagitta ``R - sqrt(R² - t²)``; zero for a flat face."""
    if radius_m is None:
        return 0.0
    require(
        abs(offset_m) < radius_m,
        "offset must be inside the curvature radius",
        offset_m,
    )
    return radius_m - math.sqrt(radius_m**2 - offset_m**2)


def face_sagitta(spec: ClubSpec, toe_m: float, high_m: float) -> float:
    """Face set-back [m] at an offset from face center.

    ``toe_m`` is the horizontal offset toward the toe (bulge direction),
    ``high_m`` the vertical offset toward the crown (roll direction).
    Zero everywhere when both radii are ``None`` (flat face).
    """
    require_finite(toe_m, name="toe_m")
    require_finite(high_m, name="high_m")
    sag = _sagitta(spec.face_bulge_radius_m, toe_m) + _sagitta(
        spec.face_roll_radius_m, high_m
    )
    ensure(sag >= 0.0, "sagitta is non-negative")
    return sag


def _loft_rotation(loft_deg: float) -> np.ndarray:
    """Rotation about +z tilting the face normal up by the loft angle.

    Used for **normals only** (:func:`face_normal_at_offset`); mesh
    positions are lofted by the leading-edge lean map instead
    (#4799 G1 — see :func:`build_parametric_head`).
    """
    lam = math.radians(loft_deg)
    rotation: np.ndarray = np.array(
        [
            [math.cos(lam), -math.sin(lam), 0.0],
            [math.sin(lam), math.cos(lam), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rotation


def face_normal_at_offset(
    spec: ClubSpec, toe_mm: float, high_mm: float
) -> tuple[float, float, float]:
    """Outward unit face normal at an impact offset from face center.

    Args:
        spec: The club whose face to evaluate.
        toe_mm: Offset toward the toe (+) or heel (-), millimeters.
        high_mm: Offset above (+) or below (-) face center, millimeters.

    Returns:
        Unit vector in the head frame (x target, y up, z toe). At
        center this is ``(cos loft, sin loft, 0)``; bulge tilts it
        toward the toe on toe strikes, roll tilts it further up on
        high strikes — the geometry gear-effect corrections start from.
    """
    require_finite(toe_mm, name="toe_mm")
    require_finite(high_mm, name="high_mm")
    toe_m, high_m = toe_mm * 1.0e-3, high_mm * 1.0e-3

    slope_z = 0.0
    if spec.face_bulge_radius_m is not None:
        radius = spec.face_bulge_radius_m
        require(abs(toe_m) < radius, "toe offset inside bulge radius", toe_m)
        slope_z = toe_m / math.sqrt(radius**2 - toe_m**2)
    slope_y = 0.0
    if spec.face_roll_radius_m is not None:
        radius = spec.face_roll_radius_m
        require(abs(high_m) < radius, "high offset inside roll radius", high_m)
        slope_y = high_m / math.sqrt(radius**2 - high_m**2)

    local = np.array([1.0, slope_y, slope_z])
    normal = _loft_rotation(spec.loft_deg) @ (local / np.linalg.norm(local))
    ensure(
        bool(np.isclose(np.linalg.norm(normal), 1.0, rtol=1e-12)),
        "face normal is unit length",
    )
    return (float(normal[0]), float(normal[1]), float(normal[2]))


def build_parametric_head(spec: ClubSpec) -> np.ndarray:
    """Representative head triangles ``(n, 3, 3)`` for a spec, meters.

    Deterministic: identical specs produce bit-identical arrays. The
    cross-sections come from the club type's :class:`~rate_of_closure.
    club.head_profiles.HeadProfile` (woods, hybrids, iron/wedge blades,
    mallet and blade putters), so the silhouette is recognizably
    type-specific. The mesh is always closed: for ``n`` sections it has
    ``(2 (n - 1) + 6) * RING_POINTS`` triangles — body bands, two
    face-patch bands, and the face/tail fans.
    """
    profile = profile_for(spec)
    scale = mass_scale(spec)
    authored_sections = [
        (x * scale, hh * scale, hw * scale, yc * scale)
        for x, hh, hw, yc in profile.sections
    ]
    sections = _refined_sections(authored_sections)

    def body_ring(section: tuple[float, float, float, float]) -> np.ndarray:
        x, hh, hw, yc = section
        ring: np.ndarray = superellipse_ring(x, hh, hw)
        ring[:, 1] += yc
        return ring

    rings = [body_ring(section) for section in sections]
    face_x, _hh0, _hw0, face_yc = sections[0]
    center = np.array([face_x, face_yc, 0.0])

    def face_ring(fraction: float) -> np.ndarray:
        """One concentric unlofted face ring with its curvature set-back."""
        scaled = rings[0].copy()
        scaled[:, 1] = face_yc + (scaled[:, 1] - face_yc) * fraction
        scaled[:, 2] *= fraction
        for row in scaled:
            row[0] = face_x - face_sagitta(spec, float(row[2]), float(row[1] - face_yc))
        return scaled

    face_rings = [face_ring(fraction) for fraction in _FACE_FRACTIONS]
    face_center = center.copy()  # sagitta at (0, 0) is zero

    triangles: list[np.ndarray] = []
    # Body: lofted bands from the (tilted) face boundary back to the
    # tail, flipped so they face radially outward — with the outward
    # face patch and caps this makes the whole solid consistently
    # outward-wound, as the divergence-theorem volumetrics require.
    body_rings = [face_rings[0], *rings[1:]]
    for ring_a, ring_b in zip(body_rings[:-1], body_rings[1:], strict=True):
        triangles.extend(loft_band(ring_a, ring_b, flip=True))
    # Face patch: concentric bands closing onto the center fan.
    for outer, inner in zip(face_rings[:-1], face_rings[1:], strict=True):
        triangles.extend(loft_band(outer, inner))
    triangles.extend(cap_fan(face_center, face_rings[-1], outward_x=True))
    # Tail cap; a positive recess pulls the fan center inward (+x),
    # forming the cavity-back recess on irons.
    tail_x, _hh, _hw, tail_yc = sections[-1]
    tail_center = np.array([tail_x + profile.rear_recess_m * scale, tail_yc, 0.0])
    triangles.extend(cap_fan(tail_center, rings[-1], outward_x=False))

    mesh: np.ndarray = np.array(triangles)
    # Leading-edge loft lean (#4799 G1): shear the assembled unlofted
    # solid about the y = y_le leading-edge line instead of rotating the
    # face patch about its center, so the leading edge keeps the
    # authored face station (no onset) and the authored face height
    # becomes slant height. Same map as head_profiles.lean_point;
    # Jacobian det = cos(loft) > 0 keeps the winding outward.
    lam = math.radians(spec.loft_deg)
    y_le = leading_edge_height(spec)
    dy = mesh[:, :, 1] - y_le
    mesh[:, :, 0] -= dy * math.sin(lam)
    mesh[:, :, 1] = y_le + dy * math.cos(lam)
    expected = (2 * (len(sections) - 1) + 2 * (len(face_rings) - 1) + 2) * RING_POINTS
    ensure(mesh.shape[0] == expected, "parametric head is closed")
    ensure(bool(np.isfinite(mesh).all()), "parametric head vertices finite")
    return mesh


def parametric_head_mesh(spec: ClubSpec) -> HeadMesh:
    """Renderable :class:`HeadMesh` for a spec (canonical head frame).

    The triangles come straight from :func:`build_parametric_head` —
    already centered near the origin in the canonical frame with a
    mass-plausible envelope — so no normalization pass is applied and
    the head's size follows the spec.
    """
    triangles = build_parametric_head(spec)
    return HeadMesh(triangles=triangles, normals=triangle_normals(triangles))
