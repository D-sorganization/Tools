"""Shared superellipse-loft mesh helpers.

The procedural clubhead meshes (the bundled example head and the
parametric heads generated from a :class:`~rate_of_closure.club.types.
ClubSpec`) are all built the same way: superellipse cross-section rings
lofted along the face-to-back axis, banded with quads split into
triangles, and capped with triangle fans. These helpers own that
geometry so both generators stay DRY.

Frame: the AffineDrift head frame — x face-forward (target), y up,
z toward the toe (right of target for a right-handed golfer).

Winding: :func:`cap_fan` picks its facing explicitly; :func:`loft_band`
defaults to the face-patch orientation and takes ``flip=True`` for
radially-outward body bands (see its docstring), so a generator can
produce a consistently outward-wound, watertight solid.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from rate_of_closure._contracts import require

FloatArray: TypeAlias = npt.NDArray[np.float64]

__all__ = [
    "RING_POINTS",
    "SUPERELLIPSE_EXPONENT",
    "cap_fan",
    "loft_band",
    "superellipse_ring",
]

#: Vertices per cross-section ring. Sixty-four keeps silhouettes smooth on
#: high-DPI displays while remaining inexpensive for interactive redraws.
RING_POINTS = 64
#: Superellipse exponent (4 = rounded-rectangle sections).
SUPERELLIPSE_EXPONENT = 4.0


def superellipse_ring(
    x: float,
    half_height: float,
    half_width: float,
    points: int = RING_POINTS,
    exponent: float = SUPERELLIPSE_EXPONENT,
) -> FloatArray:
    """Superellipse cross-section ring in the (y, z) plane at ``x``.

    Returns ``(points, 3)`` vertices circling from the toe (+z at
    theta 0) through the crown (+y).
    """
    require(half_height > 0.0 and half_width > 0.0, "ring half-extents positive")
    require(points >= 3, "a ring needs at least 3 points", points)
    theta = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    power = 2.0 / exponent
    y = half_height * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** power
    z = half_width * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** power
    ring: FloatArray = np.column_stack([np.full(points, x), y, z])
    return ring


def loft_band(
    ring_a: np.ndarray, ring_b: np.ndarray, flip: bool = False
) -> list[np.ndarray]:
    """Two triangles per quad between consecutive rings.

    ``ring_a`` is the ring nearer the face (+x) — or the outer ring of
    a face patch — and ``ring_b`` the next ring toward the tail (or
    face center). With the default winding the band faces +x-ward
    (face-patch orientation); ``flip=True`` reverses each triangle so a
    body band between successive loft sections faces radially outward
    — the orientation the watertight volumetrics require.
    """
    require(ring_a.shape == ring_b.shape, "rings must match", ring_a.shape)
    n = ring_a.shape[0]
    triangles: list[np.ndarray] = []
    for i in range(n):
        j = (i + 1) % n
        quad = (
            (np.array([ring_a[i], ring_b[i], ring_b[j]])),
            (np.array([ring_a[i], ring_b[j], ring_a[j]])),
        )
        if flip:
            quad = (quad[0][::-1].copy(), quad[1][::-1].copy())
        triangles.extend(quad)
    return triangles


def cap_fan(center: np.ndarray, ring: np.ndarray, outward_x: bool) -> list[np.ndarray]:
    """Triangle fan from ``center`` to ``ring``.

    ``outward_x=True`` winds the fan so its normals point +x (a face
    cap); ``False`` points them -x (a tail cap).
    """
    n = ring.shape[0]
    triangles: list[np.ndarray] = []
    for i in range(n):
        j = (i + 1) % n
        if outward_x:
            triangles.append(np.array([center, ring[j], ring[i]]))
        else:
            triangles.append(np.array([center, ring[i], ring[j]]))
    return triangles
