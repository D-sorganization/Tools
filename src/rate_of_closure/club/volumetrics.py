"""Closed-mesh volume and centroid via the divergence theorem (H1, #4125).

For a watertight triangle mesh with outward winding, the divergence
theorem turns the volume integral into a surface sum of signed
tetrahedra to the origin:

    V   = Σ det(a, b, c) / 6
    COG = Σ V_i · (a_i + b_i + c_i) / 4  /  V

(each tetrahedron's centroid is the average of its four vertices; the
origin contributes zero). Both are exact for polyhedra and independent
of the origin's location as long as the mesh is closed.

Design by Contract: :func:`mesh_volume_centroid` requires a watertight
mesh — every directed edge must appear exactly once with its reverse
present (checked combinatorially) — and ensures a positive, finite
volume. Validated against analytic solids (cube, UV sphere) to <1% in
``tests/rate_of_closure/test_club_heads.py``.

The TypeScript twin is ``web/src/model/volumetrics.ts``, parity-pinned
on the generated driver head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rate_of_closure._contracts import ensure, require

from .head_profiles import face_center_point
from .parametric_head import build_parametric_head
from .types import ClubSpec

__all__ = [
    "HEAD_VOLUME_BOUNDS_M3",
    "CogReport",
    "head_cog",
    "is_watertight",
    "mesh_volume_centroid",
]

#: Sanity band for generated-head volumes [m³]: from a compact blade
#: putter (~5e-5 = 50 cc) up to the 460 cc USGA driver limit with
#: margin.
HEAD_VOLUME_BOUNDS_M3 = (2.0e-5, 8.0e-4)


def _directed_edges(triangles: np.ndarray) -> dict[tuple[bytes, bytes], int]:
    """Count of each directed edge, keyed by exact vertex bytes."""
    edges: dict[tuple[bytes, bytes], int] = {}
    for tri in triangles:
        keys = [np.ascontiguousarray(v).tobytes() for v in tri]
        for i in range(3):
            edge = (keys[i], keys[(i + 1) % 3])
            edges[edge] = edges.get(edge, 0) + 1
    return edges


def is_watertight(triangles: np.ndarray) -> bool:
    """Whether every directed edge appears once with its reverse present.

    Exact-bit vertex matching: generated meshes share ring vertices
    bit-for-bit, so this is a true closure check for them; independently
    authored STLs with re-tessellated seams may fail and fall back to
    spec CG display.
    """
    tris = np.asarray(triangles, dtype=np.float64)
    require(tris.ndim == 3 and tris.shape[1:] == (3, 3), "triangles must be (n, 3, 3)")
    edges = _directed_edges(tris)
    return all(
        count == 1 and edges.get((b, a), 0) == 1 for (a, b), count in edges.items()
    )


def mesh_volume_centroid(triangles: np.ndarray) -> tuple[float, np.ndarray]:
    """Volume [m³] and centroid [m] of a closed, outward-wound mesh.

    Raises:
        PreconditionError: If the mesh is not watertight.
        PostconditionError: If the signed volume is not positive/finite
            (inward winding or a degenerate solid).
    """
    tris = np.asarray(triangles, dtype=np.float64)
    require(tris.ndim == 3 and tris.shape[1:] == (3, 3), "triangles must be (n, 3, 3)")
    require(bool(np.isfinite(tris).all()), "triangles must be finite")
    require(is_watertight(tris), "mesh must be watertight (closed, matched edges)")

    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    signed = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = float(signed.sum())
    ensure(np.isfinite(volume) and volume > 0.0, "volume must be positive", volume)
    centroid = (signed[:, None] * (a + b + c) / 4.0).sum(axis=0) / volume
    ensure(bool(np.isfinite(centroid).all()), "centroid must be finite")
    return volume, np.asarray(centroid)


@dataclass(frozen=True)
class CogReport:
    """Geometric (volumetric) COG of a generated head vs its spec CG.

    Attributes:
        volume_m3: Enclosed volume of the generated head.
        cog: Uniform-density centroid in the head frame (x target,
            y up, z toe), meters.
        cg_depth_m: Centroid distance back from the forward (face)
            extent — comparable to ``ClubSpec.cg_depth_m``.
        cg_height_m: Centroid height above the sole plane (lowest mesh
            point) — comparable to ``ClubSpec.cg_height_m``.
        spec_cg_depth_m: The spec's published-typical CG depth.
        spec_cg_height_m: The spec's published-typical CG height.
        face_center: Face-plate center in the head frame, meters.
    """

    volume_m3: float
    cog: tuple[float, float, float]
    cg_depth_m: float
    cg_height_m: float
    spec_cg_depth_m: float
    spec_cg_height_m: float
    face_center: tuple[float, float, float]


def head_cog(spec: ClubSpec) -> CogReport:
    """Volumetric COG report for a spec's generated head.

    Computes the divergence-theorem volume and centroid of the
    deterministic parametric head, converts them into the spec-sheet
    conventions (depth back from the face, height above the sole), and
    reports them alongside the spec's own CG values so the two can be
    reconciled — the geometric COG of the uniform-density envelope
    lands in the plausible band of published values per type.
    """
    triangles = build_parametric_head(spec)
    volume, centroid = mesh_volume_centroid(triangles)
    low, high = HEAD_VOLUME_BOUNDS_M3
    ensure(low <= volume <= high, "head volume outside the sane band", volume)
    flat = triangles.reshape(-1, 3)
    depth = float(flat[:, 0].max() - centroid[0])
    height = float(centroid[1] - flat[:, 1].min())
    ensure(depth > 0.0 and height > 0.0, "COG must sit inside the envelope")
    return CogReport(
        volume_m3=volume,
        cog=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        cg_depth_m=depth,
        cg_height_m=height,
        spec_cg_depth_m=spec.cg_depth_m,
        spec_cg_height_m=spec.cg_height_m,
        face_center=face_center_point(spec),
    )
