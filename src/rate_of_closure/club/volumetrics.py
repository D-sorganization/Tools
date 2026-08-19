"""Closed-mesh volume and centroid via the divergence theorem (H1, #4125).

The math authority moved to the shared layer for the club-tester epic
(#4549 C1): :mod:`shared.python.golf_club.mesh_mass_properties` owns
watertightness, volume, centroid, and the full inertia tensor, so
UpstreamDrift reaches one implementation through ``vendor/ud-tools``.
This module keeps the two functions its public API always had as
call-time delegates and the tool-local :func:`head_cog` report on top.

The delegation is deliberately **lazy**: importing the shared function at
module scope executes ``golf_club/__init__``, whose eager surface reaches
SciPy through the turf chain — which breaks the Morris UI import contract
(``test_morris_ui_client``). Same rule as this package's ``__init__``
lazy-export map; do not "simplify" it back to a top-level import.

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
from typing import TYPE_CHECKING

from rate_of_closure._contracts import ensure

from .head_profiles import face_center_point
from .parametric_head import build_parametric_head
from .types import ClubSpec

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "HEAD_VOLUME_BOUNDS_M3",
    "CogReport",
    "head_cog",
    "is_watertight",
    "mesh_volume_centroid",
]


def is_watertight(triangles: np.ndarray) -> bool:
    """Whether every directed edge appears once with its reverse present.

    Delegates to :func:`shared.python.golf_club.mesh_mass_properties.is_watertight`
    (see the module docstring for why the import is call-time).
    """
    from shared.python.golf_club import mesh_mass_properties

    return mesh_mass_properties.is_watertight(triangles)


def mesh_volume_centroid(triangles: np.ndarray) -> tuple[float, np.ndarray]:
    """Volume [m³] and centroid [m] of a closed, outward-wound mesh.

    Delegates to the shared authority; raises the same
    ``PreconditionError`` / ``PostconditionError`` contracts it always
    did (the shim and the shared module share exception classes).
    """
    from shared.python.golf_club import mesh_mass_properties

    return mesh_mass_properties.mesh_volume_centroid(triangles)


#: Sanity band for generated-head volumes [m³]: from a compact blade
#: putter (~5e-5 = 50 cc) up to the 460 cc USGA driver limit with
#: margin.
HEAD_VOLUME_BOUNDS_M3 = (2.0e-5, 8.0e-4)


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
