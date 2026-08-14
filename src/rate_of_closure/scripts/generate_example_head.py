"""Generate the bundled example clubhead STL — no third-party geometry.

The mesh is a stylized driver head built entirely from parametric
superellipse cross-sections lofted along the face-to-back axis, so the
shipped file carries zero licensing risk (nothing is downloaded or
traced from a manufacturer model). Proportions follow the procedural
wireframe: 0.11 m deep, 0.124 m wide, 0.062 m tall, face plate at +x.

Run as a module to (re)generate the asset::

    python -m rate_of_closure.scripts.generate_example_head
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from rate_of_closure._contracts import ensure
from rate_of_closure.club.geometry import (
    RING_POINTS,
    cap_fan,
    loft_band,
    superellipse_ring,
)
from rate_of_closure.club.parametric_head import BASE_SECTIONS
from rate_of_closure.mesh import write_binary_stl

FloatArray: TypeAlias = npt.NDArray[np.float64]

__all__ = ["ASSET_PATH", "build_example_head", "main"]

#: Where the generated STL ships inside the package.
ASSET_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / ("example_driver_head.stl")
)

#: Loft cross-sections: (x [m], half-height [m], half-width [m]) —
#: shared with the parametric generator (the 200 g reference envelope).
_SECTIONS = BASE_SECTIONS


def build_example_head() -> FloatArray:
    """Triangles ``(n, 3, 3)`` of the stylized driver head, meters."""
    rings = [superellipse_ring(*section) for section in _SECTIONS]
    triangles: list[np.ndarray] = []

    # Side bands: two triangles per quad between consecutive rings.
    for ring_a, ring_b in zip(rings[:-1], rings[1:], strict=True):
        triangles.extend(loft_band(ring_a, ring_b))

    # Caps: triangle fans to the section centers (face and tail).
    face_center = np.array([_SECTIONS[0][0], 0.0, 0.0])
    tail_center = np.array([_SECTIONS[-1][0], 0.0, 0.0])
    triangles.extend(cap_fan(face_center, rings[0], outward_x=True))
    triangles.extend(cap_fan(tail_center, rings[-1], outward_x=False))

    mesh: FloatArray = np.array(triangles)
    # (sections-1) bands of 2*N triangles plus two N-triangle caps.
    ensure(mesh.shape[0] == len(_SECTIONS) * 2 * RING_POINTS, "loft closed")
    return mesh


def main() -> None:
    """Write the example STL asset next to the package."""
    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    ASSET_PATH.write_bytes(
        write_binary_stl(build_example_head(), header="rate_of_closure example head")
    )
    logging.getLogger(__name__).info("wrote %s", ASSET_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
