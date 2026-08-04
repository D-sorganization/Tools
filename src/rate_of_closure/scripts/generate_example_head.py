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

import numpy as np

from rate_of_closure._contracts import ensure
from rate_of_closure.mesh import write_binary_stl

__all__ = ["ASSET_PATH", "build_example_head", "main"]

#: Where the generated STL ships inside the package.
ASSET_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / ("example_driver_head.stl")
)

_RING_POINTS = 24
_SUPERELLIPSE_EXPONENT = 4.0

#: Loft cross-sections: (x [m], half-height [m], half-width [m]).
_SECTIONS: tuple[tuple[float, float, float], ...] = (
    (0.055, 0.028, 0.058),  # face plate
    (0.010, 0.031, 0.062),  # crown bulge
    (-0.035, 0.024, 0.048),  # rear taper
    (-0.055, 0.010, 0.020),  # tail
)


def _ring(x: float, half_height: float, half_width: float) -> np.ndarray:
    """Superellipse cross-section ring in the (y, z) plane at ``x``."""
    theta = np.linspace(0.0, 2.0 * np.pi, _RING_POINTS, endpoint=False)
    power = 2.0 / _SUPERELLIPSE_EXPONENT
    y = half_height * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** power
    z = half_width * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** power
    return np.column_stack([np.full(_RING_POINTS, x), y, z])


def build_example_head() -> np.ndarray:
    """Triangles ``(n, 3, 3)`` of the stylized driver head, meters."""
    rings = [_ring(*section) for section in _SECTIONS]
    triangles: list[np.ndarray] = []

    # Side bands: two triangles per quad between consecutive rings.
    for ring_a, ring_b in zip(rings[:-1], rings[1:], strict=True):
        for i in range(_RING_POINTS):
            j = (i + 1) % _RING_POINTS
            triangles.append(np.array([ring_a[i], ring_b[i], ring_b[j]]))
            triangles.append(np.array([ring_a[i], ring_b[j], ring_a[j]]))

    # Caps: triangle fans to the section centers (face and tail).
    face_center = np.array([_SECTIONS[0][0], 0.0, 0.0])
    tail_center = np.array([_SECTIONS[-1][0], 0.0, 0.0])
    for i in range(_RING_POINTS):
        j = (i + 1) % _RING_POINTS
        triangles.append(np.array([face_center, rings[0][j], rings[0][i]]))
        triangles.append(np.array([tail_center, rings[-1][i], rings[-1][j]]))

    mesh = np.array(triangles)
    # (sections-1) bands of 2*N triangles plus two N-triangle caps.
    ensure(mesh.shape[0] == len(_SECTIONS) * 2 * _RING_POINTS, "loft closed")
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
