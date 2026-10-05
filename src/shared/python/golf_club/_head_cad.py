"""Private build123d helpers shared by every exact head family (wedge, iron).

One definition each for the operations two or more head families need:
datum-face recovery from a finished solid, the hollow hosel tube, and the
deterministic single-file artifact writer. Family modules own their profiles,
schemas, and manifests; nothing family-specific belongs here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

MM_PER_M = 1_000.0
_FIXED_STEP_TIMESTAMP = datetime(1970, 1, 1)
_PLANAR_ALIGNMENT_TOLERANCE = 1.0e-9
_BORE_OVERRUN_FACTOR = 2.0  # bore longer than the tube, so the cut is clean


def matching_planar_face(
    solid: Any, expected_normal: np.ndarray
) -> tuple[Any, np.ndarray]:
    """Return the largest face whose normal matches ``expected_normal``.

    Precondition: ``expected_normal`` is a unit vector in the solid frame.
    Postcondition: the returned unit normal is within ``1e-9`` of it, so the
    caller recovers its datum from the solid rather than from its inputs.
    """
    candidates: list[tuple[float, float, Any, np.ndarray]] = []
    for face in solid.faces():
        try:
            normal = np.array(tuple(face.normal_at()), dtype=float)
        except (AttributeError, ValueError):
            continue
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            continue
        unit = normal / norm
        alignment = float(np.dot(unit, expected_normal))
        candidates.append((alignment, float(face.area), face, unit))
    if not candidates:
        raise RuntimeError("solid has no measurable planar faces")
    alignment, _, face, normal = max(candidates, key=lambda item: (item[0], item[1]))
    if alignment < 1.0 - _PLANAR_ALIGNMENT_TOLERANCE:
        raise RuntimeError("requested datum plane was not recovered from the solid")
    return face, normal


def hollow_tube(
    origin_mm: np.ndarray,
    axis: np.ndarray,
    radii_mm: tuple[float, float],
    length_mm: float,
) -> Any:
    """Return an open-topped tube of ``(outer, bore)`` radii along ``axis``.

    The bore runs past the open end, so the result is one annular solid whose
    closed bottom is supplied by whatever body the caller fuses it into.
    """
    from build123d import Plane, Solid

    outer_radius, bore_radius = radii_mm
    assert 0.0 < bore_radius < outer_radius, "tube radii must nest"
    plane = Plane(
        origin=tuple(float(value) for value in origin_mm),
        x_dir=(1.0, 0.0, 0.0),
        z_dir=tuple(float(value) for value in axis),
    )
    outer = Solid.make_cylinder(outer_radius, length_mm, plane=plane)
    bore = Solid.make_cylinder(
        bore_radius, _BORE_OVERRUN_FACTOR * length_mm, plane=plane
    )
    return outer.cut(bore)


def export_solid_file(
    solid: Any,
    path: Path,
    format_name: str,
    tolerances: tuple[float, float],
) -> None:
    """Write one deterministic STEP, STL, or BREP file for ``solid``.

    ``tolerances`` is ``(linear_m, angular_rad)`` and applies to STL only.
    STEP carries a fixed timestamp so identical solids give identical bytes.
    """
    from build123d import export_brep, export_step, export_stl

    linear_m, angular_rad = tolerances
    if format_name == "step":
        succeeded = export_step(solid, path, timestamp=_FIXED_STEP_TIMESTAMP)
    elif format_name == "stl":
        succeeded = export_stl(
            solid,
            path,
            tolerance=linear_m * MM_PER_M,
            angular_tolerance=angular_rad,
            ascii_format=False,
        )
    elif format_name == "brep":
        succeeded = export_brep(solid, path)
    else:
        raise ValueError(f"unsupported export format {format_name!r}")
    if not succeeded or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"failed to export {format_name} artifact")
