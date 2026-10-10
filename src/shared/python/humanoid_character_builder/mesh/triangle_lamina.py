"""Unsigned triangle-multiset mass moments, independently of solid-volume gates.

Each positive-area triangle receives its area fraction of a specified mass.
Coincident triangles count with multiplicity; this is NOT surface-union density.
No mesh repair, thickness, volume, anatomical inference or tensor projection is
performed. Coordinates must already inhabit one orthonormal frame in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from shared.python.humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    InertiaResult,
)


@dataclass(frozen=True, slots=True)
class TriangleLaminaResult:
    """COM inertia and source-multiset diagnostics; no solid qualification.

    Boundary/nonmanifold counts use source vertex indices and positive-area
    faces. Coincident-face counts instead use exact vertex coordinates, ignoring
    winding. Partial overlap and self-intersection are not tested. The embedded
    existing inertia representation remains mutable for compatibility.
    """

    inertia: InertiaResult
    total_area_m2: float
    zero_area_triangles: int
    duplicate_positive_area_triangles: int
    boundary_edges: int
    nonmanifold_edges: int


def _validated_triangles(
    vertices: NDArray[np.float64], faces: NDArray[np.integer], mass: float
) -> NDArray[np.float64]:
    """Validate the public boundary before any indexing or arithmetic."""
    if isinstance(mass, (bool, np.bool_)) or not isinstance(
        mass, (int, float, np.integer, np.floating)
    ):
        raise TypeError("mass must be a positive finite scalar")
    if not np.isfinite(float(mass)) or float(mass) <= 0:
        raise ValueError("mass must be a positive finite scalar")
    if np.iscomplexobj(vertices):
        raise ValueError("vertices must contain real coordinates")
    points = np.asarray(vertices, dtype=np.float64)
    indices = np.asarray(faces)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("vertices must have finite shape (N, 3)")
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("faces must have shape (T, 3)")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("faces must contain integer vertex indices")
    if np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError("faces contain an out-of-range vertex index")
    return points[indices]


def _area_weights(
    triangles: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, NDArray[np.bool_]]:
    """Return unsigned area weights without discarding any nonzero face."""
    edges = triangles[:, 1:] - triangles[:, :1]
    cross = np.cross(edges[:, 0], edges[:, 1])
    areas = np.hypot(np.hypot(cross[:, 0], cross[:, 1]), cross[:, 2]) / 2
    # Distinguish true zero-area faces from an unrepresentable nonzero area.
    scales = np.max(np.abs(edges), axis=2, keepdims=True)
    normalized = np.divide(edges, scales, out=np.zeros_like(edges), where=scales > 0)
    normalized_cross = np.cross(normalized[:, 0], normalized[:, 1])
    if np.any((areas == 0) & np.any(normalized_cross != 0, axis=1)):
        raise ValueError("area is nonzero but not representable in float64")
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("area must have a positive finite total")
    weights = areas / total_area
    if np.any((areas > 0) & (weights == 0)):
        raise ValueError("area fraction is not representable in float64")
    return weights, total_area, areas > 0


def _centered_moments(
    triangles: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Accumulate exact within/between-triangle covariance near source origin."""
    # A remote negligible-area face must not choose a destructive origin.
    origin = triangles[int(np.argmax(weights)), 0]
    relative = triangles - origin
    centroids = relative.mean(axis=1)
    center = np.einsum("t,ti->i", weights, centroids)
    local_vertices = relative - centroids[:, None, :]
    # Uniform triangle barycentric covariance: sum(v_i-c)(v_i-c)^T / 12.
    within = np.einsum("t,tvi,tvj->ij", weights, local_vertices, local_vertices) / 12
    delta = centroids - center
    covariance = within + np.einsum("t,ti,tj->ij", weights, delta, delta)
    return origin + center, covariance


def _topology_counts(
    triangles: NDArray[np.float64], faces: NDArray[np.integer]
) -> tuple[int, int, int]:
    """Count exact duplicates and indexed edge degree; never alter geometry."""
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]))
    ordered = np.take_along_axis(triangles, order[:, :, None], axis=1)
    distinct_faces = np.unique(ordered.reshape(-1, 9), axis=0)
    duplicate_count = len(triangles) - len(distinct_faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, degree = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return (
        duplicate_count,
        int(np.count_nonzero(degree == 1)),
        int(np.count_nonzero(degree > 2)),
    )


def compute_triangle_lamina(
    vertices: NDArray[np.float64], faces: NDArray[np.integer], mass: float
) -> TriangleLaminaResult:
    """Compute COM and COM inertia for a declared triangle-multiset lamina.

    Args:
        vertices: Finite (N,3) source positions, metres in one orthonormal frame.
        faces: Integer (T,3) indices; duplicated triangles retain multiplicity.
        mass: Positive finite total mass, kg, owned once by this body.

    Returns:
        Area-weighted COM (m), inertia (kg m²) and explicit topology diagnostics.
        Inputs unchanged; volume=0 denotes zero thickness and makes no solid claim.

    Raises:
        TypeError: Noninteger faces or nonscalar/boolean mass.
        ValueError: Nonfinite/invalid geometry, mass, area or output moments.

    No source/frame/tissue/overlap/native qualification is inferred.
    """
    triangles = _validated_triangles(vertices, faces, mass)
    weights, total_area, positive = _area_weights(triangles)
    center, covariance = _centered_moments(triangles[positive], weights[positive])
    tensor = mass * (np.trace(covariance) * np.eye(3) - covariance)
    if not np.isfinite(center).all() or not np.isfinite(tensor).all():
        raise ValueError("vertices produced nonfinite mass moments")
    duplicate_count, boundary, nonmanifold = _topology_counts(
        triangles[positive], np.asarray(faces)[positive]
    )
    inertia = InertiaResult(
        ixx=float(tensor[0, 0]),
        iyy=float(tensor[1, 1]),
        izz=float(tensor[2, 2]),
        ixy=float(tensor[0, 1]),
        ixz=float(tensor[0, 2]),
        iyz=float(tensor[1, 2]),
        center_of_mass=tuple(float(value) for value in center),
        mass=float(mass),
        volume=0.0,
        was_watertight=False,
        mode=InertiaMode.TRIANGLE_MULTISET_LAMINA_MASS,
    )
    return TriangleLaminaResult(
        inertia,
        total_area,
        int(np.count_nonzero(~positive)),
        duplicate_count,
        boundary,
        nonmanifold,
    )
