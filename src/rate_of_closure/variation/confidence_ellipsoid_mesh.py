"""Deterministic bounded confidence-ellipsoid surface geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.python.contracts import require
from shared.python.swing_sim.variation import (
    ESTIMABLE,
    GAUSSIAN_POSITION_CONTENT_REGION,
)
from shared.python.swing_sim.variation.ensemble_types import immutable_array

APP_FRAME_ID = "app_frame:x_target,y_up,z_right"
MAX_ELLIPSOID_LONGITUDE_SEGMENTS = 12
MAX_ELLIPSOID_LATITUDE_SEGMENTS = 6
ELLIPSOID_LONGITUDE_SEGMENTS = MAX_ELLIPSOID_LONGITUDE_SEGMENTS
ELLIPSOID_LATITUDE_SEGMENTS = MAX_ELLIPSOID_LATITUDE_SEGMENTS
MAX_RENDERED_ELLIPSOIDS = 48
MAX_ELLIPSOID_VERTICES = 2_976
MAX_ELLIPSOID_TRIANGLES = 5_760


@dataclass(frozen=True)
class ConfidenceEllipsoidMesh:
    """One bounded triangle mesh in SI units and the declared app frame."""

    coordinate_frame: str
    interpretation: str
    sample_indices: tuple[int, ...]
    vertices_m: np.ndarray = field(repr=False)
    triangles: np.ndarray = field(repr=False)
    vertices_per_ellipsoid: int
    triangles_per_ellipsoid: int

    def __post_init__(self) -> None:
        require(isinstance(self.vertices_m, np.ndarray), "vertices must be an array")
        require(isinstance(self.triangles, np.ndarray), "triangles must be an array")
        vertices = np.asarray(self.vertices_m)
        triangles = np.asarray(self.triangles)
        require(self.coordinate_frame == APP_FRAME_ID, "invalid coordinate frame")
        require(
            self.interpretation == GAUSSIAN_POSITION_CONTENT_REGION,
            "invalid interpretation",
        )
        require(type(self.sample_indices) is tuple, "samples must be a tuple")
        require(
            all(type(index) is int and index >= 0 for index in self.sample_indices),
            "samples must be genuine non-negative integers",
        )
        require(
            len(self.sample_indices) <= MAX_RENDERED_ELLIPSOIDS,
            "sample count exceeds the hard render cap",
        )
        _bounded_integer(
            self.vertices_per_ellipsoid,
            "vertices_per_ellipsoid",
            1,
            MAX_ELLIPSOID_VERTICES,
        )
        _bounded_integer(
            self.triangles_per_ellipsoid,
            "triangles_per_ellipsoid",
            1,
            MAX_ELLIPSOID_TRIANGLES,
        )
        require(vertices.ndim == 2 and vertices.shape[1:] == (3,), "invalid vertices")
        require(
            triangles.ndim == 2 and triangles.shape[1:] == (3,), "invalid triangles"
        )
        require(vertices.shape[0] <= MAX_ELLIPSOID_VERTICES, "too many vertices")
        require(triangles.shape[0] <= MAX_ELLIPSOID_TRIANGLES, "too many triangles")
        require(
            np.issubdtype(vertices.dtype, np.number) and np.isrealobj(vertices),
            "vertices must be real numbers",
        )
        require(bool(np.all(np.isfinite(vertices))), "vertices must be finite")
        require(
            np.issubdtype(triangles.dtype, np.integer), "triangles must be integers"
        )
        require(
            tuple(sorted(set(self.sample_indices))) == self.sample_indices,
            "invalid samples",
        )
        require(self.vertices_per_ellipsoid > 0, "invalid vertex count")
        require(self.triangles_per_ellipsoid > 0, "invalid triangle count")
        require(
            vertices.shape[0] == len(self.sample_indices) * self.vertices_per_ellipsoid,
            "vertex count mismatch",
        )
        require(
            triangles.shape[0]
            == len(self.sample_indices) * self.triangles_per_ellipsoid,
            "triangle count mismatch",
        )
        if triangles.size:
            require(int(triangles.min()) >= 0, "triangle index must be non-negative")
            require(
                int(triangles.max()) < vertices.shape[0], "triangle index out of range"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            owned_vertices = immutable_array(vertices, float)
            owned_triangles = immutable_array(triangles, int)
        require(bool(np.all(np.isfinite(owned_vertices))), "vertices must be finite")
        object.__setattr__(self, "vertices_m", owned_vertices)
        object.__setattr__(self, "triangles", owned_triangles)


def build_confidence_ellipsoid_mesh(
    centers_m: np.ndarray,
    principal_axes: np.ndarray,
    semi_axis_lengths_m: np.ndarray,
    adequacy: tuple[str, ...],
    coordinate_frame: str,
    *,
    longitude_segments: int = ELLIPSOID_LONGITUDE_SEGMENTS,
    latitude_segments: int = ELLIPSOID_LATITUDE_SEGMENTS,
    max_ellipsoids: int = MAX_RENDERED_ELLIPSOIDS,
    max_vertices: int = MAX_ELLIPSOID_VERTICES,
    max_triangles: int = MAX_ELLIPSOID_TRIANGLES,
) -> ConfidenceEllipsoidMesh:
    """Build finite surfaces for estimable samples within fixed geometry budgets."""
    vertices_per_ellipsoid, triangles_per_ellipsoid = _validated_mesh_budget(
        longitude_segments,
        latitude_segments,
        max_ellipsoids,
        max_vertices,
        max_triangles,
    )
    centers = np.asarray(centers_m)
    axes = np.asarray(principal_axes)
    semi_axes = np.asarray(semi_axis_lengths_m)
    _validate_inputs(centers, axes, semi_axes, adequacy, coordinate_frame)
    capacity = min(
        max_ellipsoids,
        max_vertices // vertices_per_ellipsoid,
        max_triangles // triangles_per_ellipsoid,
    )
    eligible = tuple(
        index for index, state in enumerate(adequacy) if state == ESTIMABLE
    )
    selected = _decimated_indices(eligible, capacity)
    vertices: tuple[np.ndarray, ...]
    triangles: tuple[np.ndarray, ...]
    if selected:
        unit_vertices, unit_triangles = _unit_sphere(
            longitude_segments, latitude_segments
        )
        with np.errstate(over="ignore", invalid="ignore"):
            vertices = tuple(
                centers[index] + (axes[index] @ (semi_axes[index] * unit_vertices).T).T
                for index in selected
            )
        offsets = tuple(
            index * vertices_per_ellipsoid for index in range(len(selected))
        )
        triangles = tuple(unit_triangles + offset for offset in offsets)
    else:
        vertices = ()
        triangles = ()
    return ConfidenceEllipsoidMesh(
        coordinate_frame=coordinate_frame,
        interpretation=GAUSSIAN_POSITION_CONTENT_REGION,
        sample_indices=selected,
        vertices_m=(np.concatenate(vertices) if vertices else np.empty((0, 3))),
        triangles=(
            np.concatenate(triangles) if triangles else np.empty((0, 3), dtype=int)
        ),
        vertices_per_ellipsoid=vertices_per_ellipsoid,
        triangles_per_ellipsoid=triangles_per_ellipsoid,
    )


def _validated_mesh_budget(
    longitudes: int,
    latitudes: int,
    max_ellipsoids: int,
    max_vertices: int,
    max_triangles: int,
) -> tuple[int, int]:
    """Validate genuine integers and return allocation-free per-surface counts."""
    _bounded_integer(
        longitudes, "longitude_segments", 3, MAX_ELLIPSOID_LONGITUDE_SEGMENTS
    )
    _bounded_integer(latitudes, "latitude_segments", 2, MAX_ELLIPSOID_LATITUDE_SEGMENTS)
    _bounded_integer(max_ellipsoids, "max_ellipsoids", 0, MAX_RENDERED_ELLIPSOIDS)
    _bounded_integer(max_vertices, "max_vertices", 0, MAX_ELLIPSOID_VERTICES)
    _bounded_integer(max_triangles, "max_triangles", 0, MAX_ELLIPSOID_TRIANGLES)
    ring_count = latitudes - 1
    return 2 + ring_count * longitudes, 2 * ring_count * longitudes


def _bounded_integer(value: int, name: str, minimum: int, maximum: int) -> int:
    require(type(value) is int, f"{name} must be an integer")
    require(minimum <= value <= maximum, f"{name} exceeds its hard bounds")
    return value


def _validate_inputs(
    centers: np.ndarray,
    axes: np.ndarray,
    semi_axes: np.ndarray,
    adequacy: tuple[str, ...],
    coordinate_frame: str,
) -> None:
    samples = centers.shape[0] if centers.ndim == 2 else -1
    require(coordinate_frame == APP_FRAME_ID, "invalid coordinate frame")
    require(centers.shape == (samples, 3), "invalid centers")
    require(axes.shape == (samples, 3, 3), "invalid principal axes")
    require(semi_axes.shape == (samples, 3), "invalid semi axes")
    require(len(adequacy) == samples, "adequacy length mismatch")
    for index, state in enumerate(adequacy):
        if state != ESTIMABLE:
            continue
        require(
            bool(np.all(np.isfinite(centers[index]))),
            "estimable center must be finite",
        )
        require(
            bool(np.all(np.isfinite(axes[index]))),
            "estimable axes must be finite",
        )
        require(
            bool(np.all(np.isfinite(semi_axes[index]))),
            "estimable semi axes must be finite",
        )
        require(
            bool(np.all(semi_axes[index] > 0)),
            "estimable semi axes must be positive",
        )
        require(
            np.allclose(axes[index].T @ axes[index], np.eye(3), rtol=0.0, atol=1e-10),
            "estimable axes must be orthonormal",
        )


def _unit_sphere(longitudes: int, latitudes: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.pi * np.arange(1, latitudes) / latitudes
    phi = 2.0 * np.pi * np.arange(longitudes) / longitudes
    rings = np.stack(
        (
            (np.sin(theta)[:, None] * np.cos(phi)[None, :]),
            (np.sin(theta)[:, None] * np.sin(phi)[None, :]),
            np.broadcast_to(np.cos(theta)[:, None], (latitudes - 1, longitudes)),
        ),
        axis=-1,
    ).reshape(-1, 3)
    vertices = np.vstack(([0.0, 0.0, 1.0], rings, [0.0, 0.0, -1.0]))
    triangles: list[tuple[int, int, int]] = []
    for longitude in range(longitudes):
        next_index = (longitude + 1) % longitudes
        triangles.append((0, 1 + longitude, 1 + next_index))
    for latitude in range(latitudes - 2):
        first = 1 + latitude * longitudes
        second = first + longitudes
        for longitude in range(longitudes):
            next_index = (longitude + 1) % longitudes
            triangles.extend(
                (
                    (first + longitude, second + longitude, second + next_index),
                    (first + longitude, second + next_index, first + next_index),
                )
            )
    south = vertices.shape[0] - 1
    last_ring = 1 + (latitudes - 2) * longitudes
    for longitude in range(longitudes):
        next_index = (longitude + 1) % longitudes
        triangles.append((south, last_ring + next_index, last_ring + longitude))
    return vertices, np.asarray(triangles, dtype=int)


def _decimated_indices(indices: tuple[int, ...], capacity: int) -> tuple[int, ...]:
    if capacity <= 0 or not indices:
        return ()
    if len(indices) <= capacity:
        return indices
    if capacity == 1:
        return (indices[0],)
    positions = np.floor(
        np.arange(capacity) * (len(indices) - 1) / (capacity - 1)
    ).astype(int)
    return tuple(indices[position] for position in positions)


__all__ = [
    "APP_FRAME_ID",
    "ConfidenceEllipsoidMesh",
    "MAX_ELLIPSOID_LATITUDE_SEGMENTS",
    "MAX_ELLIPSOID_LONGITUDE_SEGMENTS",
    "MAX_ELLIPSOID_TRIANGLES",
    "MAX_ELLIPSOID_VERTICES",
    "MAX_RENDERED_ELLIPSOIDS",
    "build_confidence_ellipsoid_mesh",
]
