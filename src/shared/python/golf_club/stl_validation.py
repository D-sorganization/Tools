"""Dependency-light topology and fidelity validation for binary STL files."""

from __future__ import annotations

import math
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from .cad_validation import (
    M3_PER_MM3,
    MM_PER_M,
    CadGeometryReference,
    Vector3,
    artifact_path_from,
    finite_float,
    max_bounds_error,
    relative_error,
    vector3,
)

Triangle: TypeAlias = tuple[Vector3, Vector3, Vector3]
DirectedEdge: TypeAlias = tuple[Vector3, Vector3]
EdgeMap: TypeAlias = dict[
    tuple[Vector3, Vector3],
    list[tuple[DirectedEdge, int]],
]

_STL_HEADER_BYTES = 84
_STL_TRIANGLE_BYTES = 50
_STL_TRIANGLE_STRUCT = struct.Struct("<12fH")
_MAX_STL_BYTES = 256 * 1024 * 1024
_MIN_NORMAL_ALIGNMENT = 1.0 - 1.0e-5
_BOUNDS_TOLERANCE_FACTOR = 2.0
_VOLUME_TOLERANCE_FACTOR = 4.0
_VOLUME_ERROR_FLOOR = 0.005
_VOLUME_ERROR_CEILING = 0.10


@dataclass(frozen=True)
class StlMeshValidation:
    """Topology, orientation, and geometric fidelity of a binary STL mesh."""

    passed: bool
    reader: str
    triangle_count: int
    unique_vertex_count: int
    connected_component_count: int
    is_watertight: bool
    is_winding_consistent: bool
    has_outward_orientation: bool
    volume_m3: float
    volume_relative_error: float
    allowed_volume_relative_error: float
    max_bounds_error_m: float
    allowed_bounds_error_m: float


def validate_binary_stl(
    path: Path | str,
    reference: CadGeometryReference,
    *,
    linear_tolerance_m: float,
) -> StlMeshValidation:
    """Validate a binary STL against its canonical exact-solid reference."""
    artifact_path = artifact_path_from(path, maximum_bytes=_MAX_STL_BYTES)
    if not isinstance(reference, CadGeometryReference):
        raise TypeError("reference must be CadGeometryReference")
    tolerance = finite_float(
        linear_tolerance_m,
        "linear_tolerance_m",
        positive=True,
    )
    if tolerance >= reference.minimum_span_m:
        raise ValueError("linear_tolerance_m must be smaller than the reference span")
    triangles = _read_binary_stl(artifact_path)
    edges, vertices, signed_volume_mm3 = _inspect_triangles(triangles)
    components = _validate_topology(edges, len(triangles))
    measured = _mesh_reference(vertices, abs(signed_volume_mm3))
    volume_error = relative_error(measured.volume_m3, reference.volume_m3)
    bounds_error = max_bounds_error(measured, reference)
    volume_limit = _volume_error_limit(reference, tolerance)
    bounds_limit = _BOUNDS_TOLERANCE_FACTOR * tolerance
    _require_mesh_match(
        signed_volume_mm3,
        volume_error,
        volume_limit,
        bounds_error,
        bounds_limit,
    )
    return StlMeshValidation(
        passed=True,
        reader="binary-stl/1",
        triangle_count=len(triangles),
        unique_vertex_count=len(vertices),
        connected_component_count=components,
        is_watertight=True,
        is_winding_consistent=True,
        has_outward_orientation=True,
        volume_m3=measured.volume_m3,
        volume_relative_error=volume_error,
        allowed_volume_relative_error=volume_limit,
        max_bounds_error_m=bounds_error,
        allowed_bounds_error_m=bounds_limit,
    )


def read_binary_stl(path: Path | str) -> tuple[Triangle, ...]:
    """Parse a binary STL into vertex triangles (shared STL reading seam).

    The single package-wide binary-STL reader (#4800 P3 promotes it to
    the public surface so `putter_head` never grows a second parser).
    Each triangle record is checked for finiteness, degeneracy, and
    stored-normal/winding agreement exactly as release validation does.

    Raises:
        ValueError: If the byte layout is not a valid binary STL.
        RuntimeError: If a triangle is degenerate or its stored normal
            disagrees with its winding.
    """
    return _read_binary_stl(Path(path))


def _read_binary_stl(path: Path) -> tuple[Triangle, ...]:
    payload = path.read_bytes()
    if len(payload) < _STL_HEADER_BYTES:
        raise ValueError("binary STL is shorter than its header")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    if triangle_count == 0:
        raise ValueError("binary STL must contain at least one triangle")
    expected_bytes = _STL_HEADER_BYTES + _STL_TRIANGLE_BYTES * triangle_count
    if len(payload) != expected_bytes:
        raise ValueError("binary STL byte length does not match triangle count")
    return tuple(
        _triangle_from_record(
            payload,
            _STL_HEADER_BYTES + index * _STL_TRIANGLE_BYTES,
        )
        for index in range(triangle_count)
    )


def _triangle_from_record(payload: bytes, offset: int) -> Triangle:
    values = _STL_TRIANGLE_STRUCT.unpack_from(payload, offset)
    normal = vector3(values[0:3], "triangle normal")
    triangle = (
        vector3(values[3:6], "triangle vertex"),
        vector3(values[6:9], "triangle vertex"),
        vector3(values[9:12], "triangle vertex"),
    )
    first_edge = _subtract(triangle[1], triangle[0])
    second_edge = _subtract(triangle[2], triangle[0])
    geometric_normal = _cross(first_edge, second_edge)
    area_scale = _norm(geometric_normal)
    recorded_scale = _norm(normal)
    if area_scale == 0.0:
        raise RuntimeError("binary STL contains a degenerate triangle")
    if recorded_scale == 0.0:
        raise RuntimeError("binary STL contains a zero triangle normal")
    alignment = _dot(normal, geometric_normal) / (recorded_scale * area_scale)
    if alignment < _MIN_NORMAL_ALIGNMENT:
        raise RuntimeError("binary STL triangle normal does not match winding")
    return triangle


def _inspect_triangles(
    triangles: tuple[Triangle, ...],
) -> tuple[EdgeMap, set[Vector3], float]:
    edges: EdgeMap = defaultdict(list)
    vertices: set[Vector3] = set()
    signed_volume = 0.0
    for triangle_index, triangle in enumerate(triangles):
        vertices.update(triangle)
        signed_volume += _signed_tetrahedron_volume(triangle)
        for vertex_index in range(3):
            edge = (
                triangle[vertex_index],
                triangle[(vertex_index + 1) % 3],
            )
            edge_key = edge if edge[0] <= edge[1] else (edge[1], edge[0])
            edges[edge_key].append((edge, triangle_index))
    return edges, vertices, signed_volume


def _validate_topology(edges: EdgeMap, triangle_count: int) -> int:
    adjacency: list[set[int]] = [set() for _ in range(triangle_count)]
    for uses in edges.values():
        if len(uses) != 2:
            raise RuntimeError("binary STL is not watertight and two-manifold")
        first_edge, first_triangle = uses[0]
        second_edge, second_triangle = uses[1]
        if first_edge != (second_edge[1], second_edge[0]):
            raise RuntimeError("binary STL has inconsistent edge winding")
        adjacency[first_triangle].add(second_triangle)
        adjacency[second_triangle].add(first_triangle)
    component_count = _component_count(adjacency)
    if component_count != 1:
        raise RuntimeError("binary STL must contain one connected component")
    return component_count


def _component_count(adjacency: list[set[int]]) -> int:
    remaining = set(range(len(adjacency)))
    count = 0
    while remaining:
        count += 1
        pending = [remaining.pop()]
        while pending:
            current = pending.pop()
            neighbors = adjacency[current] & remaining
            remaining.difference_update(neighbors)
            pending.extend(neighbors)
    return count


def _mesh_reference(
    vertices: set[Vector3],
    volume_mm3: float,
) -> CadGeometryReference:
    if not vertices:
        raise RuntimeError("binary STL must contain vertices")
    minimum = tuple(
        min(vertex[axis] for vertex in vertices) / MM_PER_M for axis in range(3)
    )
    maximum = tuple(
        max(vertex[axis] for vertex in vertices) / MM_PER_M for axis in range(3)
    )
    return CadGeometryReference(
        volume_m3=volume_mm3 * M3_PER_MM3,
        bounds_min_m=vector3(minimum, "bounds_min_m"),
        bounds_max_m=vector3(maximum, "bounds_max_m"),
    )


def _volume_error_limit(
    reference: CadGeometryReference,
    tolerance_m: float,
) -> float:
    scaled = float(_VOLUME_TOLERANCE_FACTOR * tolerance_m / reference.minimum_span_m)
    return min(
        _VOLUME_ERROR_CEILING,
        max(_VOLUME_ERROR_FLOOR, scaled),
    )


def _require_mesh_match(
    signed_volume_mm3: float,
    volume_error: float,
    volume_limit: float,
    bounds_error: float,
    bounds_limit: float,
) -> None:
    if signed_volume_mm3 <= 0.0:
        raise RuntimeError("binary STL must have outward orientation")
    if volume_error > volume_limit:
        raise RuntimeError("binary STL volume exceeds tessellation tolerance")
    if bounds_error > bounds_limit:
        raise RuntimeError("binary STL bounds exceed tessellation tolerance")


def _signed_tetrahedron_volume(triangle: Triangle) -> float:
    return _dot(triangle[0], _cross(triangle[1], triangle[2])) / 6.0


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Vector3, right: Vector3) -> float:
    return float(left[0] * right[0] + left[1] * right[1] + left[2] * right[2])


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


__all__ = ["StlMeshValidation", "read_binary_stl", "validate_binary_stl"]
