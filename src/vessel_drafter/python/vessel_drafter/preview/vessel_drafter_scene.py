"""3D preview scene builders for the vessel drafter GUI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import pi
from typing import TypeAlias, cast

import numpy as np

from vessel_drafter.models.vessel_drafter import (
    DEFAULT_VESSEL_DRAFTER_LAYOUT,
    MM_PER_INCH,
    VesselDrafterLayout,
)
from vessel_drafter.preview.vessel_drafter_view_options import (
    Vessel3DViewOptions,
)
from vessel_drafter.projects.vessel_drafter_profiles import (
    ProfilePoint,
    build_bottom_head_curve,
    build_glass_boundary_half,
    build_shell_band_profiles,
    build_top_head_curve,
)

FloatArray: TypeAlias = np.ndarray
IntArray: TypeAlias = np.ndarray

AXIS_EPSILON_IN = 1e-6
DEFAULT_ANGULAR_SEGMENTS = 24
DEFAULT_HEAD_SAMPLE_COUNT = 12
DEFAULT_ELECTRODE_SEGMENTS = 12
DEFAULT_VIEW_OPTIONS = Vessel3DViewOptions()


@dataclass(frozen=True)
class VesselSceneMesh:
    label: str
    group_label: str
    display_name: str
    color_hex: str
    alpha: float
    vertices: FloatArray
    faces: IntArray

    @property
    def polygons(self) -> FloatArray:
        return self.vertices[self.faces]


@dataclass(frozen=True)
class Vessel3DScene:
    meshes: tuple[VesselSceneMesh, ...]
    bounds: tuple[float, float, float, float, float, float]


def build_vessel_3d_scene(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
    *,
    visible_labels: set[str] | None = None,
    view_options: Vessel3DViewOptions = DEFAULT_VIEW_OPTIONS,
) -> Vessel3DScene:
    return _build_vessel_3d_scene_cached(
        layout,
        _visible_label_key(visible_labels),
        view_options.split_enabled,
        view_options.normalized_split_angle_degrees,
    )


@lru_cache(maxsize=48)
def _build_vessel_3d_scene_cached(
    layout: VesselDrafterLayout,
    visible_label_key: tuple[str, ...] | None,
    split_enabled: bool,
    split_angle_degrees: float,
) -> Vessel3DScene:
    view_options = Vessel3DViewOptions(
        split_enabled=split_enabled,
        split_angle_degrees=split_angle_degrees,
    )
    visible_labels = None if visible_label_key is None else set(visible_label_key)
    if layout.side_ports or layout.lid_ports:
        meshes = _build_exact_meshes(layout, visible_labels, view_options)
    else:
        meshes = _build_fast_meshes(layout, visible_labels, view_options)
    return Vessel3DScene(meshes=meshes, bounds=_scene_bounds(meshes))


def _build_fast_meshes(
    layout: VesselDrafterLayout,
    visible_labels: set[str] | None,
    view_options: Vessel3DViewOptions,
) -> tuple[VesselSceneMesh, ...]:
    meshes = [
        _build_glass_mesh(layout, visible_labels, view_options),
        *_build_shell_meshes(layout, visible_labels, view_options),
        *_build_electrode_meshes(layout, visible_labels, view_options),
    ]
    return tuple(mesh for mesh in meshes if mesh is not None)


def _build_glass_mesh(
    layout: VesselDrafterLayout,
    visible_labels: set[str] | None,
    view_options: Vessel3DViewOptions,
) -> VesselSceneMesh | None:
    layer = layout.layers[0]
    if not _is_visible(layer.name, layer.name, visible_labels):
        return None
    return _make_mesh(
        label=layer.name,
        group_label=layer.name,
        display_name=layer.display_name,
        color_hex=layer.color_hex,
        alpha=_display_alpha(layer.preview_alpha, view_options),
        vertices_faces=_revolved_profile_mesh(
            build_glass_boundary_half(layout),
            view_options,
        ),
    )


def _build_shell_meshes(
    layout: VesselDrafterLayout,
    visible_labels: set[str] | None,
    view_options: Vessel3DViewOptions,
) -> tuple[VesselSceneMesh, ...]:
    meshes: list[VesselSceneMesh] = []
    for profile in build_shell_band_profiles(layout):
        band = profile.band
        if not _is_visible(band.label, band.label, visible_labels):
            continue
        inner_half = (
            _build_cavity_boundary_half(layout)
            if profile.inner_offset_in == 0.0
            else _build_shell_boundary_half(layout, profile.inner_offset_in)
        )
        outer_half = _build_shell_boundary_half(layout, profile.outer_offset_in)
        meshes.append(
            _make_mesh(
                label=band.label,
                group_label=band.label,
                display_name=layout.material_properties_by_name[
                    band.label
                ].display_name,
                color_hex=band.color_hex,
                alpha=_display_alpha(
                    layout.material_properties_by_name[band.label].preview_alpha,
                    view_options,
                ),
                vertices_faces=_revolved_profile_mesh(
                    outer_half + tuple(reversed(inner_half)),
                    view_options,
                ),
            )
        )
    return tuple(meshes)


def _build_electrode_meshes(
    layout: VesselDrafterLayout,
    visible_labels: set[str] | None,
    view_options: Vessel3DViewOptions,
) -> tuple[VesselSceneMesh, ...]:
    if not _is_visible("electrodes", "electrodes", visible_labels):
        return ()
    meshes: list[VesselSceneMesh] = []
    material = layout.material_properties_by_name["electrodes"]
    for placement in layout.electrode_placements:
        label = f"electrode_{placement.index}"
        if not _is_visible(label, "electrodes", visible_labels):
            continue
        start_point = np.array(
            [
                placement.inner_tip_radius_in * np.cos(placement.angle_radians),
                placement.inner_tip_radius_in * np.sin(placement.angle_radians),
                layout.electrode_centerline_height_in,
            ],
            dtype=np.float64,
        )
        end_point = np.array(
            [
                placement.outer_tip_radius_in * np.cos(placement.angle_radians),
                placement.outer_tip_radius_in * np.sin(placement.angle_radians),
                layout.electrode_centerline_height_in,
            ],
            dtype=np.float64,
        )
        vertices_faces = _maybe_apply_section_cut(
            _cylinder_mesh(
                start_point,
                end_point,
                layout.electrode_radius_in,
            ),
            view_options,
        )
        if vertices_faces[1].size == 0:
            continue
        meshes.append(
            _make_mesh(
                label=label,
                group_label="electrodes",
                display_name=material.display_name,
                color_hex=material.color_hex,
                alpha=_display_alpha(material.preview_alpha, view_options),
                vertices_faces=vertices_faces,
            )
        )
    return tuple(meshes)


def _build_exact_meshes(
    layout: VesselDrafterLayout,
    visible_labels: set[str] | None,
    view_options: Vessel3DViewOptions,
) -> tuple[VesselSceneMesh, ...]:
    from vessel_drafter.projects.vessel_drafter_layout import (  # lazy: needs build123d
        build_vessel_drafter_components,
    )

    meshes: list[VesselSceneMesh] = []
    for component in build_vessel_drafter_components(layout):
        if not _is_visible(component.label, component.group_label, visible_labels):
            continue
        vertices_faces = _maybe_apply_section_cut(
            _exact_component_mesh(component),
            view_options,
        )
        if vertices_faces[1].size == 0:
            continue
        meshes.append(
            _make_mesh(
                label=component.label,
                group_label=component.group_label,
                display_name=component.display_name,
                color_hex=component.color_hex,
                alpha=_display_alpha(component.preview_alpha, view_options),
                vertices_faces=vertices_faces,
            )
        )
    return tuple(meshes)


def _exact_component_mesh(component) -> tuple[FloatArray, IntArray]:
    vertices, triangle_indices = component.shape.tessellate(6.0)
    vertex_array = np.array(
        [_vector_to_inch_point(vertex) for vertex in vertices],
        dtype=np.float64,
    )
    face_array = np.array(triangle_indices, dtype=np.int32)
    return vertex_array, face_array


def _vector_to_inch_point(vector) -> tuple[float, float, float]:
    return (
        vector.X / MM_PER_INCH,
        vector.Y / MM_PER_INCH,
        vector.Z / MM_PER_INCH,
    )


def _build_cavity_boundary_half(
    layout: VesselDrafterLayout,
) -> tuple[ProfilePoint, ...]:
    return (
        ProfilePoint(0.0, 0.0),
        ProfilePoint(layout.inner_radius_in, 0.0),
        ProfilePoint(layout.inner_radius_in, layout.straight_shell_height_in),
        *build_top_head_curve(
            layout,
            0.0,
            sample_count=DEFAULT_HEAD_SAMPLE_COUNT,
        )[1:],
    )


def _build_shell_boundary_half(
    layout: VesselDrafterLayout,
    offset_in: float,
) -> tuple[ProfilePoint, ...]:
    radius_in = layout.inner_radius_in + offset_in
    return (
        *build_bottom_head_curve(
            layout,
            offset_in,
            sample_count=DEFAULT_HEAD_SAMPLE_COUNT,
        ),
        ProfilePoint(radius_in, layout.straight_shell_height_in),
        *build_top_head_curve(
            layout,
            offset_in,
            sample_count=DEFAULT_HEAD_SAMPLE_COUNT,
        )[1:],
    )


def _revolved_profile_mesh(
    half_profile: tuple[ProfilePoint, ...],
    view_options: Vessel3DViewOptions,
) -> tuple[FloatArray, IntArray]:
    angles = _preview_angles(view_options)
    angle_count = len(angles)
    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)
    vertex_groups: list[int | IntArray] = []
    vertex_parts: list[FloatArray] = []
    vertex_count = 0

    for point in half_profile:
        if abs(point.x_in) <= AXIS_EPSILON_IN:
            vertex_groups.append(vertex_count)
            vertex_parts.append(np.array([[0.0, 0.0, point.z_in]], dtype=np.float64))
            vertex_count += 1
            continue
        ring = np.column_stack(
            (
                point.x_in * cos_theta,
                point.x_in * sin_theta,
                np.full(angle_count, point.z_in, dtype=np.float64),
            )
        )
        vertex_groups.append(
            np.arange(
                vertex_count,
                vertex_count + angle_count,
                dtype=np.int32,
            )
        )
        vertex_parts.append(ring)
        vertex_count += angle_count

    vertices = np.vstack(vertex_parts)
    face_parts = []
    for start_group, end_group in _profile_segments(vertex_groups):
        segment_faces = _segment_faces(
            start_group,
            end_group,
            wrap_around=not view_options.split_enabled,
        )
        if segment_faces.size > 0:
            face_parts.append(segment_faces)
    if view_options.split_enabled:
        section_vertices, section_faces = _section_cap_mesh(
            half_profile,
            view_options,
        )
        if section_faces.size > 0:
            section_faces = section_faces + len(vertices)
            vertices = np.vstack((vertices, section_vertices))
            face_parts.append(section_faces)
    faces = np.vstack(face_parts)
    return vertices, faces


def _profile_segments(
    vertex_groups: list[int | IntArray],
) -> tuple[tuple[int | IntArray, int | IntArray], ...]:
    return tuple(
        (vertex_groups[index], vertex_groups[(index + 1) % len(vertex_groups)])
        for index in range(len(vertex_groups))
    )


def _segment_faces(
    start_group: int | IntArray,
    end_group: int | IntArray,
    *,
    wrap_around: bool,
) -> IntArray:
    if isinstance(start_group, int) and isinstance(end_group, int):
        return np.empty((0, 3), dtype=np.int32)

    if isinstance(start_group, int):
        ring_indices = _ring_index_pairs(cast(IntArray, end_group), wrap_around)
        return np.column_stack(
            (
                np.full(len(ring_indices), start_group, dtype=np.int32),
                ring_indices[:, 0],
                ring_indices[:, 1],
            )
        )

    if isinstance(end_group, int):
        ring_indices = _ring_index_pairs(cast(IntArray, start_group), wrap_around)
        return np.column_stack(
            (
                np.full(len(ring_indices), end_group, dtype=np.int32),
                ring_indices[:, 1],
                ring_indices[:, 0],
            )
        )

    start_pairs = _ring_index_pairs(start_group, wrap_around)
    end_pairs = _ring_index_pairs(end_group, wrap_around)
    first_faces = np.column_stack(
        (
            start_pairs[:, 0],
            end_pairs[:, 0],
            end_pairs[:, 1],
        )
    )
    second_faces = np.column_stack(
        (
            start_pairs[:, 0],
            end_pairs[:, 1],
            start_pairs[:, 1],
        )
    )
    return np.vstack((first_faces, second_faces))


def _cylinder_mesh(
    start_point: FloatArray,
    end_point: FloatArray,
    radius_in: float,
) -> tuple[FloatArray, IntArray]:
    axis_vector = end_point - start_point
    axis_length = np.linalg.norm(axis_vector)
    axis_unit = axis_vector / axis_length
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(axis_unit, reference)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    radial_unit = np.cross(axis_unit, reference)
    radial_unit /= np.linalg.norm(radial_unit)
    binormal_unit = np.cross(axis_unit, radial_unit)

    angles = np.linspace(
        0.0,
        2.0 * pi,
        DEFAULT_ELECTRODE_SEGMENTS,
        endpoint=False,
        dtype=np.float64,
    )
    ring_offsets = np.column_stack(
        (
            np.cos(angles),
            np.sin(angles),
        )
    )
    radial_offsets = (
        ring_offsets[:, :1] * radial_unit[np.newaxis, :]
        + ring_offsets[:, 1:] * binormal_unit[np.newaxis, :]
    ) * radius_in
    start_ring = start_point[np.newaxis, :] + radial_offsets
    end_ring = end_point[np.newaxis, :] + radial_offsets
    vertices = np.vstack((start_ring, end_ring, start_point, end_point))

    lower_indices = np.arange(DEFAULT_ELECTRODE_SEGMENTS, dtype=np.int32)
    upper_indices = lower_indices + DEFAULT_ELECTRODE_SEGMENTS
    start_center_index = np.int32(DEFAULT_ELECTRODE_SEGMENTS * 2)
    end_center_index = np.int32(start_center_index + 1)
    side_faces = np.vstack(
        (
            np.column_stack((lower_indices, upper_indices, np.roll(upper_indices, -1))),
            np.column_stack(
                (
                    lower_indices,
                    np.roll(upper_indices, -1),
                    np.roll(lower_indices, -1),
                )
            ),
        )
    )
    start_cap = np.column_stack(
        (
            np.full(DEFAULT_ELECTRODE_SEGMENTS, start_center_index, dtype=np.int32),
            np.roll(lower_indices, -1),
            lower_indices,
        )
    )
    end_cap = np.column_stack(
        (
            np.full(DEFAULT_ELECTRODE_SEGMENTS, end_center_index, dtype=np.int32),
            upper_indices,
            np.roll(upper_indices, -1),
        )
    )
    return vertices, np.vstack((side_faces, start_cap, end_cap))


def _preview_angles(view_options: Vessel3DViewOptions) -> FloatArray:
    if not view_options.split_enabled:
        return np.linspace(
            0.0,
            2.0 * pi,
            DEFAULT_ANGULAR_SEGMENTS,
            endpoint=False,
            dtype=np.float64,
        )
    return np.linspace(
        view_options.normalized_split_angle_radians,
        view_options.normalized_split_angle_radians + pi,
        (DEFAULT_ANGULAR_SEGMENTS // 2) + 1,
        endpoint=True,
        dtype=np.float64,
    )


def _section_cap_mesh(
    half_profile: tuple[ProfilePoint, ...],
    view_options: Vessel3DViewOptions,
) -> tuple[FloatArray, IntArray]:
    triangles = _triangulate_profile_loop(half_profile)
    if not triangles:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32)

    start_vertices = _profile_vertices_on_plane(
        half_profile,
        view_options.normalized_split_angle_radians,
    )
    end_vertices = _profile_vertices_on_plane(
        half_profile,
        view_options.normalized_split_angle_radians + pi,
    )
    start_faces = np.array(triangles, dtype=np.int32)
    end_faces = np.array(
        [(third, second, first) for first, second, third in triangles],
        dtype=np.int32,
    ) + len(start_vertices)
    return (
        np.vstack((start_vertices, end_vertices)),
        np.vstack((start_faces, end_faces)),
    )


def _profile_vertices_on_plane(
    half_profile: tuple[ProfilePoint, ...],
    angle_radians: float,
) -> FloatArray:
    direction = np.array(
        [np.cos(angle_radians), np.sin(angle_radians), 0.0],
        dtype=np.float64,
    )
    return np.array(
        [
            (
                point.x_in * direction[0],
                point.x_in * direction[1],
                point.z_in,
            )
            for point in half_profile
        ],
        dtype=np.float64,
    )


def _triangulate_profile_loop(
    half_profile: tuple[ProfilePoint, ...],
) -> tuple[tuple[int, int, int], ...]:
    points = np.array(
        [(point.x_in, point.z_in) for point in half_profile], dtype=np.float64
    )
    indices = list(range(len(points)))
    if len(indices) < 3:
        return ()

    triangles: list[tuple[int, int, int]] = []
    orientation = 1.0 if _signed_area(points) >= 0.0 else -1.0
    guard_limit = len(indices) * len(indices)
    guard_count = 0

    while len(indices) > 3 and guard_count < guard_limit:
        guard_count += 1
        ear_index = _find_ear(points, indices, orientation)
        if ear_index is None:
            break
        triangles.append(ear_index)
        indices.remove(ear_index[1])

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))

    if triangles:
        return tuple(triangles)
    return _fan_triangulation(indices)


def _find_ear(
    points: FloatArray,
    indices: list[int],
    orientation: float,
) -> tuple[int, int, int] | None:
    for position, current in enumerate(indices):
        previous = indices[position - 1]
        following = indices[(position + 1) % len(indices)]
        if not _is_convex(
            points[previous], points[current], points[following], orientation
        ):
            continue
        triangle = np.array(
            [points[previous], points[current], points[following]],
            dtype=np.float64,
        )
        if any(
            _point_in_triangle(points[candidate], triangle)
            for candidate in indices
            if candidate not in (previous, current, following)
        ):
            continue
        return previous, current, following
    return None


def _is_convex(
    previous: FloatArray,
    current: FloatArray,
    following: FloatArray,
    orientation: float,
) -> bool:
    cross_value = _cross_z(previous, current, following)
    return (cross_value * orientation) > 1e-9


def _point_in_triangle(point: FloatArray, triangle: FloatArray) -> bool:
    first_sign = _edge_sign(point, triangle[0], triangle[1])
    second_sign = _edge_sign(point, triangle[1], triangle[2])
    third_sign = _edge_sign(point, triangle[2], triangle[0])
    has_negative = (first_sign < -1e-9) or (second_sign < -1e-9) or (third_sign < -1e-9)
    has_positive = (first_sign > 1e-9) or (second_sign > 1e-9) or (third_sign > 1e-9)
    return not (has_negative and has_positive)


def _edge_sign(point: FloatArray, first: FloatArray, second: FloatArray) -> float:
    return ((point[0] - second[0]) * (first[1] - second[1])) - (
        (first[0] - second[0]) * (point[1] - second[1])
    )


def _cross_z(previous: FloatArray, current: FloatArray, following: FloatArray) -> float:
    first = current - previous
    second = following - current
    return (first[0] * second[1]) - (first[1] * second[0])


def _signed_area(points: FloatArray) -> float:
    shifted = np.roll(points, -1, axis=0)
    cross = (points[:, 0] * shifted[:, 1]) - (shifted[:, 0] * points[:, 1])
    return 0.5 * float(np.sum(cross))


def _fan_triangulation(indices: list[int]) -> tuple[tuple[int, int, int], ...]:
    if len(indices) < 3:
        return ()
    return tuple(
        (indices[0], indices[position], indices[position + 1])
        for position in range(1, len(indices) - 1)
    )


def _ring_index_pairs(indices: IntArray, wrap_around: bool) -> IntArray:
    if wrap_around:
        return np.column_stack((indices, np.roll(indices, -1)))
    return np.column_stack((indices[:-1], indices[1:]))


def _maybe_apply_section_cut(
    vertices_faces: tuple[FloatArray, IntArray],
    view_options: Vessel3DViewOptions,
) -> tuple[FloatArray, IntArray]:
    if not view_options.split_enabled:
        return vertices_faces
    vertices, faces = vertices_faces
    if faces.size == 0:
        return vertices, faces
    normal = np.array(
        [
            -np.sin(view_options.normalized_split_angle_radians),
            np.cos(view_options.normalized_split_angle_radians),
            0.0,
        ],
        dtype=np.float64,
    )
    centroids = vertices[faces].mean(axis=1)
    mask = (centroids @ normal) >= -1e-9
    return vertices, faces[mask]


def _visible_label_key(visible_labels: set[str] | None) -> tuple[str, ...] | None:
    if visible_labels is None:
        return None
    return tuple(sorted(visible_labels))


def _is_visible(
    label: str,
    group_label: str,
    visible_labels: set[str] | None,
) -> bool:
    if visible_labels is None:
        return True
    return label in visible_labels or group_label in visible_labels


def _scene_bounds(
    meshes: tuple[VesselSceneMesh, ...],
) -> tuple[float, float, float, float, float, float]:
    vertex_parts = [
        mesh.polygons.reshape(-1, 3) for mesh in meshes if len(mesh.faces) > 0
    ]
    if not vertex_parts:
        return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    all_vertices = np.vstack(vertex_parts)
    return (
        float(np.min(all_vertices[:, 0])),
        float(np.max(all_vertices[:, 0])),
        float(np.min(all_vertices[:, 1])),
        float(np.max(all_vertices[:, 1])),
        float(np.min(all_vertices[:, 2])),
        float(np.max(all_vertices[:, 2])),
    )


def _display_alpha(base_alpha: float, view_options: Vessel3DViewOptions) -> float:
    if view_options.split_enabled:
        return 1.0
    return base_alpha


def _make_mesh(
    *,
    label: str,
    group_label: str,
    display_name: str,
    color_hex: str,
    alpha: float,
    vertices_faces: tuple[FloatArray, IntArray],
) -> VesselSceneMesh:
    vertices, faces = vertices_faces
    return VesselSceneMesh(
        label=label,
        group_label=group_label,
        display_name=display_name,
        color_hex=color_hex,
        alpha=alpha,
        vertices=vertices,
        faces=faces,
    )
