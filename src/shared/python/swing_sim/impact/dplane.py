"""Reference-frame-explicit three-dimensional D-plane geometry.

This module deliberately stops at geometry.  It does not infer launch direction,
ball spin, or aerodynamic curvature without a declared collision/flight model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import numpy as np

from shared.python.contracts import require

Vector3: TypeAlias = tuple[float, float, float]
_EPSILON = 1e-12


class DPlaneStatus(StrEnum):
    """Typed availability state for the plane spanned by travel and face."""

    DEFINED = "defined"
    ZERO_TRAVEL = "zero_travel"
    PARALLEL = "parallel"
    ANTIPARALLEL = "antiparallel"


@dataclass(frozen=True)
class DPlaneAnalysis:
    """Complete geometric D-plane diagnostics in a declared target frame."""

    status: DPlaneStatus
    frame_id: str
    travel_direction_unit: Vector3 | None
    face_normal_unit: Vector3
    target_unit: Vector3
    up_unit: Vector3
    right_unit: Vector3
    dplane_normal_unit: Vector3 | None
    ground_intersection_unit: Vector3 | None
    spin_loft_3d_deg: float | None
    planar_spin_loft_deg: float | None
    signed_planar_gap_deg: float | None
    spin_loft_residual_deg: float | None
    club_path_deg: float | None
    attack_angle_deg: float | None
    face_angle_deg: float | None
    dynamic_loft_deg: float
    face_to_path_deg: float | None
    dplane_normal_azimuth_deg: float | None
    dplane_tilt_deg: float | None
    dplane_inclination_deg: float | None
    ground_intersection_azimuth_deg: float | None


def _as_vector(value: object, name: str) -> np.ndarray:
    vector: np.ndarray = np.asarray(value, dtype=float)
    require(vector.shape == (3,), f"{name} must contain exactly three components")
    require(bool(np.all(np.isfinite(vector))), f"{name} must be finite")
    return vector


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    require(magnitude > _EPSILON, f"{name} must be nonzero")
    normalized: np.ndarray = vector / magnitude
    return normalized


def _tuple(vector: np.ndarray) -> Vector3:
    return float(vector[0]), float(vector[1]), float(vector[2])


def _horizontal(vector: np.ndarray, up: np.ndarray) -> np.ndarray:
    horizontal: np.ndarray = vector - float(np.dot(vector, up)) * up
    return horizontal


def _heading_deg(
    vector: np.ndarray, target: np.ndarray, right: np.ndarray, up: np.ndarray
) -> float | None:
    horizontal = _horizontal(vector, up)
    if float(np.linalg.norm(horizontal)) <= _EPSILON:
        return None
    return math.degrees(
        math.atan2(float(np.dot(horizontal, right)), float(np.dot(horizontal, target)))
    )


def _elevation_deg(vector: np.ndarray, up: np.ndarray) -> float:
    vertical = float(np.dot(vector, up))
    horizontal_magnitude = float(np.linalg.norm(_horizontal(vector, up)))
    return math.degrees(math.atan2(vertical, horizontal_magnitude))


def _wrapped_delta_deg(first: float, second: float) -> float:
    return (first - second + 180.0) % 360.0 - 180.0


def analyze_dplane(
    travel_vector: object,
    face_normal: object,
    *,
    target: object = (1.0, 0.0, 0.0),
    up: object = (0.0, 1.0, 0.0),
    frame_id: str = "app_frame:x_target,y_up,z_right",
) -> DPlaneAnalysis:
    """Analyze the exact 3-D D-plane from travel and face-orientation vectors.

    ``travel_vector`` may carry speed; only its direction enters the geometry.
    The target and up axes must be orthogonal.  Their cross product defines the
    positive-right axis and therefore every signed horizontal result.
    """
    travel = _as_vector(travel_vector, "travel_vector")
    face = _unit(_as_vector(face_normal, "face_normal"), "face_normal")
    target_unit = _unit(_as_vector(target, "target"), "target")
    up_unit = _unit(_as_vector(up, "up"), "up")
    require(
        abs(float(np.dot(target_unit, up_unit))) <= 1e-10,
        "target and up axes must be orthogonal",
    )
    right_unit = _unit(np.cross(target_unit, up_unit), "target x up right axis")

    face_angle = _heading_deg(face, target_unit, right_unit, up_unit)
    dynamic_loft = _elevation_deg(face, up_unit)
    travel_magnitude = float(np.linalg.norm(travel))
    if travel_magnitude <= _EPSILON:
        return DPlaneAnalysis(
            DPlaneStatus.ZERO_TRAVEL,
            frame_id,
            None,
            _tuple(face),
            _tuple(target_unit),
            _tuple(up_unit),
            _tuple(right_unit),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            face_angle,
            dynamic_loft,
            None,
            None,
            None,
            None,
            None,
        )

    travel_unit = travel / travel_magnitude
    club_path = _heading_deg(travel_unit, target_unit, right_unit, up_unit)
    attack_angle = _elevation_deg(travel_unit, up_unit)
    face_to_path = (
        None
        if face_angle is None or club_path is None
        else _wrapped_delta_deg(face_angle, club_path)
    )
    signed_planar_gap = dynamic_loft - attack_angle
    planar_spin_loft = abs(signed_planar_gap)
    cosine = float(np.clip(np.dot(travel_unit, face), -1.0, 1.0))
    spin_loft = math.degrees(math.acos(cosine))
    residual = spin_loft - planar_spin_loft

    normal_raw = np.cross(travel_unit, face)
    normal_magnitude = float(np.linalg.norm(normal_raw))
    if normal_magnitude <= _EPSILON:
        status = DPlaneStatus.PARALLEL if cosine >= 0.0 else DPlaneStatus.ANTIPARALLEL
        return DPlaneAnalysis(
            status,
            frame_id,
            _tuple(travel_unit),
            _tuple(face),
            _tuple(target_unit),
            _tuple(up_unit),
            _tuple(right_unit),
            None,
            None,
            spin_loft,
            planar_spin_loft,
            signed_planar_gap,
            residual,
            club_path,
            attack_angle,
            face_angle,
            dynamic_loft,
            face_to_path,
            None,
            None,
            None,
            None,
        )

    normal = normal_raw / normal_magnitude
    normal_horizontal = _horizontal(normal, up_unit)
    normal_horizontal_magnitude = float(np.linalg.norm(normal_horizontal))
    normal_azimuth = _heading_deg(normal, target_unit, right_unit, up_unit)
    tilt = math.degrees(
        math.atan2(-float(np.dot(normal, up_unit)), normal_horizontal_magnitude)
    )
    inclination = math.degrees(
        math.acos(float(np.clip(abs(np.dot(normal, up_unit)), 0.0, 1.0)))
    )
    ground_raw = np.cross(up_unit, normal)
    if float(np.linalg.norm(ground_raw)) <= _EPSILON:
        ground_intersection = None
        ground_azimuth = None
    else:
        ground = ground_raw / float(np.linalg.norm(ground_raw))
        if float(np.dot(ground, target_unit)) < 0.0:
            ground = -ground
        ground_intersection = _tuple(ground)
        ground_azimuth = _heading_deg(ground, target_unit, right_unit, up_unit)

    return DPlaneAnalysis(
        DPlaneStatus.DEFINED,
        frame_id,
        _tuple(travel_unit),
        _tuple(face),
        _tuple(target_unit),
        _tuple(up_unit),
        _tuple(right_unit),
        _tuple(normal),
        ground_intersection,
        spin_loft,
        planar_spin_loft,
        signed_planar_gap,
        residual,
        club_path,
        attack_angle,
        face_angle,
        dynamic_loft,
        face_to_path,
        normal_azimuth,
        tilt,
        inclination,
        ground_azimuth,
    )


def spin_loft_sector_directions(
    analysis: DPlaneAnalysis, segments: int = 24
) -> tuple[Vector3, ...]:
    """Return a shortest-arc unit fan from travel toward the face normal."""
    require(segments >= 2, "segments must be at least two", segments)
    if (
        analysis.status is not DPlaneStatus.DEFINED
        or analysis.travel_direction_unit is None
        or analysis.spin_loft_3d_deg is None
    ):
        return ()
    travel = np.asarray(analysis.travel_direction_unit)
    face = np.asarray(analysis.face_normal_unit)
    angle = math.radians(analysis.spin_loft_3d_deg)
    sine = math.sin(angle)
    require(abs(sine) > _EPSILON, "defined D-plane must have a nonzero sector")
    directions = []
    for index in range(segments + 1):
        fraction = index / segments
        direction = (
            math.sin((1.0 - fraction) * angle) / sine * travel
            + math.sin(fraction * angle) / sine * face
        )
        directions.append(_tuple(_unit(direction, "sector direction")))
    return tuple(directions)
