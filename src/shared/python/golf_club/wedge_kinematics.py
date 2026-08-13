"""Frame-explicit rigid-body metrics for a wedge at a declared contact point."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._validation import Vector3, require_identifier, require_vector3

_UNIT_TOLERANCE = 1e-9
_ORTHOGONAL_TOLERANCE = 1e-9
_SPEED_TOLERANCE_MPS = 1e-12
_ANGULAR_SPEED_TOLERANCE_RAD_S = 1e-12
SASHO_FACE_CENTER_ROTATION_METHOD_ID = (
    "sasho_nearest_shaft_face_center_rotation_only_aoa_v1"
)


def _as_vector(value: Vector3) -> np.ndarray:
    vector: np.ndarray = np.asarray(value, dtype=float)
    return vector


def _tuple3(value: np.ndarray) -> Vector3:
    if value.shape != (3,):
        raise AssertionError("internal vector must have shape (3,)")
    return (float(value[0]), float(value[1]), float(value[2]))


def _require_unit_vector(value: object, name: str) -> Vector3:
    vector = require_vector3(value, name)
    if not math.isclose(
        float(np.linalg.norm(vector)), 1.0, abs_tol=_UNIT_TOLERANCE, rel_tol=0.0
    ):
        raise ValueError(f"{name} must be unit length")
    return vector


def _require_orthogonal(first: Vector3, second: Vector3, message: str) -> None:
    if not math.isclose(
        float(np.dot(first, second)),
        0.0,
        abs_tol=_ORTHOGONAL_TOLERANCE,
        rel_tol=0.0,
    ):
        raise ValueError(message)


@dataclass(frozen=True)
class WedgeKinematicState:
    """One instantaneous wedge pose derivative in a declared inertial frame.

    ``reference_velocity_mps`` and ``angular_velocity_rad_s`` form a rigid-body
    twist at ``reference_position_m``. ``shaft_axis_point_m`` is any physical
    point on the shaft centerline; changing the twist reference must not change
    it. Unit directions and their rates are expressed in ``frame_id``.
    """

    frame_id: str
    reference_position_m: Vector3
    reference_velocity_mps: Vector3
    angular_velocity_rad_s: Vector3
    shaft_axis_point_m: Vector3
    shaft_axis_unit: Vector3
    face_center_point_m: Vector3
    contact_point_m: Vector3
    face_normal_unit: Vector3
    leading_edge_tangent_unit: Vector3
    ground_up_unit: Vector3
    arc_tangent_unit: Vector3
    arc_tangent_rate_per_s: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", require_identifier(self.frame_id, "frame_id")
        )
        self._normalize_positions_and_twist()
        self._normalize_directions()
        self._validate_direction_geometry()

    def _normalize_positions_and_twist(self) -> None:
        for name in (
            "reference_position_m",
            "reference_velocity_mps",
            "angular_velocity_rad_s",
            "shaft_axis_point_m",
            "face_center_point_m",
            "contact_point_m",
            "arc_tangent_rate_per_s",
        ):
            object.__setattr__(self, name, require_vector3(getattr(self, name), name))

    def _normalize_directions(self) -> None:
        for name in (
            "shaft_axis_unit",
            "face_normal_unit",
            "leading_edge_tangent_unit",
            "ground_up_unit",
            "arc_tangent_unit",
        ):
            object.__setattr__(
                self, name, _require_unit_vector(getattr(self, name), name)
            )

    def _validate_direction_geometry(self) -> None:
        _require_orthogonal(
            self.face_normal_unit,
            self.leading_edge_tangent_unit,
            "face normal and leading edge tangent must be orthogonal",
        )
        _require_orthogonal(
            self.arc_tangent_unit,
            self.arc_tangent_rate_per_s,
            "arc tangent rate must be orthogonal to arc tangent",
        )


@dataclass(frozen=True)
class InstantaneousScrewAxis:
    """Instantaneous screw-axis geometry for a nonzero rigid-body twist."""

    point_nearest_origin_m: Vector3
    direction_unit: Vector3
    pitch_m_per_rad: float
    contact_distance_m: float


@dataclass(frozen=True)
class SashoFaceCenterRotationAoa:
    """Descriptive face-center rotation metric about the nearest shaft point."""

    method_id: str
    nearest_shaft_point_m: Vector3
    lever_arm_m: Vector3
    velocity_mps: Vector3
    aoa_deg: float | None


@dataclass(frozen=True)
class WedgeKinematicAnalysis:
    """Auditable impact-point decomposition and orientation-rate metrics."""

    frame_id: str
    contact_velocity_mps: Vector3
    shaft_axis_velocity_mps: Vector3
    shaft_rotation_velocity_mps: Vector3
    non_shaft_rotation_velocity_mps: Vector3
    without_shaft_velocity_mps: Vector3
    total_aoa_deg: float | None
    without_shaft_aoa_deg: float | None
    shaft_axis_translation_aoa_deg: float | None
    shaft_counterfactual_aoa_delta_deg: float | None
    shaft_shapley_aoa_deg: float | None
    non_shaft_shapley_aoa_deg: float | None
    shaft_vertical_velocity_share: float | None
    shaft_rotation_rate_rad_s: float
    face_normal_rate_per_s: Vector3
    face_normal_3d_rate_rad_s: float
    leading_edge_rate_per_s: Vector3
    leading_edge_3d_rate_rad_s: float
    leading_edge_ground_heading_rate_rad_s: float | None
    arc_ground_heading_rate_rad_s: float | None
    leading_edge_relative_arc_heading_rate_rad_s: float | None
    sasho_face_center_rotation: SashoFaceCenterRotationAoa
    screw_axis: InstantaneousScrewAxis | None


def angle_of_attack_deg(velocity_mps: object, ground_up_unit: object) -> float | None:
    """Return signed AoA from velocity, or ``None`` at zero horizontal speed.

    Positive angles travel upward; descending contact motion is negative. The
    caller must supply an inertial ground-up unit direction in the same frame.
    """
    velocity = _as_vector(require_vector3(velocity_mps, "velocity_mps"))
    ground_up = _as_vector(_require_unit_vector(ground_up_unit, "ground_up_unit"))
    vertical_speed = float(np.dot(velocity, ground_up))
    horizontal = velocity - vertical_speed * ground_up
    horizontal_speed = float(np.linalg.norm(horizontal))
    if horizontal_speed <= _SPEED_TOLERANCE_MPS:
        return None
    return math.degrees(math.atan2(vertical_speed, horizontal_speed))


def sasho_face_center_rotation_aoa(
    *,
    angular_velocity_rad_s: object,
    shaft_axis_point_m: object,
    shaft_axis_unit: object,
    face_center_point_m: object,
    ground_up_unit: object,
) -> SashoFaceCenterRotationAoa:
    """Return Sasho's rotation-only face-center AoA about the shaft line.

    This descriptive metric uses the complete club angular velocity and the
    lever from the nearest point on the physical shaft line to face center. It
    is not the shaft-axis-only counterfactual or a causal AoA contribution.
    """
    angular_velocity = _as_vector(
        require_vector3(angular_velocity_rad_s, "angular_velocity_rad_s")
    )
    shaft_point = _as_vector(require_vector3(shaft_axis_point_m, "shaft_axis_point_m"))
    shaft_axis = _as_vector(_require_unit_vector(shaft_axis_unit, "shaft_axis_unit"))
    face_center = _as_vector(
        require_vector3(face_center_point_m, "face_center_point_m")
    )
    offset = face_center - shaft_point
    nearest = shaft_point + float(np.dot(offset, shaft_axis)) * shaft_axis
    lever = face_center - nearest
    velocity = np.cross(angular_velocity, lever)
    return SashoFaceCenterRotationAoa(
        method_id=SASHO_FACE_CENTER_ROTATION_METHOD_ID,
        nearest_shaft_point_m=_tuple3(nearest),
        lever_arm_m=_tuple3(lever),
        velocity_mps=_tuple3(velocity),
        aoa_deg=angle_of_attack_deg(velocity, ground_up_unit),
    )


def _heading_rate_rad_s(
    direction: np.ndarray, rate: np.ndarray, ground_up: np.ndarray
) -> float | None:
    projected = direction - float(np.dot(direction, ground_up)) * ground_up
    projected_rate = rate - float(np.dot(rate, ground_up)) * ground_up
    magnitude_squared = float(np.dot(projected, projected))
    if magnitude_squared <= _UNIT_TOLERANCE**2:
        return None
    return (
        float(np.dot(ground_up, np.cross(projected, projected_rate)))
        / magnitude_squared
    )


def _shapley_aoa(
    base: np.ndarray,
    shaft: np.ndarray,
    other: np.ndarray,
    ground_up: Vector3,
) -> tuple[float | None, float | None]:
    base_aoa = angle_of_attack_deg(base, ground_up)
    shaft_aoa = angle_of_attack_deg(base + shaft, ground_up)
    other_aoa = angle_of_attack_deg(base + other, ground_up)
    total_aoa = angle_of_attack_deg(base + shaft + other, ground_up)
    if None in (base_aoa, shaft_aoa, other_aoa, total_aoa):
        return None, None
    assert base_aoa is not None
    assert shaft_aoa is not None
    assert other_aoa is not None
    assert total_aoa is not None
    shaft_share = 0.5 * ((shaft_aoa - base_aoa) + (total_aoa - other_aoa))
    other_share = 0.5 * ((other_aoa - base_aoa) + (total_aoa - shaft_aoa))
    return shaft_share, other_share


def _instantaneous_screw_axis(
    state: WedgeKinematicState,
) -> InstantaneousScrewAxis | None:
    angular_velocity = _as_vector(state.angular_velocity_rad_s)
    angular_speed_squared = float(np.dot(angular_velocity, angular_velocity))
    if angular_speed_squared <= _ANGULAR_SPEED_TOLERANCE_RAD_S**2:
        return None
    reference_position = _as_vector(state.reference_position_m)
    reference_velocity = _as_vector(state.reference_velocity_mps)
    velocity_at_origin = reference_velocity - np.cross(
        angular_velocity, reference_position
    )
    axis_point = np.cross(angular_velocity, velocity_at_origin) / angular_speed_squared
    direction = angular_velocity / math.sqrt(angular_speed_squared)
    pitch = float(np.dot(angular_velocity, velocity_at_origin)) / angular_speed_squared
    contact_offset = _as_vector(state.contact_point_m) - axis_point
    radial_offset = (
        contact_offset - float(np.dot(contact_offset, direction)) * direction
    )
    return InstantaneousScrewAxis(
        point_nearest_origin_m=_tuple3(axis_point),
        direction_unit=_tuple3(direction),
        pitch_m_per_rad=pitch,
        contact_distance_m=float(np.linalg.norm(radial_offset)),
    )


def _counterfactual_delta(
    total_aoa: float | None, without_shaft_aoa: float | None
) -> float | None:
    if total_aoa is None or without_shaft_aoa is None:
        return None
    return total_aoa - without_shaft_aoa


def _vertical_share(
    shaft_velocity: np.ndarray, total_velocity: np.ndarray, ground_up: np.ndarray
) -> float | None:
    total_vertical = float(np.dot(total_velocity, ground_up))
    if abs(total_vertical) <= _SPEED_TOLERANCE_MPS:
        return None
    return float(np.dot(shaft_velocity, ground_up)) / total_vertical


def analyze_wedge_kinematics(state: WedgeKinematicState) -> WedgeKinematicAnalysis:
    """Decompose contact velocity and orientation rates about the shaft datum.

    The exact velocity identity is ``v_contact = v_axis + v_shaft + v_other``.
    AoA attribution includes both a direct remove-shaft counterfactual and an
    order-independent two-factor Shapley decomposition about ``v_axis``.
    """
    if not isinstance(state, WedgeKinematicState):
        raise TypeError("state must be a WedgeKinematicState")
    reference = _as_vector(state.reference_position_m)
    shaft_point = _as_vector(state.shaft_axis_point_m)
    contact = _as_vector(state.contact_point_m)
    angular_velocity = _as_vector(state.angular_velocity_rad_s)
    shaft_axis = _as_vector(state.shaft_axis_unit)
    reference_velocity = _as_vector(state.reference_velocity_mps)
    axis_velocity = reference_velocity + np.cross(
        angular_velocity, shaft_point - reference
    )
    shaft_rate = float(np.dot(angular_velocity, shaft_axis))
    shaft_omega = shaft_rate * shaft_axis
    other_omega = angular_velocity - shaft_omega
    shaft_lever = contact - shaft_point
    shaft_velocity = np.cross(shaft_omega, shaft_lever)
    other_velocity = np.cross(other_omega, shaft_lever)
    total_velocity = axis_velocity + shaft_velocity + other_velocity
    without_shaft = total_velocity - shaft_velocity
    total_aoa = angle_of_attack_deg(total_velocity, state.ground_up_unit)
    without_shaft_aoa = angle_of_attack_deg(without_shaft, state.ground_up_unit)
    translation_aoa = angle_of_attack_deg(axis_velocity, state.ground_up_unit)
    shaft_shapley, other_shapley = _shapley_aoa(
        axis_velocity, shaft_velocity, other_velocity, state.ground_up_unit
    )
    face_rate = np.cross(angular_velocity, _as_vector(state.face_normal_unit))
    edge_rate = np.cross(angular_velocity, _as_vector(state.leading_edge_tangent_unit))
    ground_up = _as_vector(state.ground_up_unit)
    edge_heading = _heading_rate_rad_s(
        _as_vector(state.leading_edge_tangent_unit), edge_rate, ground_up
    )
    arc_heading = _heading_rate_rad_s(
        _as_vector(state.arc_tangent_unit),
        _as_vector(state.arc_tangent_rate_per_s),
        ground_up,
    )
    relative_heading = (
        None
        if edge_heading is None or arc_heading is None
        else edge_heading - arc_heading
    )
    sasho_rotation = sasho_face_center_rotation_aoa(
        angular_velocity_rad_s=state.angular_velocity_rad_s,
        shaft_axis_point_m=state.shaft_axis_point_m,
        shaft_axis_unit=state.shaft_axis_unit,
        face_center_point_m=state.face_center_point_m,
        ground_up_unit=state.ground_up_unit,
    )
    return WedgeKinematicAnalysis(
        frame_id=state.frame_id,
        contact_velocity_mps=_tuple3(total_velocity),
        shaft_axis_velocity_mps=_tuple3(axis_velocity),
        shaft_rotation_velocity_mps=_tuple3(shaft_velocity),
        non_shaft_rotation_velocity_mps=_tuple3(other_velocity),
        without_shaft_velocity_mps=_tuple3(without_shaft),
        total_aoa_deg=total_aoa,
        without_shaft_aoa_deg=without_shaft_aoa,
        shaft_axis_translation_aoa_deg=translation_aoa,
        shaft_counterfactual_aoa_delta_deg=_counterfactual_delta(
            total_aoa, without_shaft_aoa
        ),
        shaft_shapley_aoa_deg=shaft_shapley,
        non_shaft_shapley_aoa_deg=other_shapley,
        shaft_vertical_velocity_share=_vertical_share(
            shaft_velocity, total_velocity, ground_up
        ),
        shaft_rotation_rate_rad_s=shaft_rate,
        face_normal_rate_per_s=_tuple3(face_rate),
        face_normal_3d_rate_rad_s=float(np.linalg.norm(face_rate)),
        leading_edge_rate_per_s=_tuple3(edge_rate),
        leading_edge_3d_rate_rad_s=float(np.linalg.norm(edge_rate)),
        leading_edge_ground_heading_rate_rad_s=edge_heading,
        arc_ground_heading_rate_rad_s=arc_heading,
        leading_edge_relative_arc_heading_rate_rad_s=relative_heading,
        sasho_face_center_rotation=sasho_rotation,
        screw_axis=_instantaneous_screw_axis(state),
    )


__all__ = [
    "InstantaneousScrewAxis",
    "SASHO_FACE_CENTER_ROTATION_METHOD_ID",
    "SashoFaceCenterRotationAoa",
    "WedgeKinematicAnalysis",
    "WedgeKinematicState",
    "analyze_wedge_kinematics",
    "angle_of_attack_deg",
    "sasho_face_center_rotation_aoa",
]
