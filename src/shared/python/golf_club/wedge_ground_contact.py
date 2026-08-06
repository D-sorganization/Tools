"""Continuous swept wedge clearance against an immutable planar ground datum."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ._validation import (
    Vector3,
    require_finite_float,
    require_identifier,
    require_vector3,
)
from ._wedge_sweep import (
    interpolated_pose,
    interpolated_twist,
    swept_times,
    validated_sweep_arrays,
)
from .wedge_geometry import (
    WedgeContactCandidate,
    WedgeContactFeature,
    wedge_contact_candidates,
)
from .wedge_parameters import WedgeHeadParameters

_UNIT_TOLERANCE = 1e-9
_CONTACT_TOLERANCE_M = 1e-10
_TIME_TOLERANCE_S = 1e-9
_SWEEP_SUBDIVISIONS = 8
_BISECTION_ITERATIONS = 48
_LIMITATIONS = (
    "Rigid geometric contact against a static plane only; no turf deformation, "
    "soil force, divot depth, friction impulse, or injury inference is modeled."
)


class ContactSequence(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Explicit ball and first-ground-contact ordering."""

    BALL_FIRST = "ball_first"
    GROUND_FIRST = "ground_first"
    SIMULTANEOUS = "simultaneous"
    BALL_ONLY = "ball_only"
    GROUND_ONLY_MISS = "ground_only_miss"
    NO_CONTACT_MISS = "no_contact_miss"


@dataclass(frozen=True)
class GroundPlane:
    """Static planar ground expressed in one declared inertial frame."""

    frame_id: str = "ground_frame:x_target,y_up,z_right"
    point_m: Vector3 = (0.0, 0.0, 0.0)
    normal_unit: Vector3 = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", require_identifier(self.frame_id, "frame_id")
        )
        object.__setattr__(self, "point_m", require_vector3(self.point_m, "point_m"))
        normal = require_vector3(self.normal_unit, "normal_unit")
        if not math.isclose(
            float(np.linalg.norm(normal)), 1.0, abs_tol=_UNIT_TOLERANCE, rel_tol=0.0
        ):
            raise ValueError("normal_unit must be unit length")
        object.__setattr__(self, "normal_unit", normal)


@dataclass(frozen=True)
class WedgeClearanceSample:
    """Minimum named-candidate clearance at one swept-envelope instant."""

    time_s: float
    minimum_clearance_m: float
    feature: WedgeContactFeature
    world_point_m: Vector3


@dataclass(frozen=True)
class WedgeGroundContactEvent:
    """Refined first plane crossing and its instantaneous rigid-body velocity."""

    time_s: float
    feature: WedgeContactFeature
    world_point_m: Vector3
    normal_velocity_mps: float
    tangential_velocity_mps: Vector3
    pose_head_to_ground: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class WedgeGroundClearanceAnalysis:
    """Auditable clearance envelope, event, sequencing, and impact margins."""

    frame_id: str
    envelope: tuple[WedgeClearanceSample, ...]
    first_ground_contact: WedgeGroundContactEvent | None
    sequence: ContactSequence
    ball_contact_time_s: float | None
    leading_edge_clearance_at_ball_m: float | None
    minimum_pre_ball_clearance_m: float | None
    ground_after_ball_time_margin_s: float | None
    low_point_time_s: float
    low_point_world_m: Vector3
    low_point_feature: WedgeContactFeature
    delivered_bounce_deg_at_ball: float | None
    limitations: str = _LIMITATIONS


def _world_point(pose: np.ndarray, candidate: WedgeContactCandidate) -> np.ndarray:
    result: np.ndarray = (
        pose[:3, :3] @ np.asarray(candidate.local_point_m) + pose[:3, 3]
    )
    return result


def _tuple3(values: np.ndarray) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _matrix4(
    values: np.ndarray,
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        (float(row[0]), float(row[1]), float(row[2]), float(row[3])) for row in values
    )


def _clearance(point: np.ndarray, ground: GroundPlane) -> float:
    return float(np.dot(point - np.asarray(ground.point_m), ground.normal_unit))


def _candidate_values(
    time_s: float,
    times: np.ndarray,
    poses: np.ndarray,
    candidates: tuple[WedgeContactCandidate, ...],
    ground: GroundPlane,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = interpolated_pose(times, poses, time_s)
    points = np.vstack([_world_point(pose, candidate) for candidate in candidates])
    clearances = np.asarray([_clearance(point, ground) for point in points])
    return pose, points, clearances


def _refined_crossing(
    lower_s: float,
    upper_s: float,
    candidate: WedgeContactCandidate,
    times: np.ndarray,
    poses: np.ndarray,
    ground: GroundPlane,
) -> float:
    for _ in range(_BISECTION_ITERATIONS):
        midpoint = 0.5 * (lower_s + upper_s)
        pose = interpolated_pose(times, poses, midpoint)
        if _clearance(_world_point(pose, candidate), ground) > 0.0:
            lower_s = midpoint
        else:
            upper_s = midpoint
    return upper_s


def _first_contact(
    swept_times: np.ndarray,
    all_clearances: np.ndarray,
    candidates: tuple[WedgeContactCandidate, ...],
    times: np.ndarray,
    poses: np.ndarray,
    twists: np.ndarray,
    ground: GroundPlane,
) -> WedgeGroundContactEvent | None:
    if float(np.min(all_clearances[0])) <= _CONTACT_TOLERANCE_M:
        candidate_index = int(np.argmin(all_clearances[0]))
        event_time = float(swept_times[0])
    else:
        crossings: list[tuple[float, int]] = []
        for row in range(1, len(swept_times)):
            for candidate_index, candidate in enumerate(candidates):
                if (
                    all_clearances[row - 1, candidate_index]
                    > 0.0
                    >= all_clearances[row, candidate_index]
                ):
                    crossings.append(
                        (
                            _refined_crossing(
                                float(swept_times[row - 1]),
                                float(swept_times[row]),
                                candidate,
                                times,
                                poses,
                                ground,
                            ),
                            candidate_index,
                        )
                    )
        if not crossings:
            return None
        event_time, candidate_index = min(crossings)
    candidate = candidates[candidate_index]
    pose = interpolated_pose(times, poses, event_time)
    point = _world_point(pose, candidate)
    twist = interpolated_twist(times, twists, event_time)
    velocity = twist[3:] + np.cross(twist[:3], point - pose[:3, 3])
    normal = np.asarray(ground.normal_unit)
    normal_velocity = float(np.dot(velocity, normal))
    tangential = velocity - normal_velocity * normal
    return WedgeGroundContactEvent(
        time_s=event_time,
        feature=candidate.feature,
        world_point_m=_tuple3(point),
        normal_velocity_mps=normal_velocity,
        tangential_velocity_mps=_tuple3(tangential),
        pose_head_to_ground=_matrix4(pose),
    )


def _sequence(
    ball_time_s: float | None, event: WedgeGroundContactEvent | None
) -> ContactSequence:
    if ball_time_s is None:
        return (
            ContactSequence.GROUND_ONLY_MISS
            if event is not None
            else ContactSequence.NO_CONTACT_MISS
        )
    if event is None:
        return ContactSequence.BALL_ONLY
    difference = event.time_s - ball_time_s
    if abs(difference) <= _TIME_TOLERANCE_S:
        return ContactSequence.SIMULTANEOUS
    return (
        ContactSequence.BALL_FIRST if difference > 0.0 else ContactSequence.GROUND_FIRST
    )


def _delivered_bounce_deg(
    parameters: WedgeHeadParameters, pose: np.ndarray, ground: GroundPlane
) -> float:
    bounce = math.radians(parameters.bounce_deg)
    local_sole = np.array(
        [
            -parameters.sole_width_m * math.cos(bounce),
            parameters.sole_width_m * math.sin(bounce),
            0.0,
        ]
    )
    world_sole = pose[:3, :3] @ local_sole
    vertical = float(np.dot(world_sole, ground.normal_unit))
    horizontal = world_sole - vertical * np.asarray(ground.normal_unit)
    return math.degrees(math.atan2(vertical, float(np.linalg.norm(horizontal))))


def analyze_wedge_ground_clearance(
    parameters: WedgeHeadParameters,
    times_s: object,
    poses: object,
    twists: object,
    ground: GroundPlane,
    *,
    ball_contact_time_s: float | None = None,
) -> WedgeGroundClearanceAnalysis:
    """Analyze a retained rigid-head sweep without inventing turf mechanics."""
    if not isinstance(parameters, WedgeHeadParameters):
        raise TypeError("parameters must be WedgeHeadParameters")
    if not isinstance(ground, GroundPlane):
        raise TypeError("ground must be GroundPlane")
    times, pose_array, twist_array = validated_sweep_arrays(times_s, poses, twists)
    if ball_contact_time_s is not None:
        ball_contact_time_s = require_finite_float(
            ball_contact_time_s, "ball_contact_time_s"
        )
        if ball_contact_time_s < times[0] or ball_contact_time_s > times[-1]:
            raise ValueError("ball_contact_time_s must fall within the retained sweep")
    candidates = wedge_contact_candidates(parameters)
    sweep_times = swept_times(times, _SWEEP_SUBDIVISIONS)
    points_by_time: list[np.ndarray] = []
    clearances_by_time: list[np.ndarray] = []
    envelope: list[WedgeClearanceSample] = []
    for time_s in sweep_times:
        _, points, clearances = _candidate_values(
            float(time_s), times, pose_array, candidates, ground
        )
        feature_index = int(np.argmin(clearances))
        points_by_time.append(points)
        clearances_by_time.append(clearances)
        envelope.append(
            WedgeClearanceSample(
                time_s=float(time_s),
                minimum_clearance_m=float(clearances[feature_index]),
                feature=candidates[feature_index].feature,
                world_point_m=_tuple3(points[feature_index]),
            )
        )
    all_clearances = np.vstack(clearances_by_time)
    event = _first_contact(
        sweep_times,
        all_clearances,
        candidates,
        times,
        pose_array,
        twist_array,
        ground,
    )
    low_index = int(np.argmin([sample.minimum_clearance_m for sample in envelope]))
    low = envelope[low_index]
    leading_at_ball: float | None = None
    minimum_pre_ball: float | None = None
    delivered_bounce: float | None = None
    if ball_contact_time_s is not None:
        ball_pose, _, ball_clearances = _candidate_values(
            ball_contact_time_s, times, pose_array, candidates, ground
        )
        leading_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.feature.value.startswith("leading_edge")
        ]
        leading_at_ball = float(np.min(ball_clearances[leading_indices]))
        prior = [
            sample.minimum_clearance_m
            for sample in envelope
            if sample.time_s <= ball_contact_time_s
        ]
        minimum_pre_ball = min(prior + [float(np.min(ball_clearances))])
        delivered_bounce = _delivered_bounce_deg(parameters, ball_pose, ground)
    margin = (
        None
        if ball_contact_time_s is None or event is None
        else event.time_s - ball_contact_time_s
    )
    return WedgeGroundClearanceAnalysis(
        frame_id=ground.frame_id,
        envelope=tuple(envelope),
        first_ground_contact=event,
        sequence=_sequence(ball_contact_time_s, event),
        ball_contact_time_s=ball_contact_time_s,
        leading_edge_clearance_at_ball_m=leading_at_ball,
        minimum_pre_ball_clearance_m=minimum_pre_ball,
        ground_after_ball_time_margin_s=margin,
        low_point_time_s=low.time_s,
        low_point_world_m=low.world_point_m,
        low_point_feature=low.feature,
        delivered_bounce_deg_at_ball=delivered_bounce,
    )


__all__ = [
    "ContactSequence",
    "GroundPlane",
    "WedgeClearanceSample",
    "WedgeGroundClearanceAnalysis",
    "WedgeGroundContactEvent",
    "analyze_wedge_ground_clearance",
]
