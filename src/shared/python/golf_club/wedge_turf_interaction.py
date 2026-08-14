"""Frame-explicit nine-point wedge/turf wrench quadrature."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ._validation import Vector3
from .turf_contact import (
    TurfContactKinematics,
    TurfContactProfile,
    TurfContactResponse,
    TurfContactStatus,
    evaluate_turf_contact,
)
from .wedge_geometry import WedgeContactFeature, wedge_contact_candidates
from .wedge_ground_contact import GroundPlane
from .wedge_parameters import WedgeHeadParameters

_PATCH_WEIGHT = 1.0 / 9.0
_UNIT_TOLERANCE = 1e-8
_CONTACT_TOLERANCE_M = 1e-10
_LIMITATIONS = (
    "Nine-point sole/leading-edge quadrature over the canonical wedge candidates; "
    "not a continuous pressure field, divot, grass-fracture, or granular-flow model."
)


@dataclass(frozen=True)
class WedgeTurfPatchResponse:
    """One active named wedge patch and its local contact response."""

    feature: WedgeContactFeature
    world_point_m: Vector3
    penetration_m: float
    response: TurfContactResponse


@dataclass(frozen=True)
class WedgeTurfWrench:
    """Aggregate turf wrench about the head-frame origin in the ground frame."""

    frame_id: str
    status: TurfContactStatus
    active_patches: tuple[WedgeTurfPatchResponse, ...]
    force_world_n: Vector3
    torque_at_head_origin_n_m: Vector3
    stored_elastic_energy_j: float
    dissipated_power_w: float
    maximum_penetration_m: float
    supports_turf_rankings: bool
    limitations: str = _LIMITATIONS


def _tuple3(values: np.ndarray) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _pose_array(pose_head_to_ground: object) -> np.ndarray:
    pose: np.ndarray = np.asarray(pose_head_to_ground, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("pose_head_to_ground must be a finite 4x4 matrix")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=_UNIT_TOLERANCE):
        raise ValueError("pose_head_to_ground must have a homogeneous final row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_UNIT_TOLERANCE):
        raise ValueError("pose_head_to_ground rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=_UNIT_TOLERANCE):
        raise ValueError("pose_head_to_ground rotation must be right-handed")
    return pose


def _twist_array(twist_at_head_origin: object) -> np.ndarray:
    twist: np.ndarray = np.asarray(twist_at_head_origin, dtype=float)
    if twist.shape != (6,) or not np.all(np.isfinite(twist)):
        raise ValueError("twist_at_head_origin must be a finite 6-vector")
    return twist


def _weighted_profile(profile: TurfContactProfile) -> TurfContactProfile:
    return replace(
        profile,
        normal_stiffness_n_m=_PATCH_WEIGHT * profile.normal_stiffness_n_m,
        normal_damping_n_s_m=_PATCH_WEIGHT * profile.normal_damping_n_s_m,
    )


def evaluate_wedge_turf_wrench(
    parameters: WedgeHeadParameters,
    profile: TurfContactProfile,
    pose_head_to_ground: object,
    twist_at_head_origin: object,
    ground: GroundPlane,
) -> WedgeTurfWrench:
    """Evaluate the replaceable turf law at all canonical wedge contact patches."""
    if not isinstance(parameters, WedgeHeadParameters):
        raise TypeError("parameters must be WedgeHeadParameters")
    if not isinstance(profile, TurfContactProfile):
        raise TypeError("profile must be TurfContactProfile")
    if not isinstance(ground, GroundPlane):
        raise TypeError("ground must be GroundPlane")
    pose = _pose_array(pose_head_to_ground)
    twist = _twist_array(twist_at_head_origin)
    origin = pose[:3, 3]
    normal = np.asarray(ground.normal_unit)
    ground_point = np.asarray(ground.point_m)
    weighted_profile = _weighted_profile(profile)
    patches: list[WedgeTurfPatchResponse] = []
    force = np.zeros(3)
    torque = np.zeros(3)
    stored_energy = 0.0
    dissipated_power = 0.0
    maximum_penetration = 0.0
    outside_domain = False
    for candidate in wedge_contact_candidates(parameters):
        point = pose[:3, :3] @ np.asarray(candidate.local_point_m) + origin
        signed_clearance = float((point - ground_point) @ normal)
        if signed_clearance > _CONTACT_TOLERANCE_M:
            continue
        penetration = max(0.0, -signed_clearance)
        velocity = twist[3:] + np.cross(twist[:3], point - origin)
        response = evaluate_turf_contact(
            weighted_profile,
            TurfContactKinematics(
                frame_id=ground.frame_id,
                reference_point_m=_tuple3(origin),
                application_point_m=_tuple3(point),
                surface_normal_unit=ground.normal_unit,
                surface_velocity_mps=(0.0, 0.0, 0.0),
                contact_point_velocity_mps=_tuple3(velocity),
                penetration_m=penetration,
            ),
        )
        if response.status is TurfContactStatus.OUTSIDE_CALIBRATED_DOMAIN:
            outside_domain = True
        if response.normal_force_n <= 0.0:
            continue
        patches.append(
            WedgeTurfPatchResponse(
                feature=candidate.feature,
                world_point_m=_tuple3(point),
                penetration_m=penetration,
                response=response,
            )
        )
        force += np.asarray(response.force_world_n)
        torque += np.asarray(response.torque_at_reference_n_m)
        stored_energy += response.stored_elastic_energy_j
        dissipated_power += response.dissipated_power_w
        maximum_penetration = max(maximum_penetration, penetration)
    status = TurfContactStatus.NO_CONTACT
    if patches:
        status = (
            TurfContactStatus.OUTSIDE_CALIBRATED_DOMAIN
            if outside_domain
            else TurfContactStatus.ACTIVE
        )
    elif profile.normal_stiffness_n_m == profile.normal_damping_n_s_m == 0.0:
        status = TurfContactStatus.NO_RESPONSE
    return WedgeTurfWrench(
        frame_id=ground.frame_id,
        status=status,
        active_patches=tuple(patches),
        force_world_n=_tuple3(force),
        torque_at_head_origin_n_m=_tuple3(torque),
        stored_elastic_energy_j=stored_energy,
        dissipated_power_w=dissipated_power,
        maximum_penetration_m=maximum_penetration,
        supports_turf_rankings=profile.supports_turf_rankings,
    )


__all__ = [
    "WedgeTurfPatchResponse",
    "WedgeTurfWrench",
    "evaluate_wedge_turf_wrench",
]
