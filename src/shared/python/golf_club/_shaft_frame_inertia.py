"""Rotating-frame inertial residual and derivatives on a deformed SE(3) section.

This is an instantaneous, zero-relative-velocity linearization. It supplies no
elastic equilibrium, stability certificate, grip law or calibrated bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._rotating_body_contracts import RotatingFrameState
from ._shaft_inertia import InertiaSample, SectionInertia
from ._shaft_se3 import (
    _relative_maps,
    _rigid_pose,
    exp_twist,
    log_pose,
    section_velocity_map,
    section_velocity_map_derivative,
    twist_ad,
)
from ._shaft_spatial_element import tip_spatial_inertia


@dataclass(frozen=True)
class FrameSectionInertia:
    """Fresh linear-first material-coordinate arrays, retaining every nodal DOF.

    Residual is physical inertia on the left side of the balance, in N and N m.
    Moving Jacobian differentiates that material wrench under H_i Exp(delta_i).
    It is not the fixed-chart Hessian and need not be symmetric. Gyroscopic
    force is G times relative material velocity, in the same SI convention as
    SectionInertia. No symmetry projection or damping interpretation is applied.
    """

    residual: np.ndarray
    moving_jacobian: np.ndarray
    gyroscopic: np.ndarray


def _point_frame_motion(
    pose: np.ndarray, frame: RotatingFrameState
) -> tuple[np.ndarray, np.ndarray]:
    rotation, position = pose[:3, :3], pose[:3, 3]
    omega = np.asarray(frame.angular_velocity_rad_s)
    alpha = np.asarray(frame.angular_acceleration_rad_s2)
    motion = np.r_[rotation.T @ np.cross(omega, position), rotation.T @ omega]
    acceleration = np.r_[
        rotation.T @ (frame.origin_acceleration_m_s2 + np.cross(alpha, position)),
        rotation.T @ alpha,
    ]
    return motion, acceleration


def _point_derivatives(
    inertia: np.ndarray, motion: np.ndarray, acceleration: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Material pose derivative and relative-velocity derivative of rigid inertia."""
    motion_ad = twist_ad(motion)
    momentum = inertia @ motion
    position_columns, velocity_columns = [], []
    for axis in np.eye(6):
        motion_change = motion_ad @ axis
        position_columns.append(
            inertia @ twist_ad(acceleration) @ axis
            - twist_ad(motion_change).T @ momentum
            - motion_ad.T @ inertia @ motion_change
        )
        velocity_columns.append(
            inertia @ motion_ad @ axis
            - twist_ad(axis).T @ momentum
            - motion_ad.T @ inertia @ axis
        )
    return np.column_stack(position_columns), np.column_stack(velocity_columns)


def _sample_frame_inertia(
    sample: InertiaSample,
    left_pose: np.ndarray,
    relative: np.ndarray,
    frame: RotatingFrameState,
) -> FrameSectionInertia:
    pose = left_pose @ exp_twist(sample.fraction * relative)
    mapping = section_velocity_map(relative, sample.fraction)
    relative_map, _, _ = _relative_maps(relative)
    inertia = tip_spatial_inertia(sample.body)
    motion, acceleration = _point_frame_motion(pose, frame)
    wrench = inertia @ acceleration - twist_ad(motion).T @ inertia @ motion
    position, velocity = _point_derivatives(inertia, motion, acceleration)
    columns = []
    for axis in np.eye(12):
        derivative = section_velocity_map_derivative(
            relative, sample.fraction, relative_map @ axis
        )
        columns.append(derivative.T @ wrench + mapping.T @ position @ mapping @ axis)
    return FrameSectionInertia(
        mapping.T @ wrench,
        np.column_stack(columns),
        mapping.T @ velocity @ mapping,
    )


def rotating_section_inertia(
    section: SectionInertia, poses: object, frame: RotatingFrameState
) -> FrameSectionInertia:
    """Linearize frame inertia about a supplied deformed, relatively resting shape.

    Poses map material axes into the observer frame identified by frame.frame_id.
    The existing frame record supplies physical origin/angular accelerations;
    no coordinate second derivatives or inferred swing loads are substituted.
    Samples already contain physical quadrature weights. The output retains
    applied-frame inertia separately from internal elastic and external forces.
    """
    if not isinstance(section, SectionInertia):
        raise TypeError("section must be SectionInertia")
    if not isinstance(frame, RotatingFrameState):
        raise TypeError("frame must be RotatingFrameState")
    current = finite_array(poses, (2, 4, 4), "section poses")
    left, right = _rigid_pose(current[0]), _rigid_pose(current[1])
    relative = log_pose(np.linalg.solve(left, right))
    residual, jacobian, gyroscopic = (
        np.zeros(12),
        np.zeros((12, 12)),
        np.zeros((12, 12)),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        for sample in section.samples:
            result = _sample_frame_inertia(sample, left, relative, frame)
            residual += result.residual
            jacobian += result.moving_jacobian
            gyroscopic += result.gyroscopic
    return FrameSectionInertia(
        finite_array(residual, (12,), "frame inertia residual"),
        finite_array(jacobian, (12, 12), "frame inertia Jacobian"),
        finite_array(gyroscopic, (12, 12), "frame gyroscopic matrix"),
    )


__all__ = ()
