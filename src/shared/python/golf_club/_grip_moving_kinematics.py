"""Private finite-pose kinematics and dual ports for a moving grip anchor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array, vector6
from ._shaft_se3 import (
    _rigid_pose,
    log_pose,
    right_jacobian,
    right_jacobian_derivative,
)
from ._validation import require_identifier


@dataclass(frozen=True)
class MaterialPointMotion:
    """Owned pose, body twist and its derivative in one common observer frame.

    Pose maps material axes into the observer. Linear-first twist V satisfies
    Hdot=H hat(V); twist_rate is dV/dt, not the absolute linear acceleration
    expressed in material axes. Inputs must be finite real arrays and a proper
    rigid pose. Stored tuples cannot be changed through input-array aliases.
    A state does not itself prove consistency with a measured pose history.
    """

    pose: object
    twist: object
    twist_rate: object
    observer_id: str

    def __post_init__(self) -> None:
        pose = _rigid_pose(self.pose)
        object.__setattr__(
            self, "pose", tuple(tuple(float(x) for x in row) for row in pose)
        )
        for name in ("twist", "twist_rate"):
            object.__setattr__(self, name, vector6(getattr(self, name), name))
        object.__setattr__(
            self, "observer_id", require_identifier(self.observer_id, "observer_id")
        )


@dataclass(frozen=True)
class MovingGripKinematics:
    """Fresh relative coordinates/rates and work-conjugate material motion maps.

    Coordinates are actual Cartesian separation in anchor axes followed by the
    principal relative rotation vector, not the SE(3) translation generator.
    qdot=Ar Vr+Aa Va. A generalized coordinate effort g applies material
    reactions -Ar.T g and -Aa.T g at the root and anchor origins respectively.
    This is kinematics; no nonlinear grip law, hand mass or damping is inferred.
    """

    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    root_motion_map: np.ndarray
    anchor_motion_map: np.ndarray


def _rotation_rates(
    rotation_vector: np.ndarray, relative_omega: np.ndarray, relative_rate: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.r_[np.zeros(3), rotation_vector]
    jacobian = right_jacobian(coordinates)[3:, 3:]
    inverse = np.linalg.solve(jacobian, np.eye(3))
    velocity = inverse @ relative_omega
    derivative = right_jacobian_derivative(coordinates, np.r_[np.zeros(3), velocity])[
        3:, 3:
    ]
    return inverse, velocity, inverse @ (relative_rate - derivative @ velocity)


def _motion_maps(
    rotation: np.ndarray, inverse: np.ndarray, separation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    zero = np.zeros((3, 3))
    offset_cross = np.cross(separation, np.eye(3)).T
    root = np.block([[rotation, zero], [zero, inverse]])
    anchor = np.block([[-np.eye(3), offset_cross], [zero, -inverse @ rotation.T]])
    return root, anchor


def _relative_geometry(
    root: MaterialPointMotion, anchor: MaterialPointMotion
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_pose, anchor_pose = np.asarray(root.pose), np.asarray(anchor.pose)
    rotation = anchor_pose[:3, :3].T @ root_pose[:3, :3]
    separation = anchor_pose[:3, :3].T @ (root_pose[:3, 3] - anchor_pose[:3, 3])
    relative = np.eye(4)
    relative[:3, :3] = rotation
    return rotation, separation, log_pose(relative)[3:]


def moving_grip_kinematics(
    root: MaterialPointMotion, anchor: MaterialPointMotion
) -> MovingGripKinematics:
    """Map two consistent material states into one finite relative-coordinate port.

    Require the same observer and the existing principal-log chart margin.
    Retain all moving-anchor transport terms; numerical failure is a refusal.
    These maps do not extend the small-rotation PassiveGripImpedance contract.
    """
    if not isinstance(root, MaterialPointMotion) or not isinstance(
        anchor, MaterialPointMotion
    ):
        raise TypeError("root and anchor must be MaterialPointMotion")
    if root.observer_id != anchor.observer_id:
        raise ValueError("root and anchor must use the same observer")
    vr, ar, va, aa = map(
        np.asarray, (root.twist, root.twist_rate, anchor.twist, anchor.twist_rate)
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            rotation, separation, phi = _relative_geometry(root, anchor)
            omega = vr[3:] - rotation.T @ va[3:]
            omega_rate = (
                ar[3:] + np.cross(omega, rotation.T @ va[3:]) - rotation.T @ aa[3:]
            )
            inverse, phi_rate, phi_acceleration = _rotation_rates(
                phi, omega, omega_rate
            )
            rate = rotation @ vr[:3] - va[:3] + np.cross(separation, va[3:])
            acceleration = (
                rotation @ (np.cross(omega, vr[:3]) + ar[:3])
                - aa[:3]
                + np.cross(rate, va[3:])
                + np.cross(separation, aa[3:])
            )
            root_map, anchor_map = _motion_maps(rotation, inverse, separation)
            return MovingGripKinematics(
                finite_array(np.r_[separation, phi], (6,), "relative coordinates"),
                finite_array(np.r_[rate, phi_rate], (6,), "relative velocity"),
                finite_array(
                    np.r_[acceleration, phi_acceleration], (6,), "relative acceleration"
                ),
                finite_array(root_map, (6, 6), "root motion map"),
                finite_array(anchor_map, (6, 6), "anchor motion map"),
            )
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("moving grip numerical evaluation failed") from error


__all__ = ()
