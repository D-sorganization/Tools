"""Full head inertia linearized in a translating and rotating observer frame."""

from __future__ import annotations

import numpy as np

from ._rotating_body_contracts import RotatingFrameState, _BodyTangent, strict_vector3
from ._rotating_body_kernel import (
    body_geometry,
    equilibrium_force,
    kinetic_position_hessian,
    rotation_hessian,
    velocity_position,
)
from ._shaft_spatial_element import tip_spatial_inertia
from ._validation import Vector3
from .types import ComponentMassProperties


def rotating_body_tangent(
    body: ComponentMassProperties,
    frame: RotatingFrameState,
    reference_position_m: Vector3,
) -> _BodyTangent:
    """Return consistent M, gyroscopic, centrifugal, Euler and acceleration terms.

    Preconditions: matching frame, finite SI motion and reference position.
    The body's COM offset is measured from that reference position. Its inertia
    is about COM in the aligned reference orientation. Postconditions: finite,
    independent coefficient arrays; gyroscopic part does zero quadratic work.
    This local tangent alone does not establish a loaded stable equilibrium.
    """
    if not isinstance(body, ComponentMassProperties):
        raise TypeError("body must be ComponentMassProperties")
    if not isinstance(frame, RotatingFrameState):
        raise TypeError("frame must be RotatingFrameState")
    if body.frame_id != frame.frame_id:
        raise ValueError("body and observer frame must match")
    reference = strict_vector3(reference_position_m, "reference_position_m")
    motion = tuple(
        np.asarray(value)
        for value in (
            frame.angular_velocity_rad_s,
            frame.angular_acceleration_rad_s2,
            frame.origin_acceleration_m_s2,
        )
    )
    with np.errstate(over="ignore", invalid="ignore"):
        data = body_geometry(body, np.asarray(reference))
        mass = tip_spatial_inertia(body)
        mixed = velocity_position(data, motion[0])
        acceleration = np.zeros((6, 6))
        acceleration[3:, 3:] = data.mass * rotation_hessian(motion[2], data.offset)
        arrays = (
            mass,
            mixed,
            mixed - mixed.T,
            -kinetic_position_hessian(data, motion[0]),
            velocity_position(data, motion[1]),
            acceleration,
            equilibrium_force(data, motion),
        )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("rotating body tangent must be finite")
    return _BodyTangent(body.component_id, frame, reference, *arrays)


__all__ = ["RotatingFrameState", "rotating_body_tangent"]
