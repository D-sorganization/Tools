"""Distributed full-section and full-head inertial transport for Tools #5072."""

from __future__ import annotations

import numpy as np

from ._rotating_body_contracts import _BodyTangent
from ._shaft_transport_contracts import (
    ShaftRotaryInertia,
    ShaftRotatingModel,
    _ShaftTransport,
)
from ._shaft_transport_quadrature import (
    element_quadrature,
    quadrature_body,
    section_motion_map,
)
from .rotating_body import RotatingFrameState, rotating_body_tangent
from .shaft_dynamics import ShaftModalSettings
from .shaft_linear_system import (
    _element_length,
    _validate_matrices,
    assemble_shaft_linear_system,
)

_DEFAULT_SETTINGS = ShaftModalSettings()


def _matrix_stack(tangent: _BodyTangent) -> np.ndarray:
    return np.stack(
        (
            tangent.mass,
            tangent.gyroscopic,
            tangent.centrifugal_stiffness,
            tangent.euler_stiffness,
            tangent.acceleration_stiffness,
        )
    )


def _span_transport(
    model: ShaftRotatingModel, frame: RotatingFrameState, count: int
) -> tuple[np.ndarray, np.ndarray]:
    profile = model.rod.profile
    length = _element_length(profile.flexible_length_m, count)
    matrices = np.zeros((5, 6 * (count + 1), 6 * (count + 1)))
    force = np.zeros(6 * (count + 1))
    for element in range(count):
        bounds = element * length, (element + 1) * length
        positions, weights = element_quadrature(profile, bounds)
        indices = slice(6 * element, 6 * element + 12)
        for position, weight in zip(positions, weights, strict=True):
            body = quadrature_body(model, float(position), float(weight))
            tangent = rotating_body_tangent(body, frame, (0, 0, float(position)))
            shape = section_motion_map(float(position - bounds[0]), length)
            matrices[:, indices, indices] += np.einsum(
                "ia,kij,jb->kab", shape, _matrix_stack(tangent), shape
            )
            force[indices] += shape.T @ tangent.equilibrium_force
    return matrices, force


def _attachments(
    model: ShaftRotatingModel,
    frame: RotatingFrameState,
    transport: tuple[np.ndarray, np.ndarray],
) -> None:
    matrices, force = transport
    attachments = model.attachments
    if attachments.tip_body is not None:
        profile = model.rod.profile
        head = rotating_body_tangent(
            attachments.tip_body, frame, (0, 0, profile.flexible_length_m)
        )
        matrices[:, -6:, -6:] += _matrix_stack(head)
        force[-6:] += head.equilibrium_force
    if attachments.grip is not None:
        factor = np.asarray(attachments.grip.inertance_factor)
        matrices[0, :6, :6] += factor.T @ factor


def assemble_shaft_rotating_transport(
    model: ShaftRotatingModel,
    frame: RotatingFrameState,
    settings: ShaftModalSettings = _DEFAULT_SETTINGS,
) -> _ShaftTransport:
    """Assemble transport with full section rotary inertia and tip/grip ports.

    Inputs must identify the same profile/frame and contain explicit measured
    section moments. Returned mass is the quadrature-consistent Rayleigh mass,
    not the old Euler-Bernoulli mass. Elastic stiffness and grip damping reuse
    the stationary assembly. Prestress and equilibrium qualification remain
    separate required operations; the forcing is never silently dropped.
    """
    if not isinstance(model, ShaftRotatingModel):
        raise TypeError("model must be ShaftRotatingModel")
    if not isinstance(frame, RotatingFrameState):
        raise TypeError("frame must be RotatingFrameState")
    profile = model.rod.profile
    if profile.frame_id != frame.frame_id:
        raise ValueError("shaft and observer frame must match")
    stationary = assemble_shaft_linear_system(model.rod, settings, model.attachments)
    with np.errstate(over="ignore", invalid="ignore"):
        matrices, force = _span_transport(model, frame, settings.element_count)
        _attachments(model, frame, (matrices, force))
    _validate_matrices((matrices[0], *matrices[1:], force))
    return _ShaftTransport(
        model=model,
        frame=frame,
        positions_m=stationary.positions_m,
        elastic_stiffness=stationary.stiffness,
        damping=stationary.damping,
        mass=matrices[0],
        gyroscopic=matrices[1],
        centrifugal_stiffness=matrices[2],
        euler_stiffness=matrices[3],
        acceleration_stiffness=matrices[4],
        equilibrium_force=force,
    )


__all__ = [
    "ShaftRotaryInertia",
    "ShaftRotatingModel",
    "assemble_shaft_rotating_transport",
]
