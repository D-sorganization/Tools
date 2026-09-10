"""Explicit first-order contact-frame transport, including constitutive spin."""

from enum import Enum

import numpy as np

from ...golf_club._shaft_se3 import exp_twist
from ...golf_club._validation import require_finite_float
from ._spatial_contact_kinematics import PlaneSphereKinematics


class ContactTransport(Enum):
    """Distinct constitutive conventions; neither is calibrated by declaration."""

    FACE = "face-attached"
    MEAN_NORMAL_SPIN = "mean-normal-spin"


def contact_transport(
    previous: PlaneSphereKinematics,
    current: PlaneSphereKinematics,
    step_s: float,
    convention: ContactTransport,
) -> np.ndarray:
    """Map normals exactly and add the declared endpoint normal twirl.

    Mean spin adds half the ball-minus-face angular velocity along the current
    normal. This is a first-order finite-step convention, not exact integrated
    transport for an arbitrary within-step rotation history.
    """
    if not isinstance(convention, ContactTransport):
        raise TypeError("transport convention must be ContactTransport")
    for state in (previous, current):
        if not isinstance(state, PlaneSphereKinematics):
            raise TypeError("transport requires contact kinematics")
    if previous.face.observer_id != current.face.observer_id:
        raise ValueError("contact transport observers must agree")
    step = require_finite_float(step_s, "transport step", positive=True)
    old_face = np.asarray(previous.face.pose)[:3, :3]
    new_face = np.asarray(current.face.pose)[:3, :3]
    rotation = new_face @ old_face.T
    if convention is ContactTransport.MEAN_NORMAL_SPIN:
        ball_rotation = np.asarray(current.ball.pose)[:3, :3]
        relative_spin = (
            ball_rotation @ np.asarray(current.ball.twist)[3:]
            - new_face @ np.asarray(current.face.twist)[3:]
        )
        angle = 0.5 * step * float(np.asarray(current.normal) @ relative_spin)
        twirl = exp_twist(np.r_[np.zeros(3), angle * np.asarray(current.normal)])
        rotation = twirl[:3, :3] @ rotation
    return np.asarray(rotation)


__all__ = ()
