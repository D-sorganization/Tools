"""Declared contact spin, normal rotation and observer objectivity."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shared.python.swing_sim.impact._friction_transport import (
    ContactTransport,
    contact_transport,
)
from shared.python.swing_sim.impact._spatial_contact_kinematics import (
    PlaneSphereKinematics,
)

from .test_normal_shaft_contact import _case


def test_mean_normal_spin_differs_from_face_attached_transport() -> None:
    model, shaft, ball = _case()
    previous = model.kinematics(shaft, ball)
    spun_ball = replace(ball, twist=(0, 0, -0.4, 0, 0, 10))
    current = PlaneSphereKinematics(previous.face, spun_ball, previous.geometry)
    face = contact_transport(previous, current, 0.02, ContactTransport.FACE)
    mean = contact_transport(previous, current, 0.02, ContactTransport.MEAN_NORMAL_SPIN)
    np.testing.assert_allclose(face, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(
        mean, Rotation.from_rotvec([0, 0, 0.1]).as_matrix(), atol=1e-14
    )
    np.testing.assert_allclose(mean @ previous.normal, current.normal, atol=1e-14)
    assert np.linalg.det(mean) == pytest.approx(1)


def test_frame_transport_rotates_covariantly_with_observer() -> None:
    model, shaft, ball = _case()
    previous = model.kinematics(shaft, ball)
    pose = np.asarray(previous.face.pose).copy()
    pose[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    face = replace(previous.face, pose=pose, twist=(0, 0, 0, 1, 2, 3))
    current = PlaneSphereKinematics(face, ball, previous.geometry)
    observer = np.eye(4)
    observer[:3, :3] = Rotation.from_rotvec([0.4, 0.7, -0.6]).as_matrix()

    def rotated(snapshot: PlaneSphereKinematics) -> PlaneSphereKinematics:
        return PlaneSphereKinematics(
            replace(snapshot.face, pose=observer @ snapshot.face.pose),
            replace(snapshot.ball, pose=observer @ snapshot.ball.pose),
            snapshot.geometry,
        )

    for convention in ContactTransport:
        q = contact_transport(previous, current, 0.01, convention)
        transformed = contact_transport(
            rotated(previous), rotated(current), 0.01, convention
        )
        np.testing.assert_allclose(
            transformed, observer[:3, :3] @ q @ observer[:3, :3].T, atol=1e-13
        )
        np.testing.assert_allclose(q @ previous.normal, current.normal, atol=1e-13)


def test_transport_refuses_undeclared_convention_and_invalid_time() -> None:
    model, shaft, ball = _case()
    contact = model.kinematics(shaft, ball)
    with pytest.raises(TypeError):
        contact_transport(contact, contact, 0.01, "face")
    for step in (0, -1, float("nan"), True):
        with pytest.raises((TypeError, ValueError)):
            contact_transport(contact, contact, step, ContactTransport.FACE)
