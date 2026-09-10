"""Independent geometry and work oracles for the candidate spatial contact port."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from shared.python.swing_sim.impact._spatial_contact_kinematics import (
    ContactBodyState,
    PlaneSphereGeometry,
    PlaneSphereKinematics,
)


def _body(position: list[float], twist: list[float]) -> ContactBodyState:
    pose = np.eye(4)
    pose[:3, 3] = position
    pose[:3, :3] = Rotation.from_rotvec([0.2, -0.3, 0.1]).as_matrix()
    return ContactBodyState(pose, twist, "inertial")


def _case() -> PlaneSphereKinematics:
    face = _body([0.1, -0.2, 0.3], [2, -1, 0.5, 7, -3, 4])
    ball = _body([0.16, -0.18, 0.32], [-1, 2, 0.3, -8, 5, 2])
    return PlaneSphereKinematics(
        face, ball, PlaneSphereGeometry(0.021, [0.03, 0, 0], [1, 0, 0])
    )


def _advance(body: ContactBodyState, time: float) -> np.ndarray:
    """Independent homogeneous matrix exponential, with physical body twists."""
    velocity = np.asarray(body.twist)
    generator = np.zeros((4, 4))
    generator[:3, :3] = np.cross(velocity[3:], np.eye(3)).T
    generator[:3, 3] = velocity[:3]
    return np.asarray(body.pose) @ expm(time * generator)


def _gap(face: np.ndarray, ball: np.ndarray, geometry: PlaneSphereGeometry) -> float:
    normal = face[:3, :3] @ geometry.plane_normal
    point = face[:3, 3] + face[:3, :3] @ geometry.plane_point_m
    return float(normal @ (ball[:3, 3] - point) - geometry.radius_m)


def test_rotating_face_gap_rate_agrees_with_independent_pose_differences() -> None:
    contact = _case()
    for step in (2e-6, 1e-6):
        plus = _gap(
            _advance(contact.face, step), _advance(contact.ball, step), contact.geometry
        )
        minus = _gap(
            _advance(contact.face, -step),
            _advance(contact.ball, -step),
            contact.geometry,
        )
        assert contact.gap_rate_mps == pytest.approx(
            (plus - minus) / (2 * step), abs=2e-8
        )
    normal = np.asarray(contact.normal)
    face_pose, ball_pose = np.asarray(contact.face.pose), np.asarray(contact.ball.pose)
    center_only = normal @ (
        ball_pose[:3, :3] @ np.asarray(contact.ball.twist)[:3]
        - face_pose[:3, :3] @ np.asarray(contact.face.twist)[:3]
    )
    assert abs(contact.gap_rate_mps - center_only) > 0.01


def test_common_point_pair_conserves_world_force_torque_and_work() -> None:
    contact = _case()
    force = np.array([12.0, -7.0, 4.0])
    loads = contact.load_pair(force)
    total_force, total_moment = np.zeros(3), np.zeros(3)
    power = 0.0
    for body, wrench in [
        (contact.ball, loads.ball_wrench),
        (contact.face, loads.face_wrench),
    ]:
        pose, effort = np.asarray(body.pose), np.asarray(wrench)
        world_force = pose[:3, :3] @ effort[:3]
        total_force += world_force
        total_moment += pose[:3, :3] @ effort[3:] + np.cross(pose[:3, 3], world_force)
        power += effort @ body.twist
    np.testing.assert_allclose(total_force, 0, atol=2e-14)
    np.testing.assert_allclose(total_moment, 0, atol=2e-14)
    assert loads.power_w == pytest.approx(power, abs=2e-13)
    assert loads.power_w == pytest.approx(
        force @ contact.relative_velocity_mps, abs=2e-13
    )


def test_normal_work_equals_force_times_gap_rate_and_ball_normal_torque_is_zero() -> (
    None
):
    contact = _case()
    loads = contact.load_pair(13 * np.asarray(contact.normal))
    assert loads.power_w == pytest.approx(13 * contact.gap_rate_mps, abs=2e-13)
    np.testing.assert_allclose(np.asarray(loads.ball_wrench)[3:], 0, atol=2e-14)


def test_tangent_force_uses_declared_common_point_at_nonzero_compression() -> None:
    face = ContactBodyState(np.eye(4), np.zeros(6), "inertial")
    pose = np.eye(4)
    pose[0, 3] = 0.018
    ball = ContactBodyState(pose, np.zeros(6), "inertial")
    contact = PlaneSphereKinematics(
        face, ball, PlaneSphereGeometry(0.021, [0, 0, 0], [1, 0, 0])
    )
    loads = contact.load_pair([0, 10, 0])
    assert contact.gap_m == pytest.approx(-0.003)
    assert loads.ball_wrench[5] == pytest.approx(-0.18)
    assert loads.ball_wrench[5] != pytest.approx(-0.21)


def test_common_rigid_motion_produces_no_relative_contact_motion() -> None:
    contact = _case()
    omega, translation = np.array([3.0, -2.0, 7.0]), np.array([2.0, -4.0, 1.0])
    bodies = []
    for body in (contact.face, contact.ball):
        pose = np.asarray(body.pose)
        velocity = translation + np.cross(omega, pose[:3, 3])
        twist = np.r_[pose[:3, :3].T @ velocity, pose[:3, :3].T @ omega]
        bodies.append(replace(body, twist=twist))
    moved = PlaneSphereKinematics(*bodies, contact.geometry)
    np.testing.assert_allclose(moved.relative_velocity_mps, 0, atol=3e-15)
    assert moved.gap_rate_mps == pytest.approx(0, abs=3e-15)


def test_fixed_observer_rotation_translation_and_galilean_boost_preserve_ports() -> (
    None
):
    contact = _case()
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([-0.5, 0.1, 0.7]).as_matrix()
    transform[:3, 3] = [2, -3, 1]
    boost = np.array([5.0, 4.0, -2.0])
    bodies = []
    for body in (contact.face, contact.ball):
        pose = transform @ np.asarray(body.pose)
        twist = np.asarray(body.twist).copy()
        twist[:3] += pose[:3, :3].T @ boost
        bodies.append(ContactBodyState(pose, twist, "new-inertial"))
    moved = PlaneSphereKinematics(*bodies, contact.geometry)
    assert moved.gap_m == pytest.approx(contact.gap_m, abs=3e-15)
    assert moved.gap_rate_mps == pytest.approx(contact.gap_rate_mps, abs=1e-13)
    force = np.array([12.0, -7.0, 4.0])
    original, transformed = (
        contact.load_pair(force),
        moved.load_pair(transform[:3, :3] @ force),
    )
    np.testing.assert_allclose(
        transformed.ball_wrench, original.ball_wrench, atol=3e-13
    )
    np.testing.assert_allclose(
        transformed.face_wrench, original.face_wrench, atol=3e-13
    )
    assert transformed.power_w == pytest.approx(original.power_w, abs=3e-13)


def test_state_and_geometry_own_inputs_and_refuse_incompatible_observers() -> None:
    pose, twist = np.eye(4), np.zeros(6)
    state = ContactBodyState(pose, twist, "inertial")
    pose[0, 3], twist[0] = 8, 9
    assert np.asarray(state.pose)[0, 3] == 0
    assert state.twist[0] == 0
    with pytest.raises(FrozenInstanceError):
        state.__setattr__("observer_id", "changed")
    contact = _case()
    with pytest.raises(ValueError, match="observer"):
        PlaneSphereKinematics(
            contact.face, replace(contact.ball, observer_id="other"), contact.geometry
        )


@pytest.mark.parametrize("bad", [True, "0.02", 0, -1, np.nan, np.inf])
def test_radius_contract_refuses_invalid_values(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        PlaneSphereGeometry(bad, [0, 0, 0], [1, 0, 0])


@pytest.mark.parametrize("normal", [[0, 0, 0], [2, 0, 0], [True, 0, 0], [np.nan, 0, 0]])
def test_normal_contract_refuses_nonunit_or_coerced_values(
    normal: list[object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        PlaneSphereGeometry(0.02, [0, 0, 0], normal)


def test_improper_pose_and_nonfinite_force_are_refused() -> None:
    pose = np.eye(4)
    pose[0, 0] = -1
    with pytest.raises(ValueError):
        ContactBodyState(pose, np.zeros(6), "inertial")
    with pytest.raises(ValueError):
        _case().load_pair([np.inf, 0, 0])
