"""Independent finite-grip potential and boundary-work controls."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from shared.python.golf_club._grip_finite_response import (
    FinitePoseGrip,
    finite_grip_response,
)
from shared.python.golf_club._grip_moving_kinematics import MaterialPointMotion


def _motion(position: object, velocity: object) -> MaterialPointMotion:
    pose = np.eye(4)
    pose[:3, 3] = position
    return MaterialPointMotion(pose, velocity, np.zeros(6), "inertial")


def _law() -> FinitePoseGrip:
    factor = np.diag(np.arange(1.0, 7))
    factor[0, 4], factor[3, 2] = 0.7, -0.4
    return FinitePoseGrip(factor / 3, factor / 2, factor, "synthetic")


def test_finite_rotation_storage_matches_independent_coordinate_trajectory() -> None:
    law = _law()
    q = np.array([0.2, -0.4, 0.1, 0, 0, 1.2])
    velocity = np.array([0.7, 0.2, -0.3, 0, 0, -0.8])
    acceleration = np.array([-0.1, 0.4, 0.2, 0, 0, 0.3])
    pose = np.eye(4)
    rotation = Rotation.from_rotvec(q[3:]).as_matrix()
    pose[:3, :3], pose[:3, 3] = rotation, q[:3]
    body_velocity = np.r_[rotation.T @ velocity[:3], velocity[3:]]
    body_rate = np.r_[
        rotation.T @ acceleration[:3] - np.cross(velocity[3:], body_velocity[:3]),
        acceleration[3:],
    ]
    root = MaterialPointMotion(pose, body_velocity, body_rate, "inertial")
    response = finite_grip_response(law, root, _motion(np.zeros(3), np.zeros(6)))
    mass, damping, stiffness = (
        np.asarray(factor).T @ factor
        for factor in (law.inertance_factor, law.damping_factor, law.stiffness_factor)
    )
    assert response.storage.inertial_energy_j == pytest.approx(
        velocity @ mass @ velocity / 2
    )
    assert response.storage.elastic_energy_j == pytest.approx(q @ stiffness @ q / 2)
    assert response.storage.dissipated_power_w == pytest.approx(
        velocity @ damping @ velocity
    )
    expected_rate = velocity @ mass @ acceleration + q @ stiffness @ velocity
    assert response.storage.stored_energy_rate_w == pytest.approx(expected_rate)
    assert response.power_residual_w == pytest.approx(0, abs=2e-13)


def test_elastic_reaction_is_negative_independent_finite_pose_energy_gradient() -> None:
    law = replace(
        _law(), inertance_factor=np.zeros((6, 6)), damping_factor=np.zeros((6, 6))
    )
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0.8, -0.4, 0.7]).as_matrix()
    pose[:3, 3] = [0.2, -0.1, 0.3]
    root = MaterialPointMotion(pose, np.zeros(6), np.zeros(6), "inertial")
    anchor = _motion(np.zeros(3), np.zeros(6))
    response = finite_grip_response(law, root, anchor)
    factor = np.asarray(law.stiffness_factor)

    def energy(current: np.ndarray) -> float:
        coordinates = np.r_[
            current[:3, 3], Rotation.from_matrix(current[:3, :3]).as_rotvec()
        ]
        elastic = factor @ coordinates
        return float(elastic @ elastic / 2)

    gradient = []
    step = 2e-6
    for direction in np.eye(6):
        generator = np.zeros((4, 4))
        x, y, z = direction[3:]
        generator[:3, :3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        generator[:3, 3] = direction[:3]
        gradient.append(
            (
                energy(pose @ expm(step * generator))
                - energy(pose @ expm(-step * generator))
            )
            / (2 * step)
        )
    np.testing.assert_allclose(
        response.root_wrench, -np.asarray(gradient), rtol=2e-8, atol=2e-8
    )


def test_prescribed_anchor_work_is_separate_from_damping_loss() -> None:
    law = FinitePoseGrip(np.zeros((6, 6)), np.eye(6) * 2, np.zeros((6, 6)), "synthetic")
    anchor = _motion([0, 0, 0], [-3, 0, 0, 0, 0, 0])
    root = _motion([0.2, 0, 0], [-2, 0, 0, 0, 0, 0])
    response = finite_grip_response(law, root, anchor)
    assert response.root_power_w == pytest.approx(8)
    assert response.anchor_power_w == pytest.approx(-12)
    assert response.storage.dissipated_power_w == pytest.approx(4)
    assert response.power_residual_w == pytest.approx(0, abs=1e-14)
    np.testing.assert_allclose(
        np.asarray(response.root_wrench) + response.anchor_wrench, 0
    )


def test_free_boundary_is_zero_and_coefficients_are_owned_and_strict() -> None:
    zero = np.zeros((6, 6))
    law = FinitePoseGrip(zero, zero, zero, "synthetic")
    zero[0, 0] = 99
    response = finite_grip_response(
        law, _motion([0.2, 0, 0], [1, 0, 0, 0, 0, 0]), _motion([0, 0, 0], np.zeros(6))
    )
    np.testing.assert_array_equal(response.root_wrench, np.zeros(6))
    assert response.storage.input_power_w == 0
    with pytest.raises((TypeError, ValueError)):
        replace(law, damping_factor=np.ones((6, 6), dtype=complex))
    with pytest.raises(ValueError):
        replace(law, source_id="")
    with pytest.raises(TypeError):
        finite_grip_response(
            object(), _motion([0, 0, 0], np.zeros(6)), _motion([0, 0, 0], np.zeros(6))
        )


def _hat(vector: object) -> np.ndarray:
    result = np.zeros((4, 4))
    value = np.asarray(vector)
    x, y, z = value[3:]
    result[:3, :3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    result[:3, 3] = value[:3]
    return result


def _changed_observer(motion: MaterialPointMotion) -> MaterialPointMotion:
    # Differentiate H_new(t)=L(t)H(t) at L(0)=I using matrix products.
    current = np.asarray(motion.pose)
    inverse = np.linalg.inv(current)
    observer_velocity = _hat([0.7, -0.5, 0.2, 0.3, 0.1, -0.4])
    observer_rate = _hat([-0.2, 0.3, 0.1, -0.1, 0.4, 0.2])
    carried = inverse @ observer_velocity @ current
    velocity = _hat(motion.twist)
    new_velocity = velocity + carried
    new_rate = (
        _hat(motion.twist_rate)
        + inverse @ observer_rate @ current
        + carried @ velocity
        - velocity @ carried
    )

    def vector(generator: np.ndarray) -> np.ndarray:
        return np.r_[
            generator[:3, 3], generator[2, 1], generator[0, 2], generator[1, 0]
        ]

    return replace(
        motion,
        twist=vector(new_velocity),
        twist_rate=vector(new_rate),
        observer_id="moving",
    )


def test_accelerating_observer_changes_port_power_but_not_relative_storage() -> None:
    law = _law()
    root = _motion([0.2, -0.3, 0.4], [0.8, -0.2, 0.1, 0.3, 0.4, -0.2])
    pose = np.asarray(root.pose).copy()
    pose[:3, :3] = Rotation.from_rotvec([0.4, 0.2, 0.7]).as_matrix()
    root = replace(root, pose=pose)
    anchor = _motion([-0.2, 0.1, 0.3], [0.2, 0.1, -0.3, -0.4, 0.2, 0.1])
    original = finite_grip_response(law, root, anchor)
    transformed = finite_grip_response(
        law, _changed_observer(root), _changed_observer(anchor)
    )
    for field in (
        "inertial_energy_j",
        "elastic_energy_j",
        "dissipated_power_w",
        "stored_energy_rate_w",
    ):
        assert getattr(transformed.storage, field) == pytest.approx(
            getattr(original.storage, field), abs=1e-11
        )
    np.testing.assert_allclose(
        transformed.root_wrench, original.root_wrench, atol=1e-11
    )
    assert transformed.root_power_w != pytest.approx(original.root_power_w)
    assert transformed.root_power_w + transformed.anchor_power_w == pytest.approx(
        original.root_power_w + original.anchor_power_w, abs=1e-11
    )
    assert transformed.power_residual_w == pytest.approx(0, abs=1e-11)


def test_nonfinite_coordinate_power_is_refused() -> None:
    root = _motion([0, 0, 0], [1e308, 0, 0, 0, 0, 0])
    anchor = _motion([0, 0, 0], np.zeros(6))
    with pytest.raises(ValueError, match="finite"):
        finite_grip_response(_law(), root, anchor)
