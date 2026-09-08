"""Spatial-motion oracles for the private SE(3) kinetic quadrature."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm, logm

from shared.python.golf_club._shaft_inertia import InertiaSample, SectionInertia
from shared.python.golf_club._shaft_se3 import (
    exp_twist,
    section_velocity_map,
    section_velocity_map_derivative,
)
from shared.python.golf_club.types import ComponentMassProperties, ComponentRole


def _hat(twist: np.ndarray) -> np.ndarray:
    x, y, z = twist[3:]
    result = np.zeros((4, 4))
    result[:3, :3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    result[:3, 3] = twist[:3]
    return result


def _vee(matrix: np.ndarray) -> np.ndarray:
    return matrix[[2, 0, 1], [1, 2, 0]]


def _poses() -> np.ndarray:
    left = expm(_hat(np.array([0.2, -0.1, 0.3, 0.2, 0.1, -0.1])))
    right = left @ expm(_hat(np.array([0.04, -0.03, 1.0, 0.3, -0.2, 0.15])))
    return np.array([left, right])


def _point(poses: np.ndarray, fraction: float) -> np.ndarray:
    # General 4x4 matrix logarithm/exponential, independent of production SE(3).
    return np.real_if_close(
        poses[0] @ expm(fraction * logm(np.linalg.solve(poses[0], poses[1])))
    )


def _moved(poses: np.ndarray, velocity: np.ndarray, time: float) -> np.ndarray:
    return np.array(
        [
            pose @ expm(time * _hat(v))
            for pose, v in zip(poses, velocity.reshape(2, 6), strict=True)
        ]
    )


def _point_maps(poses: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    point, step = _point(poses, fraction), 1e-6
    derivatives = []
    for axis in np.eye(12):
        plus = _point(_moved(poses, axis, step), fraction)
        minus = _point(_moved(poses, axis, -step), fraction)
        derivatives.append((plus - minus) / (2 * step))
    return point, np.array(derivatives)


def _samples(order: int = 4) -> tuple[InertiaSample, ...]:
    points, weights = np.polynomial.legendre.leggauss(order)
    return tuple(
        InertiaSample(
            float((point + 1) / 2),
            ComponentMassProperties(
                "synthetic-section",
                ComponentRole.SHAFT,
                "section-material",
                2.4 * weight / 2,
                (0.02, -0.01, 0),
                tuple(map(tuple, np.diag([0.04, 0.05, 0.09]) * weight / 2)),
            ),
        )
        for point, weight in zip(points, weights, strict=True)
    )


def test_velocity_map_matches_every_spatial_pose_derivative() -> None:
    poses = _poses()
    relative = np.array([0.04, -0.03, 1.0, 0.3, -0.2, 0.15])
    for fraction in (0.0, 0.23, 0.79, 1.0):
        point, derivatives = _point_maps(poses, fraction)
        material = np.array(
            [np.linalg.solve(point, derivative) for derivative in derivatives]
        )
        oracle = np.column_stack(
            [np.r_[value[:3, 3], _vee(value)] for value in material]
        )
        np.testing.assert_allclose(
            section_velocity_map(relative, fraction), oracle, rtol=2e-7, atol=3e-9
        )


def test_velocity_map_derivative_and_zero_rotation_limit() -> None:
    relative, step, fraction = np.array([0.04, -0.03, 1.0, 0.3, -0.2, 0.15]), 1e-5, 0.37
    for direction in np.eye(6):
        numerical = (
            section_velocity_map(relative + step * direction, fraction)
            - section_velocity_map(relative - step * direction, fraction)
        ) / (2 * step)
        np.testing.assert_allclose(
            section_velocity_map_derivative(relative, fraction, direction),
            numerical,
            rtol=2e-7,
            atol=2e-10,
        )
    direction = np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1])
    linear, angular = (
        _hat(np.r_[np.zeros(3), direction[:3]])[:3, :3],
        _hat(direction)[:3, :3],
    )
    bracket = np.block([[angular, linear], [np.zeros((3, 3)), angular]])
    right = fraction * (1 - fraction) * bracket / 2
    np.testing.assert_allclose(
        section_velocity_map_derivative(np.zeros(6), fraction, direction),
        np.hstack((-right, right)),
        atol=2e-16,
    )


def test_mass_matches_independent_com_and_spin_energy() -> None:
    poses, samples = _poses(), _samples()
    result = SectionInertia(samples).evaluate(poses, np.zeros(12))
    oracle = np.zeros((12, 12))
    for sample in samples:
        point, derivatives = _point_maps(poses, sample.fraction)
        body, rotation = sample.body, point[:3, :3]
        com_map = np.column_stack(
            [
                value[:3, 3] + value[:3, :3] @ body.center_of_mass_m
                for value in derivatives
            ]
        )
        spin_map = np.column_stack(
            [_vee(value[:3, :3] @ rotation.T) for value in derivatives]
        )
        inertia = rotation @ body.inertia_at_com_kg_m2 @ rotation.T
        oracle += body.mass_kg * com_map.T @ com_map + spin_map.T @ inertia @ spin_map
    np.testing.assert_allclose(result.mass, oracle, rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(result.bias, 0, atol=1e-15)
    assert np.min(np.linalg.eigvalsh(result.mass)) > 0


def test_inertia_bias_matches_com_acceleration_and_angular_momentum() -> None:
    poses, samples = _poses(), _samples(3)
    velocity = np.random.default_rng(437).normal(size=12) * 0.3
    result = SectionInertia(samples).evaluate(poses, velocity)
    oracle, step = np.zeros(12), 1e-4
    for sample in samples:
        point, derivatives = _point_maps(poses, sample.fraction)
        plus = _point(_moved(poses, velocity, step), sample.fraction)
        minus = _point(_moved(poses, velocity, -step), sample.fraction)
        rate, acceleration = (
            (plus - minus) / (2 * step),
            (plus - 2 * point + minus) / step**2,
        )
        body, rotation = sample.body, point[:3, :3]
        spin = _vee(rate[:3, :3] @ rotation.T)
        alpha_matrix = acceleration[:3, :3] @ rotation.T + rate[:3, :3] @ rate[:3, :3].T
        alpha = _vee((alpha_matrix - alpha_matrix.T) / 2)
        force = body.mass_kg * (
            acceleration[:3, 3] + acceleration[:3, :3] @ body.center_of_mass_m
        )
        inertia = rotation @ body.inertia_at_com_kg_m2 @ rotation.T
        moment = (
            np.cross(rotation @ body.center_of_mass_m, force)
            + inertia @ alpha
            + np.cross(spin, inertia @ spin)
        )
        for index, derivative in enumerate(derivatives):
            oracle[index] += force @ derivative[:3, 3] + moment @ _vee(
                derivative[:3, :3] @ rotation.T
            )
    np.testing.assert_allclose(result.bias, oracle, rtol=3e-5, atol=3e-7)


def test_mass_rate_energy_identity_observer_change_and_copy_isolation() -> None:
    poses, velocity = _poses(), np.random.default_rng(89).normal(size=12) * 0.2
    model, step = SectionInertia(_samples()), 1e-5
    result = model.evaluate(poses, velocity)
    plus = model.evaluate(_moved(poses, velocity, step), velocity)
    minus = model.evaluate(_moved(poses, velocity, -step), velocity)
    np.testing.assert_allclose(
        result.mass_rate, (plus.mass - minus.mass) / (2 * step), rtol=2e-7, atol=1e-9
    )
    assert velocity @ result.bias == pytest.approx(
        (plus.kinetic_energy_j - minus.kinetic_energy_j) / (2 * step),
        rel=2e-7,
        abs=1e-10,
    )
    observer = exp_twist([0.4, -0.2, 0.1, -0.2, 0.3, 0.1])
    transformed = model.evaluate(observer @ poses, velocity)
    np.testing.assert_allclose(transformed.mass, result.mass, atol=1e-13)
    np.testing.assert_allclose(transformed.bias, result.bias, atol=1e-13)
    result.mass[:] = 42
    assert not np.all(model.evaluate(poses, velocity).mass == 42)


def test_curved_configuration_quadrature_converges() -> None:
    poses, zero = _poses(), np.zeros(12)
    reference = SectionInertia(_samples(20)).evaluate(poses, zero).mass
    errors = [
        np.linalg.norm(
            SectionInertia(_samples(order)).evaluate(poses, zero).mass - reference
        )
        for order in (2, 4, 8)
    ]
    assert errors[1] < errors[0] * 1e-3
    assert errors[2] < errors[1] * 0.1


@pytest.mark.parametrize("fraction", [-0.1, 1.1, True, "0.5", float("nan")])
def test_invalid_material_fraction_is_rejected(fraction: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        InertiaSample(fraction, _samples()[0].body)
    with pytest.raises((TypeError, ValueError)):
        section_velocity_map(np.zeros(6), fraction)


def test_sample_and_motion_contracts() -> None:
    samples = _samples()
    with pytest.raises(ValueError, match="sample"):
        SectionInertia(())
    with pytest.raises(TypeError, match="body"):
        InertiaSample(0.5, None)
    with pytest.raises(ValueError, match="frame"):
        SectionInertia(
            (
                samples[0],
                replace(
                    samples[1], body=replace(samples[1].body, frame_id="different")
                ),
            )
        )
    model = SectionInertia(samples)
    with pytest.raises((TypeError, ValueError)):
        model.evaluate(_poses(), [True] * 12)
    with pytest.raises(ValueError, match="shape"):
        model.evaluate(np.eye(4), np.zeros(12))


@pytest.mark.parametrize(
    "relative",
    [
        [True] * 6,
        [0] * 5,
        [0, 0, 0, np.pi, 0, 0],
        [0, 0, np.inf, 0, 0, 0],
    ],
)
def test_velocity_mapping_rejects_invalid_relative_charts(relative: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        section_velocity_map(relative, 0.5)
    with pytest.raises((TypeError, ValueError)):
        section_velocity_map_derivative(relative, 0.5, np.zeros(6))


@pytest.mark.parametrize("direction", [["1"] * 6, [1j] * 6])
def test_velocity_derivative_rejects_coerced_directions(direction: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        section_velocity_map_derivative(np.zeros(6), 0.5, direction)


def test_lumped_quadrature_remains_singular_and_sample_inputs_are_copied() -> None:
    samples = list(_samples())
    model = SectionInertia(samples)
    samples.clear()
    assert len(model.samples) == 4
    single = SectionInertia((model.samples[0],))
    mass = single.evaluate(_poses(), np.zeros(12)).mass
    assert np.linalg.matrix_rank(mass, tol=1e-12) == 6
    # A mathematically valid lumped sample is not silently repaired into a full
    # distributed dynamic model or accompanied by a stability certificate.
