"""Independent Newton/Euler oracles for deformed-section frame transport."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club._rotating_body_contracts import RotatingFrameState
from shared.python.golf_club._shaft_frame_inertia import rotating_section_inertia
from shared.python.golf_club._shaft_inertia import SectionInertia
from shared.python.golf_club._shaft_se3 import twist_ad

from .test_shaft_inertia import _moved, _point, _point_maps, _poses, _samples


def _frame() -> RotatingFrameState:
    return RotatingFrameState(
        "observer", (0.7, -0.4, 1.3), (0.2, 0.3, -0.1), (2, -3, 1)
    )


def _physical_oracle(poses: np.ndarray, frame: RotatingFrameState) -> np.ndarray:
    result = np.zeros(12)
    omega, alpha = (
        np.array(frame.angular_velocity_rad_s),
        frame.angular_acceleration_rad_s2,
    )
    for sample in _samples():
        pose, derivatives = _point_maps(poses, sample.fraction)
        rotation, body = pose[:3, :3], sample.body
        offset = rotation @ body.center_of_mass_m
        com = pose[:3, 3] + offset
        inertia = rotation @ body.inertia_at_com_kg_m2 @ rotation.T
        force = body.mass_kg * (
            frame.origin_acceleration_m_s2
            + np.cross(alpha, com)
            + np.cross(omega, np.cross(omega, com))
        )
        torque = (
            np.cross(offset, force) + inertia @ alpha + np.cross(omega, inertia @ omega)
        )
        for column, derivative in enumerate(derivatives):
            spin = derivative[:3, :3] @ rotation.T
            angular = spin[[2, 0, 1], [1, 2, 0]]
            result[column] += force @ derivative[:3, 3] + torque @ angular
    return result


@pytest.mark.parametrize("part", ["all", "omega", "alpha", "origin"])
def test_residual_matches_independent_com_force_and_spin_torque(part: str) -> None:
    frame = _frame()
    if part != "all":
        frame = RotatingFrameState(
            "observer",
            frame.angular_velocity_rad_s if part == "omega" else (0, 0, 0),
            frame.angular_acceleration_rad_s2 if part == "alpha" else (0, 0, 0),
            frame.origin_acceleration_m_s2 if part == "origin" else (0, 0, 0),
        )
    poses = _poses()
    result = rotating_section_inertia(SectionInertia(_samples()), poses, frame)
    np.testing.assert_allclose(
        result.residual, _physical_oracle(poses, frame), atol=2e-8
    )


def test_all_moving_material_jacobian_columns_match_residual_differences() -> None:
    section, poses, frame, step = SectionInertia(_samples()), _poses(), _frame(), 2e-5
    result = rotating_section_inertia(section, poses, frame)
    columns = []
    for axis in np.eye(12):
        plus = rotating_section_inertia(section, _moved(poses, axis, step), frame)
        minus = rotating_section_inertia(section, _moved(poses, axis, -step), frame)
        columns.append((plus.residual - minus.residual) / (2 * step))
    np.testing.assert_allclose(
        result.moving_jacobian, np.column_stack(columns), atol=3e-8
    )


def _velocity_oracle(poses: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Coriolis COM force and full angular-momentum derivative, in observer axes."""
    result, omega = np.zeros(12), np.array(_frame().angular_velocity_rad_s)
    for sample in _samples():
        pose, derivatives = _point_maps(poses, sample.fraction)
        derivative = np.einsum("i,ijk->jk", motion, derivatives)
        rotation, body = pose[:3, :3], sample.body
        offset = rotation @ body.center_of_mass_m
        inertia = rotation @ body.inertia_at_com_kg_m2 @ rotation.T
        spin = derivative[:3, :3] @ rotation.T
        angular = spin[[2, 0, 1], [1, 2, 0]]
        relative_com_velocity = derivative[:3, 3] + np.cross(angular, offset)
        force = 2 * body.mass_kg * np.cross(omega, relative_com_velocity)
        torque = (
            np.cross(offset, force)
            + inertia @ np.cross(omega, angular)
            + np.cross(angular, inertia @ omega)
            + np.cross(omega, inertia @ angular)
        )
        for column, basis in enumerate(derivatives):
            spin = basis[:3, :3] @ rotation.T
            result[column] += force @ basis[:3, 3] + torque @ spin[[2, 0, 1], [1, 2, 0]]
    return result


def test_gyroscopic_force_matches_newton_euler_and_has_zero_power() -> None:
    poses = _poses()
    result = rotating_section_inertia(SectionInertia(_samples()), poses, _frame())
    direction = np.linspace(-0.4, 0.7, 12)
    np.testing.assert_allclose(
        result.gyroscopic @ direction, _velocity_oracle(poses, direction), atol=3e-8
    )
    np.testing.assert_allclose(result.gyroscopic + result.gyroscopic.T, 0, atol=2e-14)
    assert abs(direction @ result.gyroscopic @ direction) < 2e-14


def test_stationary_frame_zero_terms_and_output_copy_isolation() -> None:
    section, poses = SectionInertia(_samples()), _poses()
    frame = RotatingFrameState("observer", (0, 0, 0), (0, 0, 0), (0, 0, 0))
    result = rotating_section_inertia(section, poses, frame)
    for value in (result.residual, result.moving_jacobian, result.gyroscopic):
        np.testing.assert_array_equal(value, 0)
        value[:] = 100
    fresh = rotating_section_inertia(section, poses, frame)
    np.testing.assert_array_equal(fresh.residual, 0)
    np.testing.assert_array_equal(fresh.moving_jacobian, 0)
    np.testing.assert_array_equal(fresh.gyroscopic, 0)


def test_frame_and_pose_contracts_reject_invalid_inputs() -> None:
    section, poses = SectionInertia(_samples()), _poses()
    with pytest.raises(TypeError, match="SectionInertia"):
        rotating_section_inertia(None, poses, _frame())
    with pytest.raises(TypeError, match="RotatingFrameState"):
        rotating_section_inertia(section, poses, None)
    poses[0, 0, 0] = 3
    with pytest.raises(ValueError):
        rotating_section_inertia(section, poses, _frame())
    with pytest.raises(TypeError, match="booleans"):
        replace(_frame(), angular_velocity_rad_s=(False, 0, 0))


def test_rotated_and_translated_observer_preserves_material_results() -> None:
    section, poses, frame = SectionInertia(_samples()), _poses(), _frame()
    change = poses[0].copy()
    rotation, shift = change[:3, :3], change[:3, 3]
    omega, alpha = frame.angular_velocity_rad_s, frame.angular_acceleration_rad_s2
    changed_frame = RotatingFrameState(
        "other-observer",
        rotation.T @ omega,
        rotation.T @ alpha,
        rotation.T
        @ (
            frame.origin_acceleration_m_s2
            + np.cross(alpha, shift)
            + np.cross(omega, np.cross(omega, shift))
        ),
    )
    changed_poses = np.array([np.linalg.solve(change, pose) for pose in poses])
    original = rotating_section_inertia(section, poses, frame)
    changed = rotating_section_inertia(section, changed_poses, changed_frame)
    for name in ("residual", "moving_jacobian", "gyroscopic"):
        np.testing.assert_allclose(
            getattr(original, name), getattr(changed, name), atol=2e-14
        )


def _potential(poses: np.ndarray, frame: RotatingFrameState) -> float:
    total, omega = 0.0, np.asarray(frame.angular_velocity_rad_s)
    for sample in _samples():
        pose, body = _point(poses, sample.fraction), sample.body
        rotation = pose[:3, :3]
        com = pose[:3, 3] + rotation @ body.center_of_mass_m
        inertia = rotation @ body.inertia_at_com_kg_m2 @ rotation.T
        speed = np.cross(omega, com)
        total += body.mass_kg * (
            frame.origin_acceleration_m_s2 @ com - speed @ speed / 2
        )
        total -= omega @ inertia @ omega / 2
    return float(total)


def test_conservative_frame_has_energy_curvature_and_euler_term_retains_curl() -> None:
    section, poses = SectionInertia(_samples()), _poses()
    frame = replace(_frame(), angular_acceleration_rad_s2=(0, 0, 0))
    result = rotating_section_inertia(section, poses, frame)
    step = 1e-4
    for axis in (np.linspace(-0.3, 0.5, 12), np.cos(np.arange(12))):
        curvature = (
            _potential(_moved(poses, axis, step), frame)
            - 2 * _potential(poses, frame)
            + _potential(_moved(poses, axis, -step), frame)
        ) / step**2
        assert axis @ result.moving_jacobian @ axis == pytest.approx(
            curvature, abs=2e-6
        )
    for sample_frame, conservative in ((frame, True), (_frame(), False)):
        result = rotating_section_inertia(section, poses, sample_frame)
        chart = result.moving_jacobian.copy()
        for index, axis in enumerate(np.eye(12)):
            correction = np.concatenate(
                [
                    twist_ad(direction).T @ force / 2
                    for direction, force in zip(
                        axis.reshape(2, 6), result.residual.reshape(2, 6), strict=True
                    )
                ]
            )
            chart[:, index] -= correction
        if conservative:
            np.testing.assert_allclose(chart, chart.T, atol=2e-14)
        else:
            assert np.linalg.norm(chart - chart.T) > 0.1


def test_frame_residual_agrees_with_absolute_section_kinetic_balance() -> None:
    section, poses, frame = SectionInertia(_samples()), _poses(), _frame()
    omega, alpha = frame.angular_velocity_rad_s, frame.angular_acceleration_rad_s2
    velocities, rates = [], []
    for pose in poses:
        rotation, position = pose[:3, :3], pose[:3, 3]
        velocities.append(
            np.r_[rotation.T @ np.cross(omega, position), rotation.T @ omega]
        )
        rates.append(
            np.r_[
                rotation.T
                @ (frame.origin_acceleration_m_s2 + np.cross(alpha, position)),
                rotation.T @ alpha,
            ]
        )
    kinetics = section.evaluate(poses, np.concatenate(velocities))
    balance = kinetics.mass @ np.concatenate(rates) + kinetics.bias
    result = rotating_section_inertia(section, poses, frame)
    np.testing.assert_allclose(result.residual, balance, atol=2e-14)
