"""Independent Newton/Euler, energy and observer-momentum controls."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from shared.python.golf_club._shaft_spectrum import SpectrumScales
from shared.python.swing_sim.impact._rigid_body_response import RigidBodyInertia


def _body() -> RigidBodyInertia:
    rotation = Rotation.from_rotvec([0.4, -0.2, 0.3]).as_matrix()
    return RigidBodyInertia(
        "ball-material",
        0.046,
        [0.003, -0.002, 0.001],
        rotation @ np.diag([7e-6, 8e-6, 9e-6]) @ rotation.T,
    )


def _scales() -> SpectrumScales:
    return SpectrumScales(0.02, 0.001, 1e-12, 1e-10)


def _energy(body: RigidBodyInertia, twist: np.ndarray) -> float:
    com_velocity = twist[:3] + np.cross(twist[3:], body.center_of_mass_m)
    return float(
        0.5 * body.mass_kg * (com_velocity @ com_velocity)
        + 0.5 * twist[3:] @ body.inertia_at_com_kg_m2 @ twist[3:]
    )


def _momentum(
    body: RigidBodyInertia, pose: np.ndarray, twist: np.ndarray
) -> np.ndarray:
    rotation = pose[:3, :3]
    com = pose[:3, 3] + rotation @ body.center_of_mass_m
    linear = (
        body.mass_kg
        * rotation
        @ (twist[:3] + np.cross(twist[3:], body.center_of_mass_m))
    )
    angular = rotation @ body.inertia_at_com_kg_m2 @ twist[3:] + np.cross(com, linear)
    return np.r_[linear, angular]


def test_full_tensor_euler_and_eccentric_com_newton_acceleration() -> None:
    body = _body()
    twist = np.array([3, -2, 1, 35, -21, 17], dtype=float)
    wrench = np.array([120, -30, 15, 0.1, 0.2, -0.3])
    response = body.response(twist, wrench, "ball-material", _scales())
    rates = np.asarray(response.twist_rate)
    omega, offset = twist[3:], np.asarray(body.center_of_mass_m)
    inertia = np.asarray(body.inertia_at_com_kg_m2)
    angular = np.linalg.solve(
        inertia,
        wrench[3:] - np.cross(offset, wrench[:3]) - np.cross(omega, inertia @ omega),
    )
    acceleration_com = (
        rates[:3]
        + np.cross(omega, twist[:3])
        + np.cross(rates[3:], offset)
        + np.cross(omega, np.cross(omega, offset))
    )
    np.testing.assert_allclose(rates[3:], angular, rtol=2e-13, atol=1e-9)
    np.testing.assert_allclose(body.mass_kg * acceleration_com, wrench[:3], atol=1e-12)
    assert response.kinetic_energy_j == pytest.approx(_energy(body, twist), rel=1e-14)
    assert response.power_w == pytest.approx(twist @ wrench, rel=1e-14)
    assert response.relative_residual < 1e-13


@pytest.mark.parametrize("wrench", [np.zeros(6), [8, -2, 3, 0.1, -0.04, 0.07]])
def test_observer_momentum_derivative_and_energy_from_independent_differences(
    wrench: object,
) -> None:
    body, twist = _body(), np.array([2, -1, 3, 17, -9, 13], dtype=float)
    load = np.asarray(wrench)
    response = body.response(twist, load, "ball-material", _scales())
    rate = np.asarray(response.twist_rate)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([-0.2, 0.5, 0.3]).as_matrix()
    pose[:3, 3] = [0.3, -0.7, 1.1]
    generator = np.zeros((4, 4))
    generator[:3, :3] = np.cross(twist[3:], np.eye(3)).T
    generator[:3, 3] = twist[:3]
    force = pose[:3, :3] @ load[:3]
    torque = pose[:3, :3] @ load[3:] + np.cross(pose[:3, 3], force)
    for step in (2e-7, 1e-7):
        plus = _momentum(body, pose @ expm(step * generator), twist + step * rate)
        minus = _momentum(body, pose @ expm(-step * generator), twist - step * rate)
        np.testing.assert_allclose(
            (plus - minus) / (2 * step), np.r_[force, torque], atol=2e-8, rtol=2e-8
        )
        energy_rate = (
            _energy(body, twist + step * rate) - _energy(body, twist - step * rate)
        ) / (2 * step)
        assert energy_rate == pytest.approx(response.power_w, abs=2e-8)


def test_material_axis_relabeling_preserves_response_and_energy() -> None:
    body, twist = _body(), np.array([1, 2, 3, 4, 5, 6], dtype=float)
    wrench = np.array([8, -2, 3, 0.1, 0.2, -0.1])
    rotation = Rotation.from_rotvec([0.6, 0.2, -0.7]).as_matrix()
    mapping = np.kron(np.eye(2), rotation)
    relabeled = replace(
        body,
        material_frame_id="rotated",
        center_of_mass_m=rotation @ body.center_of_mass_m,
        inertia_at_com_kg_m2=rotation @ body.inertia_at_com_kg_m2 @ rotation.T,
    )
    before = body.response(twist, wrench, "ball-material", _scales())
    after = relabeled.response(mapping @ twist, mapping @ wrench, "rotated", _scales())
    np.testing.assert_allclose(
        after.twist_rate, mapping @ before.twist_rate, rtol=2e-13
    )
    assert after.kinetic_energy_j == pytest.approx(before.kinetic_energy_j)
    assert after.power_w == pytest.approx(before.power_w)


def test_com_applied_force_does_not_create_spurious_spin_acceleration() -> None:
    body, force = _body(), np.array([4, -3, 7])
    load = np.r_[force, np.cross(body.center_of_mass_m, force)]
    response = body.response(np.zeros(6), load, "ball-material", _scales())
    np.testing.assert_allclose(response.twist_rate[3:], 0, atol=1e-11)
    np.testing.assert_allclose(response.twist_rate[:3], force / body.mass_kg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mass_kg", 0),
        ("mass_kg", -1),
        ("mass_kg", True),
        ("mass_kg", np.inf),
        ("material_frame_id", " "),
        ("center_of_mass_m", [0, np.nan, 0]),
        ("inertia_at_com_kg_m2", np.diag([-1e-5, 2e-5, 2e-5])),
        ("inertia_at_com_kg_m2", np.diag([1e-5, 1e-5, 3e-5])),
        ("inertia_at_com_kg_m2", [[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
    ],
)
def test_invalid_mass_properties_are_refused(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_body(), **{field: value})


@pytest.mark.parametrize("inertia", [np.zeros((3, 3)), np.diag([1e-20, 1, 1])])
def test_unresolved_or_singular_inertia_is_not_regularized(inertia: object) -> None:
    body = replace(_body(), inertia_at_com_kg_m2=inertia)
    with pytest.raises(ValueError, match="positive definite|resolved"):
        body.response(np.zeros(6), np.zeros(6), "ball-material", _scales())


def test_response_refuses_frame_mismatch_and_malformed_vectors() -> None:
    body = _body()
    with pytest.raises(ValueError, match="frame"):
        body.response(np.zeros(6), np.zeros(6), "other", _scales())
    for twist, load in [(np.zeros(5), np.zeros(6)), (np.zeros(6), [np.nan] * 6)]:
        with pytest.raises(ValueError):
            body.response(twist, load, "ball-material", _scales())


def test_mass_properties_and_response_own_immutable_storage() -> None:
    offset, inertia = np.array([0.001, 0, 0]), np.eye(3) * 8e-6
    body = RigidBodyInertia("material", 0.046, offset, inertia)
    offset[:] = 99
    inertia[:] = 0
    assert body.center_of_mass_m == (0.001, 0, 0)
    assert body.inertia_at_com_kg_m2[0][0] == 8e-6
    response = body.response(np.zeros(6), np.zeros(6), "material", _scales())
    assert isinstance(response.twist_rate, tuple)
    with pytest.raises(FrozenInstanceError):
        response.power_w = 1


@pytest.mark.parametrize("scale", [1e-18, 1e-9, 1.0])
@pytest.mark.parametrize(
    "tensor",
    [
        np.diag([1.0, 1.0, 3.0]),
        np.array([[2.0, 0.1, 0], [0, 2.0, 0], [0, 0, 2.0]]),
    ],
)
def test_inertia_domain_is_checked_relative_to_its_own_units(
    scale: float,
    tensor: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="triangle|symmetric"):
        replace(_body(), inertia_at_com_kg_m2=scale * tensor)
