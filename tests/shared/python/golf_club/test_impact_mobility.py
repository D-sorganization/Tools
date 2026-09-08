"""Independent impulse mechanics gates for IA-T1 (#5069), written first."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from shared.python.golf_club.impact_mobility import (
    RigidContactBody,
    contact_inverse_mass,
    normal_effective_mass,
    normal_impulse,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _head(offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> RigidContactBody:
    return RigidContactBody(0.2, np.diag([0.0003, 0.0004, 0.0005]), offset)


def test_centered_contact_recovers_translational_mass() -> None:
    np.testing.assert_allclose(contact_inverse_mass(_head()), np.eye(3) / 0.2)
    assert normal_effective_mass(_head(), (1, 0, 0)) == pytest.approx(0.2)


def test_offset_principal_axis_limit_and_axial_offset_invariance() -> None:
    expected = 1 / (1 / 0.2 + 0.03**2 / 0.0005)
    assert normal_effective_mass(_head((0, 0.03, 0)), (1, 0, 0)) == pytest.approx(
        expected
    )
    assert normal_effective_mass(_head((0.08, 0.03, 0)), (1, 0, 0)) == pytest.approx(
        expected
    )


def test_tensor_response_matches_direct_linear_and_angular_impulse() -> None:
    inertia = np.array([[4, 1, 0], [1, 5, 1], [0, 1, 6]]) * 1e-4
    body = RigidContactBody(0.21, inertia, (0.02, -0.03, 0.01))
    impulse = np.array([1.1, -0.4, 0.7])
    offset = np.array(body.contact_offset_m)
    angular_jump = np.linalg.solve(inertia, np.cross(offset, impulse))
    direct = impulse / body.mass_kg + np.cross(angular_jump, offset)
    mobility = contact_inverse_mass(body)
    np.testing.assert_allclose(mobility @ impulse, direct)
    np.testing.assert_allclose(mobility, mobility.T, atol=1e-12)
    assert np.linalg.eigvalsh(mobility).min() > 0


def test_simultaneous_frame_rotation_preserves_normal_mass() -> None:
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    body = _head((0.02, 0.03, -0.01))
    rotated = RigidContactBody(
        body.mass_kg,
        rotation @ np.array(body.inertia_at_com_kg_m2) @ rotation.T,
        rotation @ body.contact_offset_m,
    )
    np.testing.assert_allclose(
        contact_inverse_mass(rotated),
        rotation @ contact_inverse_mass(body) @ rotation.T,
    )
    assert normal_effective_mass(body, (1, 0, 0)) == pytest.approx(
        normal_effective_mass(rotated, rotation @ np.array([1, 0, 0]))
    )


@pytest.mark.parametrize("restitution", [0.0, 0.4, 1.0])
def test_normal_collision_conserves_momenta_and_has_exact_energy_loss(
    restitution: float,
) -> None:
    head = _head((0, 0.03, 0))
    normal = np.array([1.0, 0, 0])
    head_mass = normal_effective_mass(head, normal)
    ball_mass, approach = 0.0459, 40.0
    impulse = normal_impulse(approach, head_mass, ball_mass, restitution)
    head_velocity = np.array([approach - impulse / head.mass_kg, 0, 0])
    ball_velocity = normal * impulse / ball_mass
    offset = np.array(head.contact_offset_m)
    inertia = np.array(head.inertia_at_com_kg_m2)
    head_spin = np.linalg.solve(inertia, np.cross(offset, -impulse * normal))
    np.testing.assert_allclose(
        head.mass_kg * head_velocity + ball_mass * ball_velocity,
        head.mass_kg * approach * normal,
    )
    np.testing.assert_allclose(
        inertia @ head_spin + np.cross(offset, ball_mass * ball_velocity),
        0,
        atol=1e-12,
    )
    separating_speed = normal @ (
        ball_velocity - head_velocity - np.cross(head_spin, offset)
    )
    assert separating_speed == pytest.approx(restitution * approach, abs=1e-12)
    initial = 0.5 * head.mass_kg * approach**2
    final = (
        0.5 * head.mass_kg * (head_velocity @ head_velocity)
        + 0.5 * ball_mass * (ball_velocity @ ball_velocity)
        + 0.5 * head_spin @ inertia @ head_spin
    )
    reduced_mass = 1 / (1 / head_mass + 1 / ball_mass)
    assert initial - final == pytest.approx(
        0.5 * reduced_mass * (1 - restitution**2) * approach**2, abs=1e-12
    )


@pytest.mark.parametrize("approach", [-4.0, 0.0])
def test_separating_contact_has_no_impulse(approach: float) -> None:
    assert normal_impulse(approach, 0.2, 0.0459, 0.8) == 0


def test_body_copies_mutable_input() -> None:
    inertia, offset = np.eye(3), np.zeros(3)
    body = RigidContactBody(1, inertia, offset)
    inertia[:] = 0
    offset[:] = 10
    np.testing.assert_allclose(contact_inverse_mass(body), np.eye(3))


@pytest.mark.parametrize("normal", [(0, 0, 0), (2, 0, 0), (1, 2), (np.nan, 0, 0)])
def test_invalid_normal_is_refused(normal: Any) -> None:
    with pytest.raises(ValueError):
        normal_effective_mass(_head(), normal)


@pytest.mark.parametrize(
    "inertia",
    [
        np.zeros((3, 3)),
        np.diag([1, 1, 3]),
        np.diag([-1, 1, 1]),
        [[1, 2, 0], [0, 1, 0], [0, 0, 1]],
    ],
)
def test_invalid_inertia_is_refused(inertia: Any) -> None:
    with pytest.raises(ValueError):
        replace(_head(), inertia_at_com_kg_m2=inertia)


@pytest.mark.parametrize(
    "mass,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
        ("1", TypeError),
    ],
)
def test_invalid_mass_is_refused(mass: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        replace(_head(), mass_kg=mass)


@pytest.mark.parametrize(
    "args,error",
    [
        ((np.nan, 1, 1, 0.8), ValueError),
        ((1, 0, 1, 0.8), ValueError),
        ((1, 1, -1, 0.8), ValueError),
        ((1, 1, 1, 1.1), ValueError),
        ((1, 1, 1, -0.1), ValueError),
        ((True, 1, 1, 0.8), TypeError),
    ],
)
def test_invalid_impulse_inputs_are_refused(
    args: tuple[Any, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        normal_impulse(*args)


def test_wrong_body_type_is_refused() -> None:
    with pytest.raises(TypeError):
        contact_inverse_mass(None)
