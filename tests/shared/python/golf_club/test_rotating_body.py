"""Energy-Hessian and inertial-frame checks of full rotating head transport."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from shared.python.golf_club.rotating_body import (
    RotatingFrameState,
    rotating_body_tangent,
)
from shared.python.golf_club.types import ComponentMassProperties, ComponentRole

pytestmark = [pytest.mark.unit, pytest.mark.contract, pytest.mark.scientific]


@pytest.fixture
def body() -> ComponentMassProperties:
    return ComponentMassProperties(
        "synthetic-head",
        ComponentRole.HEAD,
        "shaft",
        0.4,
        (0.03, -0.02, 0.04),
        ((0.2, 0.02, 0.01), (0.02, 0.3, 0.03), (0.01, 0.03, 0.4)),
    )


@pytest.fixture
def frame() -> RotatingFrameState:
    return RotatingFrameState("shaft", (0.7, -0.4, 0.3), (0.2, 0.1, -0.3), (1, -2, 0.5))


def _energy(
    body: ComponentMassProperties,
    omega: np.ndarray,
    reference: np.ndarray,
    state: np.ndarray,
) -> float:
    """Finite Exp(theta) orientation and independently differentiated rotation."""
    q, velocity = state[:6], state[6:]
    orientation = Rotation.from_rotvec(q[3:]).as_matrix()
    step = 1e-4
    plus = Rotation.from_rotvec(q[3:] + step * velocity[3:]).as_matrix()
    minus = Rotation.from_rotvec(q[3:] - step * velocity[3:]).as_matrix()
    relative_omega = Rotation.from_matrix(plus @ minus.T).as_rotvec() / (2 * step)
    offset = orientation @ body.center_of_mass_m
    com = reference + q[:3] + offset
    com_velocity = (
        velocity[:3] + np.cross(relative_omega, offset) + np.cross(omega, com)
    )
    absolute_omega = omega + relative_omega
    inertia = orientation @ body.inertia_at_com_kg_m2 @ orientation.T
    return float(
        (
            body.mass_kg * com_velocity @ com_velocity
            + absolute_omega @ inertia @ absolute_omega
        )
        / 2
    )


def _hessian(
    energy: Callable[[np.ndarray], float], size: int, step: float
) -> np.ndarray:
    basis = np.eye(size) * step
    return np.array(
        [
            [
                (
                    energy(left + right)
                    - energy(left - right)
                    - energy(-left + right)
                    + energy(-left - right)
                )
                / (4 * step**2)
                for right in basis
            ]
            for left in basis
        ]
    )


def test_full_head_tangent_matches_finite_orientation_energy(
    body: ComponentMassProperties,
    frame: RotatingFrameState,
) -> None:
    reference = np.array([0.2, -0.1, 1.1])
    result = rotating_body_tangent(body, frame, reference)
    omega = np.asarray(frame.angular_velocity_rad_s)
    hessian = _hessian(lambda state: _energy(body, omega, reference, state), 12, 1e-3)
    np.testing.assert_allclose(result.mass, hessian[6:, 6:], atol=3e-7, rtol=3e-5)
    np.testing.assert_allclose(
        result.velocity_position, hessian[6:, :6], atol=3e-7, rtol=3e-5
    )
    np.testing.assert_allclose(
        result.centrifugal_stiffness, -hessian[:6, :6], atol=3e-7, rtol=3e-5
    )
    alpha = np.asarray(frame.angular_acceleration_rad_s2)
    alpha_hessian = _hessian(
        lambda state: _energy(body, alpha, reference, state), 12, 1e-3
    )
    np.testing.assert_allclose(
        result.euler_stiffness, alpha_hessian[6:, :6], atol=3e-7, rtol=3e-5
    )


def test_origin_acceleration_potential_hessian(
    body: ComponentMassProperties,
    frame: RotatingFrameState,
) -> None:
    result = rotating_body_tangent(body, frame, (0, 0, 1))

    def potential(q: np.ndarray) -> float:
        offset = Rotation.from_rotvec(q[3:]).apply(body.center_of_mass_m)
        return body.mass_kg * np.dot(frame.origin_acceleration_m_s2, q[:3] + offset)

    expected = _hessian(potential, 6, 1e-3)
    np.testing.assert_allclose(
        result.acceleration_stiffness, expected, atol=3e-7, rtol=3e-5
    )


def test_equilibrium_force_contains_full_offset_and_rotary_terms(
    body: ComponentMassProperties,
    frame: RotatingFrameState,
) -> None:
    reference = np.array([0.2, 0.1, 1])
    result = rotating_body_tangent(body, frame, reference)
    omega, alpha = (
        np.array(frame.angular_velocity_rad_s),
        np.array(frame.angular_acceleration_rad_s2),
    )
    com = reference + body.center_of_mass_m
    acceleration = (
        np.array(frame.origin_acceleration_m_s2)
        + np.cross(alpha, com)
        + np.cross(omega, np.cross(omega, com))
    )
    force = -body.mass_kg * acceleration
    torque = (
        np.cross(body.center_of_mass_m, force)
        - np.asarray(body.inertia_at_com_kg_m2) @ alpha
        - np.cross(omega, np.asarray(body.inertia_at_com_kg_m2) @ omega)
    )
    np.testing.assert_allclose(result.equilibrium_force, np.r_[force, torque])


def test_gyroscopic_operator_is_skew_and_does_no_work(
    body: ComponentMassProperties,
    frame: RotatingFrameState,
) -> None:
    result = rotating_body_tangent(body, frame, (0.2, 0, 1))
    np.testing.assert_allclose(result.gyroscopic + result.gyroscopic.T, 0, atol=1e-14)
    for velocity in np.eye(6).tolist() + [[1, -2, 3, 4, -5, 6]]:
        assert np.asarray(velocity) @ result.gyroscopic @ velocity == pytest.approx(
            0, abs=1e-13
        )


def test_zero_motion_recovers_full_stationary_spatial_mass(
    body: ComponentMassProperties,
) -> None:
    frame = RotatingFrameState("shaft", (0, 0, 0), (0, 0, 0), (0, 0, 0))
    result = rotating_body_tangent(body, frame, (0, 0, 1))
    for name in (
        "gyroscopic",
        "centrifugal_stiffness",
        "euler_stiffness",
        "acceleration_stiffness",
        "equilibrium_force",
    ):
        np.testing.assert_array_equal(getattr(result, name), 0)
    velocity, omega = np.array([1, 2, 3]), np.array([4, -2, 1])
    com_velocity = velocity + np.cross(omega, body.center_of_mass_m)
    expected = (
        body.mass_kg * com_velocity @ com_velocity
        + omega @ body.inertia_at_com_kg_m2 @ omega
    ) / 2
    state = np.r_[velocity, omega]
    assert state @ result.mass @ state / 2 == pytest.approx(expected)


def test_spherical_com_rotation_has_no_centrifugal_orientation_stiffness(
    body: ComponentMassProperties,
) -> None:
    body = replace(
        body,
        center_of_mass_m=(0, 0, 0),
        inertia_at_com_kg_m2=tuple(map(tuple, np.eye(3) * 0.2)),
    )
    frame = RotatingFrameState("shaft", (0.2, -0.3, 0.4), (0, 0, 0), (0, 0, 0))
    result = rotating_body_tangent(body, frame, (0, 0, 0))
    np.testing.assert_allclose(result.centrifugal_stiffness[3:, 3:], 0, atol=1e-15)
    # Canonical small rotation coordinates give I*Omega-cross here, not 2I*Omega-cross.
    expected = 0.2 * np.cross(frame.angular_velocity_rad_s, np.eye(3)).T
    np.testing.assert_allclose(result.gyroscopic[3:, 3:], expected)


def test_accelerating_rotating_frame_agrees_with_free_inertial_motion(
    body: ComponentMassProperties,
) -> None:
    body = replace(body, center_of_mass_m=(0, 0, 0))
    reference, x0, inertial_velocity = (
        np.array([0.2, 0, 1]),
        np.array([0.1, -0.1, 1.1]),
        np.array([0.3, 0.2, -0.1]),
    )
    origin_acceleration = np.array([0.1, -0.2, 0.3])

    def rotation(time: float) -> np.ndarray:
        return Rotation.from_rotvec([0, 0.7 * time + 0.2 * time**2, 0]).as_matrix()

    def rate(time: float, state: np.ndarray) -> np.ndarray:
        frame = RotatingFrameState(
            "shaft",
            (0, 0.7 + 0.4 * time, 0),
            (0, 0.4, 0),
            rotation(time).T @ origin_acceleration,
        )
        result = rotating_body_tangent(body, frame, reference)
        stiffness = (
            result.centrifugal_stiffness
            + result.euler_stiffness
            + result.acceleration_stiffness
        )
        acceleration = np.linalg.solve(
            result.mass[:3, :3],
            result.equilibrium_force[:3]
            - result.gyroscopic[:3, :3] @ state[3:]
            - stiffness[:3, :3] @ state[:3],
        )
        return np.r_[state[3:], acceleration]

    initial = np.r_[x0 - reference, inertial_velocity - np.cross([0, 0.7, 0], x0)]
    solution = solve_ivp(
        rate, (0, 0.8), initial, method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.04
    )
    assert solution.success
    positions = np.array(
        [
            0.5 * origin_acceleration * t**2 + rotation(t) @ (reference + q)
            for t, q in zip(solution.t, solution.y[:3].T, strict=True)
        ]
    )
    np.testing.assert_allclose(
        positions, x0 + solution.t[:, None] * inertial_velocity, atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("value", [(1, 2), (1, 2, np.inf), (1, True, 0), (1, 2, "3")])
def test_frame_vector_contracts(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RotatingFrameState("shaft", value, (0, 0, 0), (0, 0, 0))


def test_frame_mismatch_and_overflow_are_refused(
    body: ComponentMassProperties, frame: RotatingFrameState
) -> None:
    with pytest.raises(ValueError, match="frame"):
        rotating_body_tangent(body, replace(frame, frame_id="other"), (0, 0, 1))
    with pytest.raises(ValueError, match="finite"):
        rotating_body_tangent(
            body, replace(frame, angular_velocity_rad_s=(1e308, 0, 0)), (0, 0, 1)
        )
