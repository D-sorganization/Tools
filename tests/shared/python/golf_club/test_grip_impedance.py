"""Power, passivity and frame benchmarks for a six-axis local grip port."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import trapezoid

from shared.python.golf_club.grip_impedance import (
    GripPortResponse,
    GripPortState,
    PassiveGripImpedance,
    evaluate_grip_impedance,
    grip_frequency_impedance,
    transform_grip_impedance,
    transform_grip_state,
)
from shared.python.golf_club.types import RigidTransform

pytestmark = [pytest.mark.unit, pytest.mark.contract, pytest.mark.scientific]


@pytest.fixture
def grip() -> PassiveGripImpedance:
    return PassiveGripImpedance(
        frame_id="grip",
        source_id="synthetic-passivity-benchmark",
        inertance_factor=np.diag([2, 3, 4, 0.2, 0.3, 0.4]),
        damping_factor=np.diag([5, 6, 7, 0.5, 0.6, 0.7]),
        stiffness_factor=np.diag([8, 9, 10, 0.8, 0.9, 1]),
    )


@pytest.fixture
def state() -> GripPortState:
    return GripPortState(
        "grip",
        (0.01, -0.02, 0.03, 0.001, -0.002, 0.003),
        (0.1, -0.2, 0.3, 0.01, -0.02, 0.03),
        (1, -2, 3, 0.1, -0.2, 0.3),
    )


def test_diagonal_mechanics_and_instantaneous_power(
    grip: PassiveGripImpedance, state: GripPortState
) -> None:
    response = evaluate_grip_impedance(grip, state)
    assert response.frame_id == state.frame_id
    assert response.source_id == grip.source_id
    mass = np.diag([4, 9, 16, 0.04, 0.09, 0.16])
    damping = np.diag([25, 36, 49, 0.25, 0.36, 0.49])
    stiffness = np.diag([64, 81, 100, 0.64, 0.81, 1])
    q, velocity, acceleration = map(
        np.asarray, (state.displacement, state.velocity, state.acceleration)
    )
    expected = -(mass @ acceleration + damping @ velocity + stiffness @ q)
    np.testing.assert_allclose(response.reaction_wrench, expected)
    assert response.inertial_energy_j == pytest.approx(velocity @ mass @ velocity / 2)
    assert response.elastic_energy_j == pytest.approx(q @ stiffness @ q / 2)
    assert response.dissipated_power_w == pytest.approx(velocity @ damping @ velocity)
    assert response.input_power_w == pytest.approx(-expected @ velocity)
    assert response.power_residual_w == pytest.approx(0, abs=1e-12)


def test_cross_coupling_and_rank_deficient_passivity(state: GripPortState) -> None:
    factor = np.zeros((6, 6))
    factor[0, [0, 5]] = [2, 3]
    zero = np.zeros((6, 6))
    grip = PassiveGripImpedance("grip", zero, factor, zero, "synthetic-coupling")
    response = evaluate_grip_impedance(grip, state)
    latent = 2 * state.velocity[0] + 3 * state.velocity[5]
    assert response.reaction_wrench[0] == pytest.approx(-2 * latent)
    assert response.reaction_wrench[5] == pytest.approx(-3 * latent)
    assert response.dissipated_power_w == pytest.approx(latent**2)
    assert response.inertial_energy_j == response.elastic_energy_j == 0


def test_zero_impedance_is_free_boundary(state: GripPortState) -> None:
    zero = np.zeros((6, 6))
    response = evaluate_grip_impedance(
        PassiveGripImpedance("grip", zero, zero, zero, "synthetic-free"), state
    )
    np.testing.assert_array_equal(response.reaction_wrench, np.zeros(6))
    assert response.input_power_w == response.dissipated_power_w == 0


def test_harmonic_impedance_real_part_is_damping(grip: PassiveGripImpedance) -> None:
    omega = 100.0
    impedance = grip_frequency_impedance(grip, omega)
    np.testing.assert_allclose(impedance.real, np.diag([25, 36, 49, 0.25, 0.36, 0.49]))
    expected_imaginary = (
        omega * np.diag([4, 9, 16, 0.04, 0.09, 0.16])
        - np.diag([64, 81, 100, 0.64, 0.81, 1]) / omega
    )
    np.testing.assert_allclose(impedance.imag, expected_imaginary)
    velocity = np.array([1 + 2j, 2 - 3j, -1j, 0.3j, 1, 0])
    assert np.real(np.vdot(velocity, impedance @ velocity)) >= 0


def test_frame_change_preserves_power_and_energy(
    grip: PassiveGripImpedance, state: GripPortState
) -> None:
    transform = RigidTransform(
        "grip", "club", ((0, -1, 0), (1, 0, 0), (0, 0, 1)), (0.2, -0.1, 0.4)
    )
    mapped_grip = transform_grip_impedance(grip, transform)
    mapped_state = transform_grip_state(state, transform)
    original = evaluate_grip_impedance(grip, state)
    mapped = evaluate_grip_impedance(mapped_grip, mapped_state)
    rotation = np.asarray(transform.rotation)
    force = rotation @ np.asarray(original.reaction_wrench[:3])
    torque = rotation @ np.asarray(original.reaction_wrench[3:]) + np.cross(
        transform.translation_m, force
    )
    np.testing.assert_allclose(mapped.reaction_wrench, np.r_[force, torque], atol=1e-12)
    for name in (
        "input_power_w",
        "dissipated_power_w",
        "elastic_energy_j",
        "inertial_energy_j",
        "stored_energy_rate_w",
    ):
        assert getattr(mapped, name) == pytest.approx(
            getattr(original, name), rel=1e-12
        )
    assert mapped_grip.frame_id == mapped_state.frame_id == "club"
    assert mapped_grip.source_id == grip.source_id


def test_closed_cycle_work_equals_damping_loss(grip: PassiveGripImpedance) -> None:
    # Exact periodic trapezoid integration for a trigonometric polynomial.
    times = np.linspace(0, 2 * np.pi, 257)
    work = []
    dissipation = []
    for time in times:
        state = GripPortState(
            "grip",
            (np.sin(time), 0, 0, 0, 0, 0),
            (np.cos(time), 0, 0, 0, 0, 0),
            (-np.sin(time), 0, 0, 0, 0, 0),
        )
        response = evaluate_grip_impedance(grip, state)
        work.append(response.input_power_w)
        dissipation.append(response.dissipated_power_w)
    assert trapezoid(work, times) == pytest.approx(25 * np.pi, rel=1e-12)
    assert trapezoid(work, times) == pytest.approx(
        trapezoid(dissipation, times), rel=1e-12
    )


def test_caller_storage_is_not_retained(grip: PassiveGripImpedance) -> None:
    factor = np.eye(6)
    updated = replace(grip, damping_factor=factor)
    factor[:] = 100
    np.testing.assert_array_equal(updated.damping_factor, np.eye(6))
    with pytest.raises(TypeError):
        updated.damping_factor[0][0] = 2  # type: ignore[index]


@pytest.mark.parametrize("factor", [np.ones((3, 3)), np.full((6, 6), np.inf)])
def test_invalid_factors_are_refused(
    grip: PassiveGripImpedance, factor: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        replace(grip, damping_factor=factor)


@pytest.mark.parametrize(
    "factor", [np.ones((6, 6), dtype=bool), [["1"] * 6] * 6, np.eye(6, dtype=complex)]
)
def test_nonreal_factors_are_refused(
    grip: PassiveGripImpedance, factor: object
) -> None:
    with pytest.raises(TypeError):
        replace(grip, damping_factor=factor)


def test_mismatched_frames_are_refused(
    grip: PassiveGripImpedance, state: GripPortState
) -> None:
    with pytest.raises(ValueError, match="frame"):
        evaluate_grip_impedance(grip, replace(state, frame_id="other"))
    with pytest.raises(ValueError, match="frame"):
        transform_grip_impedance(grip, RigidTransform("other", "club"))
    with pytest.raises(ValueError, match="frame"):
        transform_grip_state(state, RigidTransform("other", "club"))


@pytest.mark.parametrize("omega", [0, -1, np.nan, np.inf])
def test_invalid_frequency_is_refused(grip: PassiveGripImpedance, omega: float) -> None:
    with pytest.raises(ValueError):
        grip_frequency_impedance(grip, omega)


def test_overflow_does_not_return_invalid_physics(
    grip: PassiveGripImpedance, state: GripPortState
) -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_grip_impedance(
            replace(grip, stiffness_factor=np.eye(6) * 1e200), state
        )


def test_mixed_boolean_factor_is_not_silently_numeric(
    grip: PassiveGripImpedance,
) -> None:
    mixed = np.eye(6).tolist()
    mixed[0][1] = True
    with pytest.raises(TypeError, match="boolean"):
        replace(grip, damping_factor=mixed)


@pytest.mark.parametrize(
    "field", ["inertial_energy_j", "elastic_energy_j", "dissipated_power_w"]
)
def test_response_cannot_claim_negative_stored_energy_or_loss(
    grip: PassiveGripImpedance, state: GripPortState, field: str
) -> None:
    response = evaluate_grip_impedance(grip, state)
    with pytest.raises(ValueError, match=field):
        replace(response, **{field: -1.0})


def test_response_cannot_claim_nonfinite_power(
    grip: PassiveGripImpedance, state: GripPortState
) -> None:
    response = evaluate_grip_impedance(grip, state)
    assert isinstance(response, GripPortResponse)
    with pytest.raises(ValueError, match="input_power"):
        replace(response, input_power_w=np.inf)
