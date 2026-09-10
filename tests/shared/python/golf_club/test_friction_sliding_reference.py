"""Independent saturated sliding force, impulse, spin and plastic-work checks."""

import json
from collections.abc import Callable

import numpy as np
import pytest

from shared.python.swing_sim.impact import _friction_contact_step as stepper
from shared.python.swing_sim.impact._friction_contact_trajectory import (
    integrate_friction_contact,
)
from shared.python.swing_sim.impact._spatial_contact_kinematics import (
    ContactBodyState,
    PlaneSphereGeometry,
    PlaneSphereKinematics,
)

from ._friction_sliding_reference import (
    gap_acceleration,
    sliding_case,
    sliding_observables,
)
from .test_friction_contact_trajectory import _controls
from .test_friction_continuous_reference import _scaled_output


@pytest.mark.parametrize("spin", [0.0, 2.0, -3.0])
def test_gap_acceleration_retains_rotating_plane_and_material_velocity_terms(
    spin: float,
) -> None:
    face_pose, ball_pose = np.eye(4), np.eye(4)
    ball_pose[:3, 3] = (0.7, 0, 1.3)
    face_velocity, ball_velocity = np.array([0.2, 0, -0.1]), np.array([0.8, 0, 0.4])
    face_spin, ball_spin = np.array([0, spin, 0]), np.array([0.2, 0.1, 0.3])
    face = ContactBodyState(face_pose, np.r_[face_velocity, face_spin], "observer")
    ball = ContactBodyState(ball_pose, np.r_[ball_velocity, ball_spin], "observer")
    contact = PlaneSphereKinematics(
        face, ball, PlaneSphereGeometry(0.02, (0.1, 0.2, 0.3), (0, 0, 1))
    )
    face_rate = np.r_[
        np.array([0.3, 0, -0.4]) - np.cross(face_spin, face_velocity), [0, 1.2, 0]
    ]
    ball_rate = np.r_[
        np.array([-0.2, 0, 0.5]) - np.cross(ball_spin, ball_velocity), [0, 0, 0]
    ]
    # Expanded world formula: n'' . d + 2 n' . d' + n . d''.
    expected = 1.2 * 0.7 - spin**2 * 1.3 + 2 * spin * 0.6 + 0.9
    assert gap_acceleration(contact, face_rate, ball_rate) == pytest.approx(
        expected, abs=2e-14
    )


@pytest.mark.parametrize("ball_velocity_mps", [-0.4, 0.4])
def test_saturated_sliding_refines_each_force_spin_impulse_and_work_channel(
    ball_velocity_mps: float,
    record_property: Callable[[str, object], None],
) -> None:
    reference = sliding_case(ball_velocity_mps)
    end = _controls().bounds_s[1]
    state, response, integrals = reference.integrate(end, 1e-11, 1e-13)
    expected = sliding_observables(state, response, integrals)
    fine_state, fine_response, fine_integrals = reference.integrate(end, 1e-12, 1e-14)
    tighter = sliding_observables(fine_state, fine_response, fine_integrals)
    np.testing.assert_allclose(expected, tighter, atol=1e-10, rtol=0)
    errors, state_errors, losses = [], [], []
    for steps in (4, 8, 16):
        last = integrate_friction_contact(
            reference.problem, reference.initial, _controls(steps)
        ).samples[-1]
        work = np.r_[
            last.work.values,
            last.normal_impulse_ns,
            last.tangential_impulse_ns,
            last.plastic_dissipation_j,
        ]
        actual = sliding_observables(last.state, last.response, work)
        errors.append(np.abs(actual - tighter))
        state_errors.append(
            np.linalg.norm(_scaled_output(last.state) - _scaled_output(fine_state))
        )
        losses.append(last.tangential_algorithmic_loss_j)
    error_matrix = np.asarray(errors)
    # Preset SI bounds: Fn, Ft [N], vx [m/s], wy [rad/s], Jn/Jtx [N s], Dp [J].
    limits = np.array([1e-2, 1e-3, 1e-4, 1e-2, 1e-6, 1e-7, 1e-6])
    np.testing.assert_array_less(error_matrix[-1, :7], limits)
    ratios = error_matrix[:-1, :7] / error_matrix[1:, :7]
    np.testing.assert_array_less(1.7, ratios)
    np.testing.assert_array_less(ratios, 2.3)
    np.testing.assert_array_less(error_matrix[-1, 7:], 1e-6)  # five work ports [J]
    assert state_errors[2] < state_errors[1] < state_errors[0] < 1e-3
    assert 0 < losses[2] < losses[1] < losses[0]
    record_property("sliding_channel_errors", json.dumps(error_matrix.tolist()))
    record_property("sliding_reference_outputs", json.dumps(tighter.tolist()))
    record_property(
        "sliding_state_errors", json.dumps([float(x) for x in state_errors])
    )
    record_property("sliding_algorithmic_losses_j", json.dumps(losses))


@pytest.mark.parametrize("ball_velocity_mps", [-0.4, 0.4])
def test_continuous_sliding_work_closes_without_endpoint_or_return_map(
    ball_velocity_mps: float,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise AssertionError("the independent reference cannot use the discrete solver")

    monkeypatch.setattr(stepper, "root", unavailable)
    monkeypatch.setattr(stepper, "advance_tangential_contact", unavailable)
    reference = sliding_case(ball_velocity_mps)
    twists = reference.initial.mechanical.twists
    _, initial, initial_response = reference.snapshot(
        0, np.r_[np.zeros(twists.size), twists.ravel()]
    )
    state, response, integrals = reference.integrate(
        _controls().bounds_s[1], 1e-12, 1e-14
    )
    storage_change = (
        state.tangential.elastic_energy_j - initial.tangential.elastic_energy_j
    )
    assert (
        abs(storage_change) > 1e-8
    )  # J; a constant-bound friction oracle is insufficient
    assert storage_change * ball_velocity_mps < 0  # rising versus shrinking cap
    defect = (
        response.mechanical_normal_energy_j
        - initial_response.mechanical_normal_energy_j
        + storage_change
        + integrals[:5] @ np.array([-1, 1, 1, 1, 1])
        + integrals[9]
    )
    assert (
        abs(defect) < 1e-10
    )  # J; full continuous work identity, not a fitted correction
    record_property("continuous_sliding_energy_defect_j", float(defect))
    record_property("continuous_sliding_storage_change_j", float(storage_change))
