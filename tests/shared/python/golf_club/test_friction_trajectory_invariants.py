"""Independent ball equations, observer invariance and solver refusal tripwires."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shared.python.swing_sim.impact import _friction_contact_step as stepper
from shared.python.swing_sim.impact._friction_contact_trajectory import (
    integrate_friction_contact,
)
from shared.python.swing_sim.impact._friction_transport import ContactTransport

from .test_friction_contact_trajectory import _controls, _friction_case


def _free_case(convention: ContactTransport) -> tuple:
    problem, initial = _friction_case(0.3)
    contact = problem.normal.contact
    chain = contact.chain
    elastic = replace(chain.shaft.elastic, loads=())
    chain = replace(chain, shaft=replace(chain.shaft, elastic=elastic), grips=())
    normal = replace(
        problem.normal,
        contact=replace(contact, chain=chain),
        anchor_history=lambda time_s: (),
    )
    problem = replace(problem, normal=normal, transport=convention)
    ball = replace(initial.mechanical.ball, twist=(0.2, -0.1, -0.4, 1, -2, 20))
    return problem, replace(initial, mechanical=replace(initial.mechanical, ball=ball))


@pytest.mark.parametrize("convention", list(ContactTransport))
def test_endpoint_ball_equations_include_material_transport_and_off_center_torque(
    convention: ContactTransport,
) -> None:
    problem, initial = _free_case(convention)
    result = integrate_friction_contact(problem, initial, _controls(2))
    for old, new in zip(result.samples[:-1], result.samples[1:], strict=True):
        step = new.time_s - old.time_s
        contact = new.response.bodies.contact
        force_world = new.response.normal.force_n * np.asarray(
            contact.normal
        ) + np.asarray(new.response.tangential_force_n)
        rotation = np.asarray(contact.ball.pose)[:3, :3]
        force_body = rotation.T @ force_world
        torque_body = np.cross(contact.ball_offset_m, force_body)
        previous = np.asarray(old.state.mechanical.ball.twist)
        current = np.asarray(new.state.mechanical.ball.twist)
        # Independent centered isotropic rigid-ball Newton/Euler equations.
        force_residual = (
            0.046
            * ((current[:3] - previous[:3]) / step + np.cross(current[3:], current[:3]))
            - force_body
        )
        torque_residual = 8e-6 * (current[3:] - previous[3:]) / step - torque_body
        np.testing.assert_allclose(force_residual, 0, atol=1e-7)
        np.testing.assert_allclose(torque_residual, 0, atol=1e-10)
        assert np.linalg.norm(torque_body) > 0


@pytest.mark.parametrize("convention", list(ContactTransport))
def test_complete_step_is_covariant_under_constant_observer_pose(
    convention: ContactTransport,
) -> None:
    problem, initial = _free_case(convention)
    observer = np.eye(4)
    observer[:3, :3] = Rotation.from_rotvec([0.3, -0.4, 0.5]).as_matrix()
    observer[:3, 3] = [0.2, -0.3, 0.4]
    rotation = observer[:3, :3]
    mechanical = initial.mechanical
    transformed = replace(
        initial,
        mechanical=replace(
            mechanical,
            shaft=replace(mechanical.shaft, poses=observer @ mechanical.shaft.poses),
            ball=replace(mechanical.ball, pose=observer @ mechanical.ball.pose),
        ),
        tangential=replace(
            initial.tangential,
            normal=rotation @ initial.tangential.normal,
            elastic_deflection_m=rotation @ initial.tangential.elastic_deflection_m,
        ),
    )
    before = integrate_friction_contact(problem, initial, _controls(2)).samples[-1]
    after = integrate_friction_contact(problem, transformed, _controls(2)).samples[-1]
    np.testing.assert_allclose(
        after.state.mechanical.twists, before.state.mechanical.twists, atol=1e-8
    )
    np.testing.assert_allclose(
        after.state.mechanical.ball.pose,
        observer @ before.state.mechanical.ball.pose,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        after.state.tangential.elastic_deflection_m,
        rotation @ before.state.tangential.elastic_deflection_m,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        after.tangential_impulse_ns, rotation @ before.tangential_impulse_ns, atol=1e-12
    )
    assert after.energy_balance_error_j == pytest.approx(
        before.energy_balance_error_j, abs=1e-10
    )


def test_declared_relative_twirl_convention_affects_coupled_response() -> None:
    problem, initial = _friction_case(0.3)
    ball = replace(initial.mechanical.ball, twist=(0.2, 0, -0.4, 0, 0, 200))
    initial = replace(
        initial,
        mechanical=replace(initial.mechanical, ball=ball),
        tangential=replace(initial.tangential, elastic_deflection_m=(0.001, 0, 0)),
    )
    results = [
        integrate_friction_contact(
            replace(problem, transport=mode), initial, _controls(2)
        )
        for mode in ContactTransport
    ]
    difference = (
        results[1].samples[-1].state.mechanical.ball.twist[1]
        - results[0].samples[-1].state.mechanical.ball.twist[1]
    )
    assert abs(difference) > 1e-8


def test_solver_success_flag_cannot_bypass_actual_endpoint_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def false_success(function: object, guess: np.ndarray, **kwargs: object) -> object:
        return SimpleNamespace(success=True, x=guess, message="claimed success")

    monkeypatch.setattr(stepper, "root", false_success)
    problem, initial = _friction_case()
    with pytest.raises(ValueError, match="residual"):
        integrate_friction_contact(problem, initial, _controls(2))
    assert initial.tangential.elastic_deflection_m == (0, 0, 0)
