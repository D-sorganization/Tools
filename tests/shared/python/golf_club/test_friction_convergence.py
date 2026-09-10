"""Residual convergence and independently checked refusal at a captured state."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared.python.swing_sim.impact import _friction_contact_step as stepper
from shared.python.swing_sim.impact._friction_contact_trajectory import (
    FrictionTrajectoryControls,
    integrate_friction_contact,
)

from .test_friction_contact_trajectory import _controls, _friction_case
from .test_friction_refinement import _release_start


def _captured_case() -> tuple:
    root = Path(__file__).resolve().parents[4]
    path = root / "tests/fixtures/impact/friction_stagnation_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    problem, base = _release_start(0.0001)
    old = base.mechanical
    mechanical = replace(
        old,
        shaft=replace(
            old.shaft, poses=data["shaft_poses"], twists=data["shaft_twists"]
        ),
        ball=replace(old.ball, pose=data["ball_pose"], twist=data["ball_twist"]),
    )
    state = replace(
        base,
        mechanical=mechanical,
        tangential=replace(
            base.tangential,
            normal=data["normal"],
            elastic_deflection_m=data["elastic_deflection_m"],
        ),
    )
    bounds = (data["cell"][0], data["cell"][2])
    return problem, state, FrictionTrajectoryControls(bounds, 1, 30000, 150, 1e-10)


def test_captured_interior_step_converges_and_records_its_actual_criterion() -> None:
    problem, initial, controls = _captured_case()
    result = integrate_friction_contact(problem, initial, controls)
    from shared.python.swing_sim.impact._friction_trajectory_contracts import (
        FrictionTermination,
    )

    sample = result.samples[-1]
    assert sample.convergence.reason is FrictionTermination.RESIDUAL
    assert sample.scaled_solver_residual <= controls.scaled_residual_tolerance
    assert result.samples[0].convergence.reason is FrictionTermination.INITIAL
    assert sample.response.normal.force_n > 0
    assert sample.response.bodies.contact.gap_m < 0
    # Independent isotropic ball angular equation, using endpoint material torque.
    contact = sample.response.bodies.contact
    force = sample.response.normal.force_n * np.asarray(contact.normal)
    force += np.asarray(sample.response.tangential_force_n)
    rotation = np.asarray(contact.ball.pose)[:3, :3]
    torque = np.cross(contact.ball_offset_m, rotation.T @ force)
    step = controls.bounds_s[1] - controls.bounds_s[0]
    old_spin = np.asarray(initial.mechanical.ball.twist)[3:]
    new_spin = np.asarray(sample.state.mechanical.ball.twist)[3:]
    np.testing.assert_allclose(8e-6 * (new_spin - old_spin) / step, torque, atol=1e-10)


@pytest.mark.parametrize("success", [False, True])
def test_backend_report_cannot_accept_an_incorrect_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    success: bool,
) -> None:
    def report(function: object, guess: np.ndarray, **kwargs: object) -> object:
        return SimpleNamespace(success=success, x=guess, message="test report")

    monkeypatch.setattr(stepper, "root", report)
    problem, initial = _friction_case()
    with pytest.raises(ValueError):
        integrate_friction_contact(problem, initial, _controls(2))
    assert initial.tangential.elastic_deflection_m == (0, 0, 0)


def test_residual_stop_still_requires_a_fresh_endpoint_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def misleading_residual(self: object, point: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros_like(point)

    monkeypatch.setattr(stepper.FrictionStep, "residual", misleading_residual)
    problem, initial = _friction_case()
    with pytest.raises(ValueError, match="residual"):
        integrate_friction_contact(problem, initial, _controls(2))
    assert calls > 0


def test_convergence_record_rejects_unknown_reason_and_invalid_residual() -> None:
    from shared.python.swing_sim.impact._friction_trajectory_contracts import (
        FrictionConvergence,
        FrictionTermination,
    )

    with pytest.raises(TypeError):
        FrictionConvergence("residual", 0.0)
    for value in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises((ValueError, TypeError)):
            FrictionConvergence(FrictionTermination.RESIDUAL, value)
    with pytest.raises(ValueError):
        FrictionConvergence(FrictionTermination.INITIAL, 0.1)
