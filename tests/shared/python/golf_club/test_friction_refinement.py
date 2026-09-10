"""Independent smooth reference and complete compress/release refinement."""

import json
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from shared.python.swing_sim.impact._friction_contact_trajectory import (
    FrictionTrajectory,
    FrictionTrajectoryControls,
    FrictionTrajectorySample,
    integrate_friction_contact,
)
from shared.python.swing_sim.impact._spatial_contact_kinematics import ContactBodyState

from .test_friction_contact_trajectory import _friction_case, _motion
from .test_normal_contact_trajectory import _generator, _powers


@pytest.mark.timeout(180)
def test_first_order_motion_refines_against_independent_matrix_exponential(
    record_property: Callable[[str, object], None],
) -> None:
    problem, initial = _friction_case()
    generator, exact_initial = _generator()
    end = 0.0004
    expected = (expm(end * generator) @ exact_initial)[:6]
    expected_force = 2e4 * (expected[1] - expected[2]) + 3 * (expected[4] - expected[5])
    expected_work = np.array(
        [
            quad(lambda t, i=index: _powers(t)[i], 0, end, epsabs=1e-13, epsrel=1e-11)[
                0
            ]
            for index in range(5)
        ]
    )
    errors, force_errors, work_errors = [], [], []
    # The coarser 4-step grid was outside the preset asymptotic ratio band.
    # Refine the grid while retaining the original error/order requirements.
    for steps in (8, 16, 32):
        controls = FrictionTrajectoryControls((0, end), steps, 10000, 100, 1e-10)
        result = integrate_friction_contact(problem, initial, controls)
        errors.append(
            float(np.linalg.norm(_motion(result.samples[-1].state) - expected))
        )
        last = result.samples[-1]
        force_errors.append(abs(last.response.normal.force_n - expected_force))
        work_errors.append(
            float(np.linalg.norm(np.asarray(last.work.values) - expected_work))
        )
    assert errors[-1] < 0.02
    np.testing.assert_array_less(1.7, np.asarray(errors[:-1]) / errors[1:])
    np.testing.assert_array_less(np.asarray(errors[:-1]) / errors[1:], 2.2)
    assert force_errors[2] < force_errors[1] < force_errors[0]
    assert work_errors[2] < work_errors[1] < work_errors[0]
    assert force_errors[-1] < 0.1  # N; independent linear-contact endpoint force
    assert work_errors[-1] < 1e-5  # J; five independent integrated power channels
    record_property("first_order_matrix_exponential_errors", json.dumps(errors))
    record_property("independent_normal_force_errors_n", json.dumps(force_errors))
    record_property("independent_work_errors_j", json.dumps(work_errors))


def _release_start(initial_gap_m: float) -> tuple:
    problem, initial = _friction_case(0.3)
    ball = initial.mechanical.ball
    pose = np.asarray(ball.pose).copy()
    pose[2, 3] += 0.001 + initial_gap_m
    ball = replace(ball, pose=pose, twist=(0.3, 0, -0.4, 0, 0, 0))
    initial = replace(initial, mechanical=replace(initial.mechanical, ball=ball))
    return problem, initial


def _check_release(result: FrictionTrajectory, initial: ContactBodyState) -> None:
    samples = result.samples
    forces = np.array([sample.response.normal.force_n for sample in samples])
    assert np.max(forces) > 0 and forces[-1] == 0
    if samples[0].response.bodies.contact.gap_m > 1e-12:
        assert forces[0] == 0
        onset = int(np.flatnonzero(forces > 0)[0])
        assert onset > 1
        _verify_unforced_ball(samples[:onset], initial)
    last = samples[-1]
    assert last.response.bodies.contact.gap_m > 0
    assert last.state.tangential.elastic_energy_j == 0
    assert last.plastic_dissipation_j > 0
    for sample in samples:
        assert np.linalg.norm(sample.response.tangential_force_n) <= (
            0.3 * sample.response.normal.force_n + 1e-12
        )
        assert sample.plastic_dissipation_j >= 0
        assert sample.tangential_algorithmic_loss_j >= 0


@pytest.mark.timeout(180)
@pytest.mark.parametrize("initial_gap_m", [0.0, 0.0001])
def test_complete_frictional_compression_release_refines_motion_and_work(
    initial_gap_m: float,
    record_property: Callable[[str, object], None],
) -> None:
    # Preset: decreasing first-order changes; finest defect <0.02 J.
    problem, initial = _release_start(initial_gap_m)
    outputs, defects, algorithmic = [], [], []
    for steps in (30, 60, 120):
        controls = FrictionTrajectoryControls((0, 0.003), steps, 30000, 150, 1e-10)
        result = integrate_friction_contact(problem, initial, controls)
        last = result.samples[-1]
        _check_release(result, initial.mechanical.ball)
        # Explicit dimensions: ball linear velocity [m/s], spin [rad/s],
        # normal/tangent impulses [N s], mechanical energy defect [J].
        outputs.append(
            np.r_[
                last.state.mechanical.ball.twist,
                last.normal_impulse_ns,
                last.tangential_impulse_ns,
            ]
        )
        defects.append(abs(last.energy_balance_error_j))
        algorithmic.append(last.tangential_algorithmic_loss_j)
    # Scale components before comparing mixed translational/angular quantities.
    scales = np.array([1, 1, 1, 100, 100, 100, 0.01, 0.01, 0.01, 0.01])
    differences = [
        float(np.linalg.norm((b - a) / scales))
        for a, b in zip(outputs[:-1], outputs[1:], strict=True)
    ]
    assert differences[1] < differences[0]
    assert defects[-1] < 0.02
    assert defects[2] < defects[1] < defects[0]
    assert algorithmic[2] < algorithmic[1] < algorithmic[0]
    record_property(
        "friction_release_refinement",
        json.dumps(
            {
                "outputs": [value.tolist() for value in outputs],
                "scaled_differences": differences,
                "mechanical_defect_j": defects,
                "tangential_algorithmic_loss_j": algorithmic,
            }
        ),
    )


def _verify_unforced_ball(
    samples: tuple[FrictionTrajectorySample, ...], initial: ContactBodyState
) -> None:
    """Before contact the freely translating ball has constant momentum."""
    velocity = np.asarray(initial.twist)[:3]
    position = np.asarray(initial.pose)[:3, 3]
    for sample in samples:
        ball = sample.state.mechanical.ball
        np.testing.assert_allclose(ball.twist, initial.twist, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            np.asarray(ball.pose)[:3, 3],
            position + sample.time_s * velocity,
            atol=1e-12,
            rtol=0,
        )
        assert sample.normal_impulse_ns == 0
        assert sample.tangential_impulse_ns == (0, 0, 0)
        assert sample.state.tangential.elastic_energy_j == 0
