"""Independent continuous sliding through force cutoff and geometric separation."""

import json
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pytest

from shared.python.swing_sim.impact._friction_contact_response import (
    FrictionContactResponse,
)
from shared.python.swing_sim.impact._friction_contact_trajectory import (
    FrictionTrajectoryControls,
    FrictionTrajectoryState,
    integrate_friction_contact,
)

from ._friction_sliding_reference import (
    SlidingReference,
    SlidingReferenceTrace,
    sliding_case,
    sliding_observables,
)
from .test_friction_continuous_reference import _scaled_output

_END_S = 0.003
_STEPS = (30, 60, 120)


@dataclass(frozen=True)
class _ReleaseCase:
    reference: SlidingReference
    coarse: SlidingReferenceTrace
    fine: SlidingReferenceTrace

    def endpoint(
        self, trace: SlidingReferenceTrace
    ) -> tuple[FrictionTrajectoryState, FrictionContactResponse, np.ndarray]:
        _, state, response = self.reference.snapshot(
            float(trace.times_s[-1]), trace.vectors[:, -1]
        )
        return state, response, trace.vectors[-10:, -1]


@pytest.fixture(scope="module", params=[-0.4, 0.4])
def release_case(request: pytest.FixtureRequest) -> _ReleaseCase:
    """Share owned, read-only ODE traces; production trials remain independent."""
    reference = sliding_case(request.param)
    return _ReleaseCase(
        reference,
        reference.trace(_END_S, 1e-11, 1e-13),
        reference.trace(_END_S, 1e-12, 1e-14),
    )


def test_continuous_sliding_releases_storage_and_separates_with_closed_work(
    release_case: _ReleaseCase,
    record_property: Callable[[str, object], None],
) -> None:
    reference = release_case.reference
    mechanical = reference.initial.mechanical
    twists = mechanical.twists
    _, initial, initial_response = reference.snapshot(
        0, np.r_[np.zeros(twists.size), twists.ravel()]
    )
    state, response, integrals = release_case.endpoint(release_case.coarse)
    finer_state, finer_response, finer_integrals = release_case.endpoint(
        release_case.fine
    )
    actual = sliding_observables(state, response, integrals)
    finer = sliding_observables(finer_state, finer_response, finer_integrals)
    # Preset absolute SI comparison; no fitted correction to force, spin or work.
    np.testing.assert_allclose(actual, finer, atol=1e-8, rtol=0)
    contact = finer_response.bodies.contact
    assert contact.gap_m > 0 and contact.gap_rate_mps > 0
    assert finer_response.normal.force_n == 0
    assert finer_state.tangential.elastic_energy_j == 0
    assert finer_integrals[4] > 0  # cutoff storage loss occurs before gap clears
    assert finer_integrals[9] > 0  # positive continuous plastic work
    defect = (
        finer_response.mechanical_normal_energy_j
        - initial_response.mechanical_normal_energy_j
        - initial.tangential.elastic_energy_j
        + finer_integrals[:5] @ np.array([-1, 1, 1, 1, 1])
        + finer_integrals[9]
    )
    assert abs(defect) < 1e-8  # J; complete continuous work, including cutoff
    record_property("separation_reference_outputs", json.dumps(finer.tolist()))
    record_property("separation_reference_energy_defect_j", float(defect))


def test_cutoff_precedes_separation_and_unforced_ball_motion_is_independent(
    release_case: _ReleaseCase,
    record_property: Callable[[str, object], None],
) -> None:
    reference = release_case.reference
    trace = release_case.fine
    vectors = trace.vectors
    assert not vectors.flags.writeable
    assert [len(times) for times in trace.event_times] == [1, 1]
    cutoff, separation = (float(times[0]) for times in trace.event_times)
    assert 0 < cutoff < separation < trace.times_s[-1]
    _, _, cutoff_response = reference.snapshot(cutoff, trace.event_vectors[0][0])
    _, separated, separated_response = reference.snapshot(
        separation, trace.event_vectors[1][0]
    )
    _, final, _ = reference.snapshot(float(trace.times_s[-1]), trace.vectors[:, -1])
    cutoff_contact = cutoff_response.bodies.contact
    separated_contact = separated_response.bodies.contact
    assert cutoff_contact.gap_m < 0 and cutoff_response.normal.force_n < 1e-9
    assert cutoff_response.normal.elastic_energy_j > 0
    assert abs(separated_contact.gap_m) < 1e-11
    released = trace.vectors[-6, -1] - trace.event_vectors[0][0, -6]
    assert released == pytest.approx(cutoff_response.normal.elastic_energy_j, abs=1e-9)
    before, after = separated.mechanical.ball, final.mechanical.ball
    before_pose, after_pose = np.asarray(before.pose), np.asarray(after.pose)
    before_twist, after_twist = np.asarray(before.twist), np.asarray(after.twist)
    velocity = before_pose[:3, :3] @ before_twist[:3]
    np.testing.assert_allclose(
        after_pose[:3, :3] @ after_twist[:3], velocity, atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        after_pose[:3, 3],
        before_pose[:3, 3] + (trace.times_s[-1] - separation) * velocity,
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        after_pose[:3, :3] @ after_twist[3:],
        before_pose[:3, :3] @ before_twist[3:],
        atol=1e-8,
        rtol=0,
    )
    record_property("cutoff_separation_times_s", json.dumps([cutoff, separation]))
    record_property(
        "cutoff_stored_normal_energy_j", cutoff_response.normal.elastic_energy_j
    )


def test_sliding_trace_refuses_an_initial_history_off_the_saturated_branch() -> None:
    reference = sliding_case()
    initial = reference.initial
    history = replace(initial.tangential, elastic_deflection_m=(0, 0, 0))
    invalid = replace(reference, initial=replace(initial, tangential=history))
    with pytest.raises(AssertionError, match="saturated"):
        invalid.trace(0.003, 1e-11, 1e-13)


def _release_outputs(
    state: FrictionTrajectoryState,
    response: FrictionContactResponse,
    integrals: np.ndarray,
) -> np.ndarray:
    channels = sliding_observables(state, response, integrals)
    ball = state.mechanical.ball
    normal_velocity = (np.asarray(ball.pose)[:3, :3] @ np.asarray(ball.twist)[:3])[2]
    return np.asarray(np.r_[channels[2:7], normal_velocity, channels[7:]])


def _trial(
    reference: SlidingReference, steps: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    controls = FrictionTrajectoryControls((0, _END_S), steps, 30000, 150, 1e-10)
    last = integrate_friction_contact(
        reference.problem, reference.initial, controls
    ).samples[-1]
    integrals = np.r_[
        last.work.values,
        last.normal_impulse_ns,
        last.tangential_impulse_ns,
        last.plastic_dissipation_j,
    ]
    response = last.response
    assert response.normal.force_n == 0
    return (
        _release_outputs(last.state, last.response, integrals),
        _scaled_output(last.state),
        abs(last.energy_balance_error_j),
        last.tangential_algorithmic_loss_j,
    )


def test_full_sliding_release_refines_each_impulse_spin_velocity_and_work_channel(
    release_case: _ReleaseCase,
    record_property: Callable[[str, object], None],
) -> None:
    reference = release_case.reference
    state, response, integrals = release_case.endpoint(release_case.fine)
    expected = _release_outputs(state, response, integrals)
    trials = [_trial(reference, steps) for steps in _STEPS]
    errors = np.abs(np.array([trial[0] for trial in trials]) - expected)
    # Preset SI bounds: vx [m/s], wy [rad/s], Jn/Jtx [N s], Dp [J], vz [m/s].
    limits = np.array([1e-2, 5e-2, 5e-4, 1e-5, 1e-4, 2e-2])
    np.testing.assert_array_less(errors[-1, :6], limits)
    ratios = errors[:-1, :6] / errors[1:, :6]
    np.testing.assert_array_less(1.5, ratios)
    np.testing.assert_array_less(ratios, 2.5)
    np.testing.assert_array_less(errors[-1, 6:], 5e-4)  # each other work port [J]
    np.testing.assert_array_less(errors[2, 6:], errors[1, 6:])
    np.testing.assert_array_less(errors[1, 6:], errors[0, 6:])
    state_errors = [
        float(np.linalg.norm(trial[1] - _scaled_output(state))) for trial in trials
    ]
    assert state_errors[2] < state_errors[1] < state_errors[0]
    defects, algorithmic = (
        [trial[2] for trial in trials],
        [trial[3] for trial in trials],
    )
    assert defects[2] < defects[1] < defects[0] and defects[2] < 0.02
    assert 0 < algorithmic[2] < algorithmic[1] < algorithmic[0]
    record_property("full_release_reference_outputs", json.dumps(expected.tolist()))
    record_property("full_release_channel_errors", json.dumps(errors.tolist()))
    record_property("full_release_state_errors", json.dumps(state_errors))
    record_property("full_release_energy_defects_j", json.dumps(defects))
    record_property("full_release_algorithmic_losses_j", json.dumps(algorithmic))
