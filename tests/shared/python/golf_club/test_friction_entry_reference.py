"""Independent first-touch, elastic loading and sliding entry qualification."""

import json
import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from shared.python.swing_sim.impact._friction_contact_response import (
    FrictionContactResponse,
)
from shared.python.swing_sim.impact._friction_trajectory_contracts import (
    FrictionTrajectoryState,
)

from ._friction_entry_reference import EntryReference, EntryTrace, entry_case
from ._friction_sliding_reference import _readonly
from ._normal_event_oracle import reference as normal_reference
from .test_friction_continuous_reference import _scaled_output
from .test_friction_separation_reference import _release_outputs, _trial

_EntryCase = tuple[EntryReference, EntryTrace]
_Trial = tuple[np.ndarray, np.ndarray, float, float]


@pytest.fixture(scope="module")
def entry_reference() -> _EntryCase:
    reference = entry_case()
    return reference, reference.trace(0.003)


def test_first_touch_matches_the_independent_free_three_mass_oracle(
    entry_reference: _EntryCase, record_property: Callable[[str, object], None]
) -> None:
    reference, result = entry_reference
    _, _, expected_times = normal_reference()
    touch, sliding = result.entry_times_s
    assert touch == pytest.approx(expected_times[0], abs=1e-10)
    assert 0 < sliding - touch < 5e-5  # s; a resolved elastic loading interval
    touch_vector = result.entry_vectors[0]
    np.testing.assert_array_equal(touch_vector[-6:], 0)
    _, _, response = reference.elastic_snapshot(touch, touch_vector)
    assert response.normal.force_n > 0  # incoming dashpot limit, no state jump
    assert response.normal.elastic_energy_j == pytest.approx(0, abs=1e-20)
    record_property("entry_times_s", json.dumps([touch, sliding]))
    record_property("incoming_normal_force_n", response.normal.force_n)


def test_elastic_history_reaches_the_coulomb_boundary_continuously(
    entry_reference: _EntryCase,
) -> None:
    reference, result = entry_reference
    time = result.entry_times_s[1]
    vector = result.entry_vectors[1]
    _, state, elastic = reference.elastic_snapshot(time, vector)
    _, saturated, sliding = reference.sliding.snapshot(time, vector[:-1])
    np.testing.assert_allclose(
        np.asarray(state.tangential.elastic_deflection_m),
        np.asarray(saturated.tangential.elastic_deflection_m),
        atol=1e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        elastic.tangential_force_n, sliding.tangential_force_n, atol=1e-7, rtol=0
    )
    assert state.tangential.elastic_energy_j > 0
    assert vector[-2] == 0  # no plastic work before first sliding


def test_entry_and_full_release_close_continuous_work(
    entry_reference: _EntryCase, record_property: Callable[[str, object], None]
) -> None:
    reference, result = entry_reference
    _, initial, start = reference.free_snapshot(0, reference.initial_vector())
    trace = result.release
    _, final, response = reference.sliding.snapshot(
        float(trace.times_s[-1]), trace.vectors[:, -1]
    )
    work = trace.vectors[-10:, -1]
    defect = (
        response.mechanical_normal_energy_j
        - start.mechanical_normal_energy_j
        - initial.tangential.elastic_energy_j
        + work[:5] @ np.array([-1, 1, 1, 1, 1])
        + work[9]
    )
    assert abs(defect) < 1e-8  # J; includes both precontact and contact work
    assert response.normal.force_n == 0 and final.tangential.elastic_energy_j == 0
    assert [len(times) for times in trace.event_times] == [1, 1]
    assert trace.event_times[0][0] < trace.event_times[1][0]
    record_property("entry_release_energy_defect_j", float(defect))


def _endpoint(
    reference: EntryReference, result: EntryTrace
) -> tuple[FrictionTrajectoryState, FrictionContactResponse, np.ndarray]:
    trace = result.release
    _, state, response = reference.sliding.snapshot(
        float(trace.times_s[-1]), trace.vectors[:, -1]
    )
    return state, response, trace.vectors[-10:, -1]


def test_tighter_entry_reference_preserves_each_event_and_output(
    entry_reference: _EntryCase, record_property: Callable[[str, object], None]
) -> None:
    reference, result = entry_reference
    tighter = replace(reference, tolerances=(1e-12, 1e-14))
    fine = tighter.trace(0.003)
    actual = _release_outputs(*_endpoint(reference, result))
    expected = _release_outputs(*_endpoint(tighter, fine))
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
    times = np.r_[result.entry_times_s, [t[0] for t in result.release.event_times]]
    fine_times = np.r_[fine.entry_times_s, [t[0] for t in fine.release.event_times]]
    np.testing.assert_allclose(times, fine_times, atol=1e-9, rtol=0)
    record_property("entry_reference_outputs", json.dumps(expected.tolist()))
    record_property("entry_all_event_times_s", json.dumps(fine_times.tolist()))


@pytest.fixture(scope="module")
def entry_midpoint_trial(entry_reference: _EntryCase) -> _Trial:
    """Share the live 120-step solve without relying on test execution order."""
    reference, _ = entry_reference
    output, state, defect, loss = _trial(reference.sliding, 120)
    return _readonly(output), _readonly(state), defect, loss


@pytest.mark.parametrize("steps", [60, 240])
def test_production_entry_refines_separate_velocity_spin_impulse_and_work(
    entry_reference: _EntryCase,
    entry_midpoint_trial: _Trial,
    steps: int,
    record_property: Callable[[str, object], None],
) -> None:
    reference, result = entry_reference
    state, response, integrals = _endpoint(reference, result)
    expected = _release_outputs(state, response, integrals)
    current = _trial(reference.sliding, steps)
    trials = (
        (current, entry_midpoint_trial)
        if steps == 60
        else (entry_midpoint_trial, current)
    )
    errors = np.abs(np.array([trial[0] for trial in trials]) - expected)
    defects = [trial[2] for trial in trials]
    algorithmic = [trial[3] for trial in trials]
    record_property("entry_production_channel_errors", json.dumps(errors.tolist()))
    record_property("entry_production_energy_defects_j", json.dumps(defects))
    record_property("entry_production_algorithmic_losses_j", json.dumps(algorithmic))
    record_property("entry_production_reference_outputs", json.dumps(expected.tolist()))
    # Preset SI: vx, wy, Jn, Jtx, Dp, vz; force impulse has no gear-effect fit.
    limits = np.array([1e-4, 1e-2, 1e-4, 2e-6, 1e-5, 5e-3])
    if steps == 240:
        np.testing.assert_array_less(errors[-1, :6], limits)
    ratios = errors[:-1, :6] / errors[1:, :6]
    np.testing.assert_array_less(1.5, ratios)
    np.testing.assert_array_less(ratios, 2.5)
    np.testing.assert_array_less(errors[-1, 6:], 1e-4)  # each work channel [J]
    # Cutoff quadrature has a jump: the analytic control below disproves
    # mandatory monotonic error. Retain its absolute bound and actual error.
    np.testing.assert_array_less(errors[1, 6:-1], errors[0, 6:-1])
    assert defects[1] < defects[0] and defects[1] < 5e-4
    assert 0 < algorithmic[1] < algorithmic[0]
    state_errors = [float(np.linalg.norm(t[1] - _scaled_output(state))) for t in trials]
    assert state_errors[1] < state_errors[0]
    record_property("entry_production_state_errors", json.dumps(state_errors))


def test_incoming_limit_is_refused_inside_unresolved_clearance() -> None:
    reference = entry_case()
    with pytest.raises(AssertionError, match="unresolved contact entry"):
        reference.elastic_snapshot(0, reference.initial_vector())


def test_endpoint_loss_quadrature_converges_without_monotonic_error() -> None:
    # Independent 1 W triangular pulse; exact energy=(b-a)/2, TV(power)=2 W.
    start, end = 0.002585, 0.002728
    exact = 0.5 * (end - start)
    errors = []
    for steps in (60, 120, 240, 480, 960):
        step = 0.003 / steps
        estimate = math.fsum(
            step * (end - index * step) / (end - start)
            for index in range(1, steps + 1)
            if start < index * step < end
        )
        error = abs(estimate - exact)
        assert error <= 2 * step  # |Riemann error| <= max step * total variation
        errors.append(error)
    assert errors[2] > errors[1] and errors[4] > errors[3]
