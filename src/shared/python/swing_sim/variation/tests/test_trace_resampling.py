"""Canonical common-grid resampling and missing-data semantics."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    CATEGORY_LAUNCH,
    EnsemblePositionTraces,
    NoiseSpec,
    VariationDataset,
    VariationPlan,
    outputs_for_mode,
)
from shared.python.swing_sim.variation.trace_resampling import (
    TRACE_RESAMPLING_POLICY_ID,
    resample_position_traces,
)

pytestmark = pytest.mark.physics

_BALL_SPEED = f"{CATEGORY_LAUNCH}.ball_speed_mph"
_POINT_IDS = ("swing.pivot", "swing.clubhead.reference")


def _variation(n_trials: int) -> VariationDataset:
    plan = VariationPlan(
        mode="launch",
        noise=(NoiseSpec(_BALL_SPEED, scale=1.0),),
        n_runs=n_trials,
        seed=17,
    )
    output_names = outputs_for_mode("launch")
    return VariationDataset(
        plan=plan,
        input_names=(_BALL_SPEED,),
        inputs=np.arange(n_trials, dtype=float).reshape(-1, 1),
        output_names=output_names,
        outputs=np.zeros((n_trials, len(output_names))),
        success=np.ones(n_trials, dtype=bool),
    )


def _traces(
    valid: np.ndarray | None = None, impacts: np.ndarray | None = None
) -> EnsemblePositionTraces:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    n_trials = 2
    positions = np.empty((n_trials, times.size, len(_POINT_IDS), 3))
    for trial in range(n_trials):
        for point in range(len(_POINT_IDS)):
            offset = 10.0 * trial + point
            positions[trial, :, point, 0] = offset + 2.0 * times
            positions[trial, :, point, 1] = offset - 3.0 * times
            positions[trial, :, point, 2] = offset + 0.5 * times
    sample_valid = (
        np.ones((n_trials, times.size), dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    positions[~sample_valid] = np.nan
    return EnsemblePositionTraces(
        variation=_variation(n_trials),
        sample_times_s=times,
        coordinate_frame="app_frame:x_target,y_up,z_right",
        point_ids=_POINT_IDS,
        positions_m=positions,
        sample_valid=sample_valid,
        impact_sample_indices=(
            np.array([2, -1]) if impacts is None else np.asarray(impacts, dtype=int)
        ),
    )


def test_exact_grid_is_an_immutable_identity_with_explicit_policy() -> None:
    source = _traces()

    result = resample_position_traces(source, source.sample_times_s)

    assert result.policy_id == TRACE_RESAMPLING_POLICY_ID
    assert result.coordinate_kind == "time"
    assert result.coordinate_unit == "s"
    assert result.position_method == "piecewise_linear_adjacent_valid_samples"
    assert result.outside_domain == "reject"
    assert result.invalid_gap == "preserve_unavailable"
    assert result.impact_marker_method == "nearest_valid_target_lower_tie"
    assert result.traces.coordinate_frame == source.coordinate_frame
    assert result.traces.point_ids == source.point_ids
    np.testing.assert_array_equal(result.traces.sample_times_s, source.sample_times_s)
    np.testing.assert_array_equal(result.traces.positions_m, source.positions_m)
    np.testing.assert_array_equal(result.traces.sample_valid, source.sample_valid)
    np.testing.assert_array_equal(
        result.traces.impact_sample_indices, source.impact_sample_indices
    )
    np.testing.assert_array_equal(result.impact_alignment_error_s, [0.0, np.nan])
    assert not result.traces.positions_m.flags.writeable
    assert not result.impact_alignment_error_s.flags.writeable


def test_affine_positions_interpolate_on_an_off_grid_and_exact_subset() -> None:
    source = _traces()
    target = np.array([0.0, 0.5, 2.0, 2.5])

    result = resample_position_traces(source, target)

    expected = np.stack(
        [
            source.positions_m[:, 0],
            0.5 * (source.positions_m[:, 0] + source.positions_m[:, 1]),
            source.positions_m[:, 2],
            0.5 * (source.positions_m[:, 2] + source.positions_m[:, 3]),
        ],
        axis=1,
    )
    np.testing.assert_allclose(result.traces.positions_m, expected)
    assert np.all(result.traces.sample_valid)


def test_missing_regions_are_never_extrapolated_or_bridged() -> None:
    source = _traces(
        np.array(
            [
                [False, True, False, True],
                [False, False, True, False],
            ]
        ),
        np.array([-1, -1]),
    )
    target = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    result = resample_position_traces(source, target)

    expected_valid = np.array(
        [
            [False, False, True, False, False, False, True],
            [False, False, False, False, True, False, False],
        ]
    )
    np.testing.assert_array_equal(result.traces.sample_valid, expected_valid)
    assert np.all(np.isnan(result.traces.positions_m[~expected_valid]))
    np.testing.assert_array_equal(
        result.traces.positions_m[0, expected_valid[0]],
        source.positions_m[0, [1, 3]],
    )
    np.testing.assert_array_equal(
        result.traces.positions_m[1, expected_valid[1]], source.positions_m[1, [2]]
    )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (np.array([]), "non-empty"),
        (np.array([0.0, 0.0]), "strictly increasing"),
        (np.array([1.0, 0.0]), "strictly increasing"),
        (np.array([0.0, np.nan]), "finite"),
        (np.array([-0.1, 0.0]), "outside the source domain"),
        (np.array([2.0, 3.1]), "outside the source domain"),
        (np.array([[0.0, 1.0]]), "1-D"),
    ],
)
def test_invalid_or_extrapolating_target_grid_fails_closed(
    target: np.ndarray, message: str
) -> None:
    with pytest.raises(ContractViolationError, match=message):
        resample_position_traces(_traces(), target)


def test_impact_marker_uses_nearest_valid_target_and_lower_tie() -> None:
    source = _traces()

    result = resample_position_traces(source, np.array([0.0, 1.5, 2.5, 3.0]))

    np.testing.assert_array_equal(result.traces.impact_sample_indices, [1, -1])
    np.testing.assert_allclose(result.impact_alignment_error_s[0], 0.5)
    assert np.isnan(result.impact_alignment_error_s[1])


def test_hit_marker_without_any_valid_target_sample_fails_closed() -> None:
    source = _traces(
        np.array([[True, False, False, False], [True, True, True, True]]),
        np.array([0, -1]),
    )

    with pytest.raises(ContractViolationError, match="impact marker has no valid"):
        resample_position_traces(source, np.array([1.0, 2.0, 3.0]))


def test_all_invalid_failure_trace_remains_explicitly_unavailable() -> None:
    source = _traces(
        np.zeros((2, 4), dtype=bool),
        np.array([-1, -1]),
    )

    result = resample_position_traces(source, np.array([0.5, 1.5, 2.5]))

    assert not np.any(result.traces.sample_valid)
    assert np.all(np.isnan(result.traces.positions_m))
    np.testing.assert_array_equal(result.traces.impact_sample_indices, [-1, -1])
    assert np.all(np.isnan(result.impact_alignment_error_s))


def test_source_arrays_are_not_aliased_by_resampling() -> None:
    source = _traces()
    result = resample_position_traces(source, np.array([0.0, 1.0]))

    assert not np.shares_memory(source.positions_m, result.traces.positions_m)
    assert not np.shares_memory(source.sample_valid, result.traces.sample_valid)
