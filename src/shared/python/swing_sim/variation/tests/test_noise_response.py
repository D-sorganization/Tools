"""Denominator-matched absolute scatter and input-noise response contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.swing_sim.variation import (
    CATEGORY_LAUNCH,
    EnsemblePositionTraces,
    NoiseSpec,
    VariationDataset,
    VariationPlan,
    compute_position_noise_response,
)

_BALL_SPEED = f"{CATEGORY_LAUNCH}.ball_speed_mph"
_POINT_IDS = ("swing.fixed", "swing.responsive")


def _traces(
    *,
    inputs: np.ndarray,
    responsive_x_m: np.ndarray,
    sample_valid: np.ndarray,
    distribution: str = "normal",
    scale: float = 2.0,
) -> EnsemblePositionTraces:
    trial_count, sample_count = responsive_x_m.shape
    spec = NoiseSpec(_BALL_SPEED, distribution=distribution, scale=scale)
    plan = VariationPlan(
        mode="launch",
        base_variables={_BALL_SPEED: 100.0},
        noise=(spec,),
        n_runs=trial_count,
        seed=17,
    )
    variation = VariationDataset(
        plan=plan,
        input_names=(_BALL_SPEED,),
        inputs=np.asarray(inputs, dtype=float).reshape(-1, 1),
        output_names=(),
        outputs=np.empty((trial_count, 0)),
        success=np.ones(trial_count, dtype=bool),
    )
    positions = np.zeros((trial_count, sample_count, len(_POINT_IDS), 3))
    positions[:, :, 1, 0] = responsive_x_m
    positions[~sample_valid] = np.nan
    return EnsemblePositionTraces(
        variation=variation,
        sample_times_s=np.arange(sample_count, dtype=float),
        coordinate_frame="swing.world",
        point_ids=_POINT_IDS,
        positions_m=positions,
        sample_valid=sample_valid,
        impact_sample_indices=np.full(trial_count, -1),
    )


def test_noise_response_uses_the_same_valid_rows_as_absolute_scatter() -> None:
    traces = _traces(
        inputs=np.array([98.0, 100.0, 102.0]),
        responsive_x_m=np.array([[-0.2, -0.1], [0.0, 0.1], [0.2, 9.0]]),
        sample_valid=np.array([[True, True], [True, True], [True, False]]),
    )

    response = compute_position_noise_response(traces)

    np.testing.assert_array_equal(response.count[:, 1], [3, 2])
    np.testing.assert_allclose(
        response.absolute_rms_radius_m[:, 1],
        [0.2 * math.sqrt(2.0 / 3.0), 0.1],
    )
    np.testing.assert_allclose(
        response.standardized_input_rms[:, 1],
        [math.sqrt(2.0 / 3.0), 0.5],
    )
    np.testing.assert_allclose(response.response_gain_m[:, 1], [0.2, 0.2])
    np.testing.assert_allclose(response.response_gain_m[:, 0], 0.0)


@pytest.mark.parametrize(
    ("distribution", "expected_input_rms"),
    [
        ("normal", math.sqrt(2.0 / 3.0)),
        ("uniform", math.sqrt(2.0)),
        ("triangular", 2.0),
    ],
)
def test_noise_response_uses_declared_distribution_standard_deviation(
    distribution: str, expected_input_rms: float
) -> None:
    traces = _traces(
        inputs=np.array([99.0, 100.0, 101.0]),
        responsive_x_m=np.array([[-0.1], [0.0], [0.1]]),
        sample_valid=np.ones((3, 1), dtype=bool),
        distribution=distribution,
        scale=1.0,
    )

    response = compute_position_noise_response(traces)

    assert response.standardized_input_rms[0, 1] == pytest.approx(expected_input_rms)


def test_zero_input_spread_and_one_row_do_not_fabricate_response() -> None:
    traces = _traces(
        inputs=np.array([100.0, 100.0, 100.0]),
        responsive_x_m=np.array([[-0.1, 0.0], [0.0, 9.0], [0.1, 9.0]]),
        sample_valid=np.array([[True, True], [True, False], [True, False]]),
    )

    response = compute_position_noise_response(traces)

    assert response.count[0, 1] == 3
    assert response.absolute_rms_radius_m[0, 1] > 0.0
    assert response.standardized_input_rms[0, 1] == 0.0
    assert np.isnan(response.response_gain_m[0, 1])
    assert response.count[1, 1] == 1
    assert np.isnan(response.standardized_input_rms[1, 1])
    assert np.isnan(response.response_gain_m[1, 1])


def test_noise_response_result_is_immutable_and_preserves_identity() -> None:
    traces = _traces(
        inputs=np.array([98.0, 100.0, 102.0]),
        responsive_x_m=np.array([[-0.2], [0.0], [0.2]]),
        sample_valid=np.ones((3, 1), dtype=bool),
    )

    response = compute_position_noise_response(traces)

    assert response.coordinate_frame == traces.coordinate_frame
    assert response.point_ids == traces.point_ids
    np.testing.assert_array_equal(response.sample_times_s, traces.sample_times_s)
    with pytest.raises(ValueError, match="read-only"):
        response.response_gain_m[0, 0] = 1.0
