"""Property and bounded-scale tests for ensemble geometry preparation."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    CATEGORY_SWING,
    EnsemblePositionTraces,
    NoiseSpec,
    PositionDispersionAccumulator,
    VariationDataset,
    VariationPlan,
    compute_position_dispersion,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
_POINT = "swing.clubhead.reference"
_INPUT = f"{CATEGORY_SWING}.yaw_deg"


def _traces(positions_m: np.ndarray) -> EnsemblePositionTraces:
    trials, samples, _points, _xyz = positions_m.shape
    plan = VariationPlan(
        mode="swing",
        noise=(NoiseSpec(_INPUT, scale=0.2),),
        n_runs=trials,
        seed=31,
    )
    variation = VariationDataset(
        plan=plan,
        input_names=(_INPUT,),
        inputs=np.zeros((trials, 1)),
        output_names=("metric",),
        outputs=np.zeros((trials, 1)),
        success=np.ones(trials, dtype=bool),
    )
    return EnsemblePositionTraces(
        variation=variation,
        sample_times_s=np.linspace(0.0, 1.0, samples),
        coordinate_frame="app_frame:x_target,y_up,z_right",
        point_ids=(_POINT,),
        positions_m=positions_m,
        sample_valid=np.ones((trials, samples), dtype=bool),
        impact_sample_indices=np.full(trials, -1),
    )


@given(
    tx=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    ty=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    tz=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, deadline=None)
def test_dispersion_is_invariant_to_rigid_translation(
    tx: float,
    ty: float,
    tz: float,
) -> None:
    rng = np.random.default_rng(9)
    positions = rng.normal(size=(4, 7, 1, 3))
    shifted = positions + np.array([tx, ty, tz])

    original = compute_position_dispersion(_traces(positions))
    translated = compute_position_dispersion(_traces(shifted))

    np.testing.assert_allclose(
        translated.rms_radius_m, original.rms_radius_m, atol=1e-12
    )
    np.testing.assert_allclose(
        translated.eigenvalues_m2, original.eigenvalues_m2, atol=1e-11
    )
    np.testing.assert_allclose(
        translated.mean_positions_m,
        original.mean_positions_m + np.array([tx, ty, tz]),
        atol=1e-12,
    )


def test_geometry_preparation_meets_500_trial_interactive_budget() -> None:
    trials = 500
    samples = 240
    times = np.linspace(0.0, 1.0, samples)
    base = np.column_stack((np.sin(times), np.cos(times), times))[None, :, None, :]
    offsets = np.linspace(-0.02, 0.02, trials)[:, None, None, None]
    directions = np.array([1.0, -0.4, 0.2])[None, None, None, :]
    positions = base + offsets * directions

    tracemalloc.start()
    try:
        started = time.perf_counter()
        dispersion = compute_position_dispersion(_traces(positions))
        elapsed_s = time.perf_counter() - started
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert elapsed_s < 5.0
    assert positions.nbytes < 5_000_000
    assert peak_bytes < 100_000_000
    assert dispersion.rms_radius_m.shape == (samples, 1)
    assert np.all(np.isfinite(dispersion.rms_radius_m))


def test_incremental_geometry_is_chunk_invariant_and_matches_materialized() -> None:
    rng = np.random.default_rng(41)
    positions = rng.normal(size=(7, 9, 2, 3))
    valid = np.ones((7, 9), dtype=bool)
    valid[1, 3:] = False
    valid[5, :2] = False
    positions[~valid] = np.nan
    ensemble = _traces(np.nan_to_num(positions[:, :, :1]))
    ensemble = type(ensemble)(
        variation=ensemble.variation,
        sample_times_s=ensemble.sample_times_s,
        coordinate_frame=ensemble.coordinate_frame,
        point_ids=("swing.wrist", _POINT),
        positions_m=positions,
        sample_valid=valid,
        impact_sample_indices=ensemble.impact_sample_indices,
    )
    expected = compute_position_dispersion(ensemble)

    accumulator = PositionDispersionAccumulator(9, 2)
    accumulator.accept(positions[:2], valid[:2])
    accumulator.accept(positions[2:6], valid[2:6])
    accumulator.accept(positions[6:], valid[6:])
    actual = accumulator.freeze(
        ensemble.sample_times_s, ensemble.coordinate_frame, ensemble.point_ids
    )

    np.testing.assert_array_equal(actual.count, expected.count)
    np.testing.assert_allclose(actual.mean_positions_m, expected.mean_positions_m)
    np.testing.assert_allclose(actual.covariance_m2, expected.covariance_m2)
    np.testing.assert_allclose(actual.rms_radius_m, expected.rms_radius_m)
    np.testing.assert_allclose(actual.eigenvalues_m2, expected.eigenvalues_m2)
    np.testing.assert_allclose(actual.principal_axes, expected.principal_axes)


def test_incremental_geometry_rejects_an_unbounded_accumulator_before_allocation() -> (
    None
):
    with pytest.raises(ContractViolationError, match="memory budget"):
        PositionDispersionAccumulator(100_000, 256)
