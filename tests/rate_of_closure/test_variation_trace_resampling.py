"""Rate source-layout and chunk equivalence for canonical trace resampling."""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.simulation import ContactMode
from rate_of_closure.variation.ensemble_chunks import CollectingEnsembleSink
from rate_of_closure.variation.simulation_adapter import (
    run_simulation_ensemble,
    run_simulation_ensemble_chunks,
    spatial_source_layouts,
)
from shared.python.swing_sim.variation.trace_resampling import (
    resample_position_traces,
)

from .test_variation_simulation_adapter import _config, _request

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.mark.parametrize("source_kind", tuple(spatial_source_layouts()))
def test_every_declared_spatial_layout_has_exact_subset_equivalence(
    source_kind: str,
) -> None:
    result = run_simulation_ensemble(
        _request((_config(ContactMode.DELIVERY_INSPECTION, source_kind=source_kind),))
    )
    target_indices = np.arange(0, result.traces.sample_times_s.size, 2)
    target_times = result.traces.sample_times_s[target_indices]

    aligned = resample_position_traces(result.traces, target_times).traces

    assert aligned.point_ids == spatial_source_layouts()[source_kind]
    assert aligned.coordinate_frame == result.traces.coordinate_frame
    np.testing.assert_array_equal(
        aligned.positions_m, result.traces.positions_m[:, target_indices]
    )
    np.testing.assert_array_equal(
        aligned.sample_valid, result.traces.sample_valid[:, target_indices]
    )


def test_spatial_source_layout_registry_is_immutable() -> None:
    layouts = spatial_source_layouts()

    with pytest.raises(TypeError):
        layouts["manual"] = ("changed",)  # type: ignore[index]


def test_serial_and_batched_ensemble_results_resample_identically() -> None:
    configs = tuple(
        _config(ContactMode.DELIVERY_INSPECTION, speed_mph=95.0 + index)
        for index in range(3)
    )
    request = _request(configs)
    serial = run_simulation_ensemble_chunks(
        request, CollectingEnsembleSink(), chunk_size=1
    )
    batched = run_simulation_ensemble_chunks(
        request, CollectingEnsembleSink(), chunk_size=3
    )
    target = serial.traces.sample_times_s[::2]

    serial_aligned = resample_position_traces(serial.traces, target)
    batched_aligned = resample_position_traces(batched.traces, target)

    np.testing.assert_array_equal(
        serial_aligned.traces.positions_m, batched_aligned.traces.positions_m
    )
    np.testing.assert_array_equal(
        serial_aligned.traces.sample_valid, batched_aligned.traces.sample_valid
    )
    np.testing.assert_array_equal(
        serial_aligned.traces.impact_sample_indices,
        batched_aligned.traces.impact_sample_indices,
    )
    np.testing.assert_array_equal(
        serial_aligned.impact_alignment_error_s,
        batched_aligned.impact_alignment_error_s,
    )
