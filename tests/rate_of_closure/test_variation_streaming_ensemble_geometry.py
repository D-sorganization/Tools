"""Incremental geometry over strict durable ensemble chunks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rate_of_closure.simulation import ContactMode, SimulationConfig, SimulationRun
from rate_of_closure.variation.durable_ensemble_chunks import DurableEnsembleChunkSink
from rate_of_closure.variation.simulation_adapter import (
    run_simulation_ensemble,
    run_simulation_ensemble_chunks,
)
from rate_of_closure.variation.streaming_ensemble_geometry import (
    analyze_durable_ensemble_geometry,
)
from shared.python.swing_sim.variation import compute_position_dispersion

from .test_variation_simulation_adapter import _config, _request

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_streamed_geometry_matches_materialized_dispersion(tmp_path: Path) -> None:
    request = _request(
        (
            _config(ContactMode.DELIVERY_INSPECTION),
            _config(ContactMode.FIXED_BALL_CONTACT),
            _config(ContactMode.DELIVERY_INSPECTION, speed_mph=99.0),
        )
    )

    def executor(config: SimulationConfig) -> SimulationRun:
        from rate_of_closure.simulation import run_simulation

        if config.scenario.clubhead_speed_mph == 99.0:
            raise RuntimeError("planted geometry failure")
        return run_simulation(config)

    materialized = run_simulation_ensemble(request, executor)
    directory = tmp_path / "geometry"
    run_simulation_ensemble_chunks(
        request,
        DurableEnsembleChunkSink(directory),
        chunk_size=2,
        executor=executor,
    )

    actual = analyze_durable_ensemble_geometry(request, directory)
    expected = compute_position_dispersion(materialized.traces)

    assert actual.archive.status == "complete"
    assert actual.analyzed_trial_count == 3
    np.testing.assert_array_equal(actual.dispersion.count, expected.count)
    np.testing.assert_allclose(
        actual.dispersion.mean_positions_m, expected.mean_positions_m
    )
    np.testing.assert_allclose(actual.dispersion.covariance_m2, expected.covariance_m2)
    np.testing.assert_allclose(actual.dispersion.rms_radius_m, expected.rms_radius_m)
    np.testing.assert_allclose(
        actual.dispersion.eigenvalues_m2, expected.eigenvalues_m2
    )
    np.testing.assert_allclose(
        actual.dispersion.principal_axes, expected.principal_axes
    )
