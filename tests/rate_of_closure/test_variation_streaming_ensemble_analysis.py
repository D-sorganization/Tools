"""Incremental analysis contracts for durable Rate ensemble archives."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from rate_of_closure.variation import analyze_durable_ensemble
from rate_of_closure.variation.durable_ensemble_chunks import DurableEnsembleChunkSink
from rate_of_closure.variation.ensemble_chunks import CollectingEnsembleSink
from rate_of_closure.variation.simulation_adapter import (
    run_simulation_ensemble_chunks,
)
from rate_of_closure.variation.simulation_types import TrialEvaluationStatus
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation.analysis import summary_stats

from .test_variation_durable_ensemble_chunks import (
    _interrupt_after_first_chunk,
    _three_trial_request,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_streamed_summary_matches_materialized_scalar_moments(tmp_path: Path) -> None:
    request = _three_trial_request()
    reference = run_simulation_ensemble_chunks(
        request, CollectingEnsembleSink(), chunk_size=2
    )
    directory = tmp_path / "campaign"
    archive = run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(directory), chunk_size=2
    )

    actual = analyze_durable_ensemble(request, directory)
    expected_counts = Counter({item.value: 0 for item in TrialEvaluationStatus})
    expected_counts.update(item.status.value for item in reference.outcomes)
    expected_stats = summary_stats(reference.variation)

    assert actual.archive == archive
    assert actual.layout.coordinate_frame == "app_frame:x_target,y_up,z_right"
    assert actual.layout.sample_count > 0
    assert actual.layout.point_ids == (
        "swing.pivot",
        "swing.wrist",
        "swing.clubhead.reference",
    )
    assert actual.status_counts == expected_counts
    assert actual.analyzed_trial_count == request.plan.n_runs
    assert tuple(item.name for item in actual.output_moments) == tuple(
        item.name for item in expected_stats
    )
    for streamed, expected in zip(actual.output_moments, expected_stats, strict=True):
        assert streamed.available_count == expected.n
        if expected.n == 0:
            assert streamed.mean is None
            assert streamed.sample_std is None
        else:
            assert streamed.mean == pytest.approx(expected.mean, abs=1e-12)
            if expected.n == 1:
                assert streamed.sample_std is None
            else:
                assert streamed.sample_std == pytest.approx(expected.std, abs=1e-12)


def test_in_progress_archive_reports_only_verified_prefix(tmp_path: Path) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    _interrupt_after_first_chunk(request, directory)

    summary = analyze_durable_ensemble(request, directory)

    assert summary.archive.status == "in_progress"
    assert summary.analyzed_trial_count == 1
    assert sum(summary.status_counts.values()) == 1
    assert all(item.available_count <= 1 for item in summary.output_moments)
    assert summary.layout.coordinate_frame == "app_frame:x_target,y_up,z_right"


def test_analysis_rejects_corrupted_archive_before_promotion(tmp_path: Path) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(directory), chunk_size=2
    )
    chunk_path = next(directory.glob("chunk-*.npz"))
    payload = bytearray(chunk_path.read_bytes())
    payload[-1] ^= 1
    chunk_path.write_bytes(payload)

    with pytest.raises(ContractViolationError, match="checksum"):
        analyze_durable_ensemble(request, directory)
