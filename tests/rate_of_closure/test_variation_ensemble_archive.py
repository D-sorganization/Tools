"""Lifecycle and adversarial tests for the resumable complete chunk archive."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rate_of_closure.simulation import ContactMode, SimulationConfig, SimulationRun
from rate_of_closure.variation.ensemble_archive import (
    DurableEnsembleArchiveSink,
    DurableEnsembleChunkSource,
)
from rate_of_closure.variation.ensemble_request_identity import (
    request_identity_sha256,
)
from rate_of_closure.variation.simulation_adapter import run_simulation_ensemble_chunks
from rate_of_closure.variation.simulation_types import SimulationEnsembleRequest
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.solver.solve import CancelledError

from .test_variation_simulation_adapter import _config, _request

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _three_trial_request():
    return _request(
        (
            _config(ContactMode.DELIVERY_INSPECTION),
            _config(ContactMode.FIXED_BALL_CONTACT),
            _config(ContactMode.DELIVERY_INSPECTION),
        )
    )


def test_archive_round_trips_lazily_and_materializes_legacy_result(
    tmp_path: Path,
) -> None:
    request = _three_trial_request()
    archive = tmp_path / "ensemble"

    committed = run_simulation_ensemble_chunks(
        request, DurableEnsembleArchiveSink(archive), chunk_size=2
    )
    source = DurableEnsembleChunkSource(archive)
    chunks = list(source)
    materialized = source.materialize_compatibility()

    assert committed.trial_count == 3
    assert committed.chunk_count == 2
    assert [chunk.start_index for chunk in chunks] == [0, 2]
    assert all(chunk.authority is not None for chunk in chunks)
    np.testing.assert_array_equal(materialized.variation.inputs, request.sampled_inputs)
    assert tuple(item.status for item in materialized.outcomes) == tuple(
        item.status for chunk in chunks for item in chunk.outcomes
    )


def test_cancelled_prefix_resumes_without_reexecuting_verified_trials(
    tmp_path: Path,
) -> None:
    request = _three_trial_request()
    archive = tmp_path / "ensemble"
    cancelled = threading.Event()

    class CancelAfterFirst(DurableEnsembleArchiveSink):
        def accept(self, chunk):  # type: ignore[no-untyped-def]
            super().accept(chunk)
            cancelled.set()

    with pytest.raises(CancelledError):
        run_simulation_ensemble_chunks(
            request,
            CancelAfterFirst(archive),
            chunk_size=1,
            cancel_event=cancelled,
        )

    executed: list[float] = []

    def executor(config: SimulationConfig) -> SimulationRun:
        from rate_of_closure.simulation import run_simulation

        executed.append(config.scenario.clubhead_speed_mph)
        return run_simulation(config)

    progress: list[int] = []
    committed = run_simulation_ensemble_chunks(
        request,
        DurableEnsembleArchiveSink(archive),
        chunk_size=2,
        executor=executor,
        progress_cb=lambda report: progress.append(report.iteration),
    )

    assert committed.trial_count == 3
    assert len(executed) == 2
    assert progress[0] == 1
    assert progress[-1] == 3


def test_provisional_archive_is_not_a_completed_result(tmp_path: Path) -> None:
    request = _request((_config(ContactMode.DELIVERY_INSPECTION),))
    archive = tmp_path / "ensemble"
    sink = DurableEnsembleArchiveSink(archive)

    class StopAfterBegin(DurableEnsembleArchiveSink):
        pass

    from rate_of_closure.simulation.pipeline import configured_swing_sample_times
    from rate_of_closure.simulation.sources import (
        commanded_torque_joint_ids,
        generalized_state_layout,
    )
    from rate_of_closure.variation.ensemble_chunks import EnsembleStreamHeader
    from rate_of_closure.variation.ensemble_trace_authority import (
        EnsembleAuthorityLayout,
    )
    from rate_of_closure.variation.simulation_adapter import APP_FRAME_ID

    state_ids, units = generalized_state_layout("double_pendulum")
    sink.begin(
        EnsembleStreamHeader(
            request.plan,
            request.sampled_inputs,
            configured_swing_sample_times(request.configs[0]),
            ("swing.pivot", "swing.wrist", "swing.clubhead.reference"),
            APP_FRAME_ID,
            EnsembleAuthorityLayout(
                state_ids, units, commanded_torque_joint_ids("double_pendulum")
            ),
            request_identity_sha256(request),
        )
    )
    sink.abort()

    with pytest.raises(ContractViolationError, match="provisional"):
        DurableEnsembleChunkSource(archive)


def test_chunk_bit_flip_is_rejected_before_exposure(tmp_path: Path) -> None:
    archive = tmp_path / "ensemble"
    run_simulation_ensemble_chunks(
        _three_trial_request(), DurableEnsembleArchiveSink(archive), chunk_size=2
    )
    chunk_path = sorted((archive / "chunks").glob("*.roc"))[0]
    data = bytearray(chunk_path.read_bytes())
    data[len(data) // 2] ^= 1
    chunk_path.write_bytes(data)

    with pytest.raises(ContractViolationError, match="checksum"):
        list(DurableEnsembleChunkSource(archive))


def test_request_identity_binds_every_config_not_only_plan() -> None:
    first = _request((_config(ContactMode.DELIVERY_INSPECTION),))
    second = _request((_config(ContactMode.FIXED_BALL_CONTACT),))

    assert request_identity_sha256(first) != request_identity_sha256(second)


def test_request_identity_binds_sample_bytes() -> None:
    first = _request(
        (
            _config(ContactMode.DELIVERY_INSPECTION, speed_mph=100.0),
            _config(ContactMode.DELIVERY_INSPECTION, speed_mph=101.0),
        )
    )
    # Rows that are not the plan's own RNG draws must declare that provenance,
    # and each config is rebound to the row it is executed with, so the request
    # remains a legal one whose only difference from `first` is the sample bytes.
    mutated = np.array(first.sampled_inputs, copy=True) + np.array([[0.0], [0.25]])
    changed_inputs = SimulationEnsembleRequest(
        first.plan,
        mutated,
        tuple(
            replace(config, plane=replace(config.plane, yaw_deg=float(row[0])))
            for config, row in zip(first.configs, mutated, strict=True)
        ),
        sample_provenance="explicit_design",
    )

    assert request_identity_sha256(first) != request_identity_sha256(changed_inputs)


def test_request_rejects_config_order_drift_outright() -> None:
    """Config order is now guaranteed structurally, not just by the digest.

    This previously asserted only that reordering the configs changed the
    request identity. The per-row config binding makes a misordered request
    unconstructable, which is the stronger property: the drift cannot reach the
    digest at all.
    """
    first = _request(
        (
            _config(ContactMode.DELIVERY_INSPECTION, speed_mph=100.0),
            _config(ContactMode.DELIVERY_INSPECTION, speed_mph=101.0),
        )
    )

    with pytest.raises(ContractViolationError):
        SimulationEnsembleRequest(
            first.plan,
            first.sampled_inputs,
            tuple(reversed(first.configs)),
            sample_provenance="explicit_design",
        )
