"""Restart and integrity tests for durable Rate ensemble chunks."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.simulation import ContactMode, SimulationConfig, SimulationRun
from rate_of_closure.variation.durable_ensemble_chunks import (
    DurableEnsembleArchive,
    DurableEnsembleChunkSink,
)
from rate_of_closure.variation.simulation_adapter import run_simulation_ensemble_chunks
from rate_of_closure.variation.simulation_types import SimulationEnsembleRequest
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.solver.solve import CancelledError, ProgressReport

from .test_variation_simulation_adapter import _config, _request

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _three_trial_request() -> SimulationEnsembleRequest:
    return _request(
        (
            _config(ContactMode.DELIVERY_INSPECTION),
            _config(ContactMode.FIXED_BALL_CONTACT),
            _config(ContactMode.DELIVERY_INSPECTION),
        )
    )


def _interrupt_after_first_chunk(
    request: SimulationEnsembleRequest, directory: Path
) -> None:
    cancelled = threading.Event()

    def stop_after_checkpoint(report: ProgressReport) -> None:
        assert report.iteration == 1
        cancelled.set()

    with pytest.raises(CancelledError):
        run_simulation_ensemble_chunks(
            request,
            DurableEnsembleChunkSink(directory),
            chunk_size=1,
            progress_cb=stop_after_checkpoint,
            cancel_event=cancelled,
        )


def test_interrupted_archive_resumes_without_reexecuting_durable_prefix(
    tmp_path: Path,
) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    _interrupt_after_first_chunk(request, directory)
    executed: list[SimulationConfig] = []
    progress: list[int] = []

    def counting_executor(config: SimulationConfig) -> SimulationRun:
        from rate_of_closure.simulation import run_simulation

        executed.append(config)
        return run_simulation(config)

    archive = run_simulation_ensemble_chunks(
        request,
        DurableEnsembleChunkSink(directory),
        chunk_size=2,
        executor=counting_executor,
        progress_cb=lambda report: progress.append(report.iteration),
    )

    assert isinstance(archive, DurableEnsembleArchive)
    assert archive.trial_count == 3
    assert archive.chunk_count == 2
    assert archive.status == "complete"
    assert len(executed) == 2
    assert progress == [1, 3]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["next_index"] == 3
    assert [item["start_index"] for item in manifest["chunks"]] == [0, 1]


def test_tampered_chunk_fails_before_any_resumed_evaluation(tmp_path: Path) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    _interrupt_after_first_chunk(request, directory)
    chunk_path = next(directory.glob("chunk-*.npz"))
    payload = bytearray(chunk_path.read_bytes())
    payload[-1] ^= 1
    chunk_path.write_bytes(payload)
    executed: list[SimulationConfig] = []

    with pytest.raises(ContractViolationError, match="checksum"):
        run_simulation_ensemble_chunks(
            request,
            DurableEnsembleChunkSink(directory),
            chunk_size=2,
            executor=lambda config: executed.append(config),  # type: ignore[arg-type]
        )

    assert executed == []


def test_changed_header_fails_before_any_resumed_evaluation(tmp_path: Path) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    _interrupt_after_first_chunk(request, directory)
    changed_inputs = request.sampled_inputs.copy()
    changed_inputs[0, 0] += 0.125
    changed = SimulationEnsembleRequest(request.plan, changed_inputs, request.configs)
    executed: list[SimulationConfig] = []

    with pytest.raises(ContractViolationError, match="header identity"):
        run_simulation_ensemble_chunks(
            changed,
            DurableEnsembleChunkSink(directory),
            chunk_size=2,
            executor=lambda config: executed.append(config),  # type: ignore[arg-type]
        )

    assert executed == []


def test_changed_simulation_configuration_cannot_reuse_prefix(tmp_path: Path) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    _interrupt_after_first_chunk(request, directory)
    changed_configs = (
        replace(request.configs[0], impact_time_offset_s=0.001),
        *request.configs[1:],
    )
    changed = SimulationEnsembleRequest(
        request.plan, request.sampled_inputs, changed_configs
    )
    executed: list[SimulationConfig] = []

    with pytest.raises(ContractViolationError, match="header identity"):
        run_simulation_ensemble_chunks(
            changed,
            DurableEnsembleChunkSink(directory),
            chunk_size=2,
            executor=lambda config: executed.append(config),  # type: ignore[arg-type]
        )

    assert executed == []


def test_failed_atomic_manifest_replace_preserves_last_valid_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    _interrupt_after_first_chunk(request, directory)
    manifest_path = directory / "manifest.json"
    retained = manifest_path.read_bytes()
    original_replace = Path.replace

    def reject_manifest(self: Path, target: Path) -> Path:
        if Path(target).name == "manifest.json":
            raise OSError("planted manifest replacement failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", reject_manifest)
    with pytest.raises(OSError, match="planted manifest replacement failure"):
        run_simulation_ensemble_chunks(
            request, DurableEnsembleChunkSink(directory), chunk_size=2
        )

    assert manifest_path.read_bytes() == retained
    manifest = json.loads(retained)
    assert manifest["next_index"] == 1
    assert manifest["status"] == "in_progress"
    monkeypatch.undo()
    inspected = DurableEnsembleChunkSink(directory).inspect(request)
    assert inspected.next_index == 1
    assert inspected.chunk_count == 1


def test_abort_retains_checksum_verified_prefix_for_later_inspection(
    tmp_path: Path,
) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    _interrupt_after_first_chunk(request, directory)

    sink = DurableEnsembleChunkSink(directory)
    archive = sink.inspect(request)

    assert archive.status == "in_progress"
    assert archive.trial_count == 3
    assert archive.next_index == 1
    assert archive.chunk_count == 1


def test_completed_archive_is_idempotent_and_executes_no_trials_again(
    tmp_path: Path,
) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    first = run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(directory), chunk_size=2
    )
    executed: list[SimulationConfig] = []
    progress: list[int] = []

    second = run_simulation_ensemble_chunks(
        request,
        DurableEnsembleChunkSink(directory),
        chunk_size=1,
        executor=lambda config: executed.append(config),  # type: ignore[arg-type]
        progress_cb=lambda report: progress.append(report.iteration),
    )

    assert executed == []
    assert progress == [3]
    assert second == first
