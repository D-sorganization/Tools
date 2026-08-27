"""Restart and integrity tests for durable Rate ensemble chunks."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rate_of_closure.simulation import ContactMode, SimulationConfig, SimulationRun
from rate_of_closure.variation._complete_trial_wire import pack_complete_records
from rate_of_closure.variation.complete_trial_record import CompleteTrialRecord
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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_identity(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _rewrite_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".rewrite.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _record_fingerprint(record: CompleteTrialRecord) -> tuple[str, str]:
    arrays = pack_complete_records((record,))
    return (
        hashlib.sha256(arrays["complete_records_json"].tobytes()).hexdigest(),
        hashlib.sha256(arrays["complete_record_values"].tobytes()).hexdigest(),
    )


def _scan_records(
    request: SimulationEnsembleRequest, directory: Path
) -> list[CompleteTrialRecord]:
    records: list[CompleteTrialRecord] = []
    DurableEnsembleChunkSink(directory).scan(
        request, lambda chunk: records.extend(chunk.complete_records)
    )
    return records


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
    assert manifest["schema_version"] == 3
    assert manifest["header"]["trial_record_schema"] == "rate-complete-trial/v1"
    assert "plan" not in manifest["header"]
    assert "execution_metadata" not in manifest["header"]
    assert manifest["header"]["plan_document"]["schema_version"] == 3
    assert manifest["header"]["plan_document"]["plan"] == request.plan.to_json_dict()


def test_complete_archive_round_trips_every_complete_trial_record(
    tmp_path: Path,
) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(directory), chunk_size=2
    )
    records = []

    archive = DurableEnsembleChunkSink(directory).scan(
        request, lambda chunk: records.extend(chunk.complete_records)
    )

    assert archive.status == "complete"
    assert [record.trial_index for record in records] == [0, 1, 2]
    assert all(record.source_kind == "double_pendulum" for record in records)
    assert all(record.swing_times_s.size > 0 for record in records)
    assert all(not record.swing_times_s.flags.writeable for record in records)
    assert np.array_equal(records[1].sampled_inputs, request.sampled_inputs[1])
    assert all(record.impact_outcome is not None for record in records)
    assert any(record.delivery_state is not None for record in records)
    assert any(record.delivery_state is None for record in records)


def test_resumed_archive_preserves_complete_records_across_process_boundary(
    tmp_path: Path,
) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    _interrupt_after_first_chunk(request, directory)
    run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(directory), chunk_size=2
    )
    chunks = []

    DurableEnsembleChunkSink(directory).scan(request, chunks.append)

    assert [chunk.start_index for chunk in chunks] == [0, 1]
    assert [
        record.trial_index for chunk in chunks for record in chunk.complete_records
    ] == [0, 1, 2]


def test_complete_record_payload_is_serial_chunk_and_resume_invariant(
    tmp_path: Path,
) -> None:
    request = _three_trial_request()
    serial = tmp_path / "serial"
    whole = tmp_path / "whole"
    resumed = tmp_path / "resumed"
    run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(serial), chunk_size=1
    )
    run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(whole), chunk_size=3
    )
    _interrupt_after_first_chunk(request, resumed)
    run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(resumed), chunk_size=2
    )

    fingerprints = [
        [_record_fingerprint(record) for record in _scan_records(request, directory)]
        for directory in (serial, whole, resumed)
    ]

    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_numerical_failure_round_trips_without_fabricated_physics(
    tmp_path: Path,
) -> None:
    from rate_of_closure.simulation import run_simulation

    request = _three_trial_request()
    directory = tmp_path / "campaign"
    call_count = 0

    def executor(config: SimulationConfig) -> SimulationRun:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise FloatingPointError("planted durable failure")
        return run_simulation(config)

    run_simulation_ensemble_chunks(
        request,
        DurableEnsembleChunkSink(directory),
        chunk_size=3,
        executor=executor,
    )
    record = _scan_records(request, directory)[1]

    assert record.status.value == "numerical_failure"
    assert record.failure_type == "FloatingPointError"
    assert record.failure_message == "planted durable failure"
    assert record.swing_times_s.size == 0
    assert record.flight_times_s.size == 0
    assert record.impact_outcome is None


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


def test_tampered_per_array_identity_fails_closed(tmp_path: Path) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(directory), chunk_size=3
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunks"][0]["arrays"]["complete_record_values"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ContractViolationError, match="array manifest"):
        DurableEnsembleChunkSink(directory).scan(request, lambda _chunk: None)


def test_corrupt_complete_record_metadata_fails_after_file_rebinding(
    tmp_path: Path,
) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(directory), chunk_size=3
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_path = directory / manifest["chunks"][0]["file"]
    with np.load(chunk_path, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    arrays["complete_records_json"][0] = ord("[")
    _rewrite_npz(chunk_path, arrays)
    manifest["chunks"][0]["sha256"] = hashlib.sha256(
        chunk_path.read_bytes()
    ).hexdigest()
    manifest["chunks"][0]["arrays"]["complete_records_json"] = _array_identity(
        arrays["complete_records_json"]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ContractViolationError, match="strict JSON"):
        DurableEnsembleChunkSink(directory).scan(request, lambda _chunk: None)


def test_schema_v2_archive_is_readable_but_cannot_resume_as_complete_retention(
    tmp_path: Path,
) -> None:
    request = _three_trial_request()
    directory = tmp_path / "campaign"
    run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(directory), chunk_size=3
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest["header"].pop("trial_record_schema")
    manifest["header_sha256"] = _canonical_sha256(manifest["header"])
    for chunk in manifest["chunks"]:
        chunk_path = directory / chunk["file"]
        with np.load(chunk_path, allow_pickle=False) as source:
            arrays = {
                name: np.array(source[name], copy=True)
                for name in source.files
                if name not in {"complete_records_json", "complete_record_values"}
            }
        _rewrite_npz(chunk_path, arrays)
        chunk["sha256"] = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
        chunk.pop("arrays")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records = []

    archive = DurableEnsembleChunkSink(directory).scan(
        request, lambda chunk: records.extend(chunk.complete_records)
    )

    assert archive.trial_record_schema is None
    assert records == []
    with pytest.raises(ContractViolationError, match="legacy durable archives"):
        run_simulation_ensemble_chunks(
            request, DurableEnsembleChunkSink(directory), chunk_size=3
        )


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
