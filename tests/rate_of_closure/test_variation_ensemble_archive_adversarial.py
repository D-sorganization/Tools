"""Tamper, interruption, and authority-binding tests for chunk archives."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.simulation import ContactMode, SimulationConfig, run_simulation
from rate_of_closure.variation import ensemble_archive as archive_module
from rate_of_closure.variation.ensemble_archive import (
    DurableEnsembleArchiveSink,
    DurableEnsembleChunkSource,
)
from rate_of_closure.variation.simulation_adapter import run_simulation_ensemble_chunks
from rate_of_closure.variation.trial_projection import capture_simulation
from shared.python.contracts import ContractViolationError

from .test_variation_simulation_adapter import _config, _request

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _request_pair(speed: float = 100.0):
    return _request(
        (
            _config(ContactMode.DELIVERY_INSPECTION, speed_mph=speed),
            _config(ContactMode.FIXED_BALL_CONTACT, speed_mph=speed),
        )
    )


def _write_archive(path: Path, speed: float = 100.0) -> None:
    run_simulation_ensemble_chunks(
        _request_pair(speed), DurableEnsembleArchiveSink(path), chunk_size=1
    )


def test_header_bound_sampled_input_corruption_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "ensemble"
    _write_archive(path)
    inputs = path / "sampled-inputs.f64"
    data = bytearray(inputs.read_bytes())
    data[0] ^= 1
    inputs.write_bytes(data)

    with pytest.raises(ContractViolationError, match="checksum"):
        DurableEnsembleChunkSource(path)


def test_false_chunk_filename_is_rejected_even_when_payload_is_authentic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ensemble"
    _write_archive(path)
    first = sorted((path / "chunks").glob("*.roc"))[0]
    first.rename(path / "chunks" / "000000000000-000000000002.roc")

    with pytest.raises(ContractViolationError, match="filename"):
        DurableEnsembleChunkSource(path)


def test_different_request_chunk_splice_fails_checksum(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_archive(first, 100.0)
    _write_archive(second, 101.0)
    replacement = sorted((second / "chunks").glob("*.roc"))[0].read_bytes()
    sorted((first / "chunks").glob("*.roc"))[0].write_bytes(replacement)

    with pytest.raises(ContractViolationError, match="checksum"):
        DurableEnsembleChunkSource(first)


def test_commit_root_tamper_fails_before_iteration(tmp_path: Path) -> None:
    path = tmp_path / "ensemble"
    _write_archive(path)
    commit_path = path / "commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["scientific_root_sha256"] = "f" * 64
    commit_path.write_text(json.dumps(commit), encoding="utf-8")

    with pytest.raises(ContractViolationError, match="root"):
        DurableEnsembleChunkSource(path)


def test_interrupted_chunk_write_leaves_resumable_verified_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ensemble"
    request = _request_pair()
    original = archive_module.write_chunk_file
    calls = 0

    def interrupted(target, chunk, archive_sha, previous_sha):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write_bytes(b"interrupted")
            raise OSError("planted storage interruption")
        return original(target, chunk, archive_sha, previous_sha)

    monkeypatch.setattr(archive_module, "write_chunk_file", interrupted)
    with pytest.raises(OSError, match="interruption"):
        run_simulation_ensemble_chunks(
            request, DurableEnsembleArchiveSink(path), chunk_size=1
        )

    monkeypatch.setattr(archive_module, "write_chunk_file", original)
    committed = run_simulation_ensemble_chunks(
        request, DurableEnsembleArchiveSink(path), chunk_size=1
    )

    assert committed.trial_count == 2
    assert len(list(DurableEnsembleChunkSource(path))) == 2


def test_resume_rejects_same_plan_with_different_config(tmp_path: Path) -> None:
    path = tmp_path / "ensemble"
    request = _request_pair()

    class StopAfterFirst(DurableEnsembleArchiveSink):
        def accept(self, chunk):  # type: ignore[no-untyped-def]
            super().accept(chunk)
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        run_simulation_ensemble_chunks(request, StopAfterFirst(path), chunk_size=1)

    changed = _request_pair(101.0)
    with pytest.raises(ContractViolationError, match="request"):
        run_simulation_ensemble_chunks(
            changed, DurableEnsembleArchiveSink(path), chunk_size=1
        )


@pytest.mark.parametrize(
    "returned_config",
    (
        _config(ContactMode.FIXED_BALL_CONTACT),
        _config(ContactMode.DELIVERY_INSPECTION, source_kind="manual"),
        _config(ContactMode.DELIVERY_INSPECTION, speed_mph=101.0),
        replace(_config(ContactMode.DELIVERY_INSPECTION), impact_time_offset_s=0.001),
    ),
    ids=("contact-mode", "source-kind", "scenario", "timing"),
)
def test_executor_cannot_substitute_a_run_from_another_config(
    returned_config: SimulationConfig,
) -> None:
    requested = _config(ContactMode.DELIVERY_INSPECTION)
    foreign_run = run_simulation(returned_config)

    with pytest.raises(ContractViolationError, match="different simulation config"):
        capture_simulation(requested, lambda _config: foreign_run)


def test_archive_rejects_executor_run_reordering_before_commit(tmp_path: Path) -> None:
    path = tmp_path / "ensemble"
    configs = (
        _config(ContactMode.DELIVERY_INSPECTION, speed_mph=100.0),
        _config(ContactMode.DELIVERY_INSPECTION, speed_mph=101.0),
    )
    request = _request(configs)
    returned = iter(tuple(run_simulation(config) for config in reversed(configs)))

    with pytest.raises(ContractViolationError, match="different simulation config"):
        run_simulation_ensemble_chunks(
            request,
            DurableEnsembleArchiveSink(path),
            chunk_size=1,
            executor=lambda _config: next(returned),
        )

    assert not (path / "commit.json").exists()
