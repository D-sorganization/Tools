"""Bounded asynchronous durable-ensemble job lifecycle tests."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from rate_of_closure.application.durable_ensemble import (
    DurableEnsembleAuthorityRequest,
    DurableEnsembleJobRegistry,
    DurableEnsembleRegistryOptions,
    EvidenceSink,
    RateDurableEnsembleService,
)
from rate_of_closure.simulation import SimulationConfig, SimulationRun, run_simulation
from rate_of_closure.variation import DurableEnsembleEvidence

from .test_durable_ensemble_authority_service import _request

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _wait(
    registry: DurableEnsembleJobRegistry,
    job_id: str,
    statuses: set[str],
):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        envelope = registry.status(job_id)
        if envelope.status in statuses:
            return envelope
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {statuses}")


def test_registry_reports_incremental_evidence_and_retains_cancelled_prefix(
    tmp_path: Path,
) -> None:
    second_started = threading.Event()
    release = threading.Event()
    calls = 0

    def blocking_executor(config: SimulationConfig) -> SimulationRun:
        nonlocal calls
        calls += 1
        if calls == 2:
            second_started.set()
            assert release.wait(10.0)
        return run_simulation(config)

    registry = DurableEnsembleJobRegistry(
        RateDurableEnsembleService(tmp_path, blocking_executor),
        DurableEnsembleRegistryOptions(max_active_jobs=1),
    )
    try:
        created = registry.create(_request())
        assert second_started.wait(10.0)
        running = registry.status(created.job_id)
        assert running.status == "running"
        assert running.completed_trials == 1
        assert running.evidence is not None

        registry.cancel(created.job_id)
        release.set()
        cancelled = _wait(registry, created.job_id, {"cancelled"})
        assert cancelled.completed_trials == 1
        assert cancelled.evidence is not None
        assert cancelled.evidence.archive.status == "in_progress"
    finally:
        release.set()
        registry.close()


def test_registry_rejects_two_active_writers_for_one_archive(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_executor(config: SimulationConfig) -> SimulationRun:
        entered.set()
        assert release.wait(10.0)
        return run_simulation(config)

    registry = DurableEnsembleJobRegistry(
        RateDurableEnsembleService(tmp_path, blocking_executor)
    )
    try:
        registry.create(_request())
        assert entered.wait(10.0)
        with pytest.raises(FileExistsError, match="active writer"):
            registry.create(_request())
    finally:
        release.set()
        registry.close()


def test_failed_job_returns_sanitized_error(tmp_path: Path) -> None:
    class RejectingService:
        def execute(
            self,
            _request: DurableEnsembleAuthorityRequest,
            _cancel: threading.Event,
            _progress: EvidenceSink,
        ) -> DurableEnsembleEvidence:
            raise RuntimeError("C:/private/archive/internal-token")

        def inspect(
            self, _request: DurableEnsembleAuthorityRequest
        ) -> DurableEnsembleEvidence:
            raise AssertionError("failed work must not be promoted")

    registry = DurableEnsembleJobRegistry(RejectingService())
    try:
        created = registry.create(_request())
        failed = _wait(registry, created.job_id, {"failed"})
        assert failed.error == "durable ensemble execution failed"
        assert "private" not in str(failed.to_json_dict())
    finally:
        registry.close()
