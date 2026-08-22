"""Execution, inspection, and exact-resume tests for the ensemble authority."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from rate_of_closure.application.durable_ensemble import (
    RateDurableEnsembleService,
    durable_ensemble_request_document,
    parse_durable_ensemble_request,
)
from rate_of_closure.simulation import SimulationConfig, SimulationRun, run_simulation
from shared.python.swing_sim.solver.solve import CancelledError

from .test_durable_ensemble_authority_contracts import _plan
from .test_variation_simulation_request import _base_config

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _request(chunk_size: int = 1):  # type: ignore[no-untyped-def]
    document = durable_ensemble_request_document(
        "request-41",
        "campaign-41",
        _plan(),
        _base_config(),
        chunk_size=chunk_size,
    )
    return parse_durable_ensemble_request(document)


def test_completed_replay_executes_zero_trials(tmp_path: Path) -> None:
    executions: list[SimulationConfig] = []

    def execute(config: SimulationConfig) -> SimulationRun:
        executions.append(config)
        return run_simulation(config)

    service = RateDurableEnsembleService(tmp_path, execute)
    first = service.execute(_request(2), threading.Event(), lambda _item: None)
    assert first.archive.status == "complete"
    assert len(executions) == 3

    executions.clear()
    second = service.execute(_request(1), threading.Event(), lambda _item: None)

    assert second == first
    assert executions == []


def test_cancelled_prefix_is_inspectable_and_exactly_resumes(tmp_path: Path) -> None:
    cancel = threading.Event()
    prefixes: list[int] = []
    service = RateDurableEnsembleService(tmp_path)

    def stop_after_first(evidence):  # type: ignore[no-untyped-def]
        prefixes.append(evidence.archive.analyzed_trial_count)
        cancel.set()

    with pytest.raises(CancelledError):
        service.execute(_request(), cancel, stop_after_first)

    retained = service.inspect(_request())
    assert prefixes == [1]
    assert retained.archive.status == "in_progress"
    assert retained.archive.analyzed_trial_count == 1

    resumed = service.execute(_request(2), threading.Event(), lambda _item: None)
    assert resumed.archive.status == "complete"
    assert resumed.archive.analyzed_trial_count == 3
