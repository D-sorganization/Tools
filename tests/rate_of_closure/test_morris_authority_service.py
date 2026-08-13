"""UI-neutral deterministic Rate Morris service tests."""

from __future__ import annotations

import threading

import pytest

from rate_of_closure.application.morris.contracts import parse_morris_request
from rate_of_closure.application.morris.service import RateMorrisService
from rate_of_closure.variation.simulation_types import ALL_OUTPUT_NAMES
from shared.python.swing_sim.variation import MorrisEvaluation

from .test_morris_authority_contracts import request_document

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def deterministic_evaluator(sample: object) -> MorrisEvaluation:
    """Return deterministic hit-shaped scalar data for service isolation."""
    values = {name: float(sample.ordinal) for name in ALL_OUTPUT_NAMES}
    return MorrisEvaluation("evaluated_hit", values)


def test_service_returns_unchanged_v1_report_deterministically() -> None:
    request = parse_morris_request(request_document())
    service = RateMorrisService(
        evaluator_factory=lambda _design, _config: deterministic_evaluator
    )

    first_report = service.execute(
        request, threading.Event(), lambda _done, _total: None
    )
    second_report = service.execute(
        request, threading.Event(), lambda _done, _total: None
    )
    first = service.execute_with_observations(
        request, threading.Event(), lambda _done, _total: None
    )
    second = service.execute_with_observations(
        request, threading.Event(), lambda _done, _total: None
    )

    assert first_report == second_report == first.report
    assert first.report == second.report
    assert first.report["schema_id"] == "swing-sim/morris-global-sensitivity-report"
    assert first.report["schema_version"] == 1
    assert first.report["design"]["total_samples"] == 4  # type: ignore[index]
    assert first.observations.study_id == request.request_id
    assert first.observations.design_sha256 == second.observations.design_sha256
    assert (
        first.observations.provenance["request_sha256"]
        == second.observations.provenance["request_sha256"]
    )
    assert first.observations.observations.outcomes.shape == (2, 2)


def test_programming_failure_aborts_whole_service_call() -> None:
    request = parse_morris_request(request_document())

    def broken(_sample: object) -> MorrisEvaluation:
        raise TypeError("C:\\private\\source.py leaked")

    service = RateMorrisService(evaluator_factory=lambda _design, _config: broken)
    with pytest.raises(TypeError, match="private"):
        service.execute(request, threading.Event(), lambda _done, _total: None)
