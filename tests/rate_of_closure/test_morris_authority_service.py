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

    first = service.execute(request, threading.Event(), lambda _done, _total: None)
    second = service.execute(request, threading.Event(), lambda _done, _total: None)

    assert first == second
    assert first["schema_id"] == "swing-sim/morris-global-sensitivity-report"
    assert first["schema_version"] == 1
    assert first["design"]["total_samples"] == 4  # type: ignore[index]


def test_programming_failure_aborts_whole_service_call() -> None:
    request = parse_morris_request(request_document())

    def broken(_sample: object) -> MorrisEvaluation:
        raise TypeError("C:\\private\\source.py leaked")

    service = RateMorrisService(evaluator_factory=lambda _design, _config: broken)
    with pytest.raises(TypeError, match="private"):
        service.execute(request, threading.Event(), lambda _done, _total: None)
