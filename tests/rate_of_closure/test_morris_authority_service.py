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


def test_service_computes_correct_elementary_effects_and_invariants() -> None:
    """Assert mathematical correctness of elementary effects and metric invariants."""
    request = parse_morris_request(request_document())

    # Linear response f(x) = 3.5 * yaw: elementary effect must equal exactly 3.5
    def linear_evaluator(sample: object) -> MorrisEvaluation:
        yaw = getattr(sample, "physical_values", {}).get("yaw", 0.0)
        values = {name: 3.5 * float(yaw) for name in ALL_OUTPUT_NAMES}
        return MorrisEvaluation("evaluated_hit", values)

    service = RateMorrisService(
        evaluator_factory=lambda _design, _config: linear_evaluator
    )
    result = service.execute_with_observations(
        request, threading.Event(), lambda _done, _total: None
    )
    report = result.report
    estimates = report["estimates"]
    assert len(estimates) == len(ALL_OUTPUT_NAMES)

    for estimate in estimates:
        effects = estimate["effects"]
        mu = effects["mu"]
        mu_star = effects["mu_star"]
        sigma = effects["sigma"]
        se = effects["mu_star_standard_error"]

        # Exact mathematical expectation for linear slope 3.5 scaled to factor
        # range [lower, upper] = [-2.0, 2.0]. Range is 4.0, so normalized EE is 14.0.
        expected_mu = 3.5 * 4.0
        assert mu == pytest.approx(expected_mu)
        assert mu_star == pytest.approx(expected_mu)
        assert sigma == pytest.approx(0.0, abs=1e-12)
        assert se == pytest.approx(0.0, abs=1e-12)

        # Invariants
        assert mu_star >= abs(mu)
        assert sigma >= 0.0
        assert se >= 0.0
        assert estimate["availability"] == "available"

    # Constant response f(x) = 42.0: effects must be 0 and availability constant-output
    def constant_evaluator(_sample: object) -> MorrisEvaluation:
        values = {name: 42.0 for name in ALL_OUTPUT_NAMES}
        return MorrisEvaluation("evaluated_hit", values)

    const_service = RateMorrisService(
        evaluator_factory=lambda _design, _config: constant_evaluator
    )
    const_result = const_service.execute_with_observations(
        request, threading.Event(), lambda _done, _total: None
    )
    for estimate in const_result.report["estimates"]:
        assert estimate["availability"] == "constant-output"
        assert estimate["effects"]["mu"] == 0.0
        assert estimate["effects"]["mu_star"] == 0.0
        assert estimate["effects"]["sigma"] == 0.0
        assert estimate["effects"]["mu_star_standard_error"] == 0.0


def test_programming_failure_aborts_whole_service_call() -> None:
    request = parse_morris_request(request_document())

    def broken(_sample: object) -> MorrisEvaluation:
        raise TypeError("C:\\private\\source.py leaked")

    service = RateMorrisService(evaluator_factory=lambda _design, _config: broken)
    with pytest.raises(TypeError, match="private"):
        service.execute(request, threading.Event(), lambda _done, _total: None)
