"""Contract tests for flight-through-regional-ground composition."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, cast

import pytest

import shared.python.swing_sim.flight.regional_ground_pipeline as subject
from shared.python.swing_sim.flight import (
    FLIGHT_REGIONAL_GROUND_PIPELINE_CONTRACT_VERSION,
    FlightGroundTransferError,
    FlightGroundTransferSettings,
    FlightRegionalGroundPipelineResult,
    FlightResult,
    LaunchConditions,
    execute_regional_ground_from_flight,
)
from shared.python.swing_sim.ground import (
    BounceTerminationReason,
    GroundRegionalMaterialPlanRequest,
    GroundResultStatus,
    RegionalGroundExecutionOptions,
    RegionalGroundExecutionStatus,
)

from ._regional_ground_pipeline_support import (
    _crossing_result,
    _empty_termination_pair,
    _launch,
    _no_contact_result,
    _plan,
    _settings,
    _time_limit_pair,
)


def test_pipeline_composes_authoritative_phases_and_preserves_identity() -> None:
    plan = _plan()

    result = execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        plan,
        capture_speed_m_s=3.0,
    )

    assert type(result) is FlightRegionalGroundPipelineResult
    assert result.contract_version == FLIGHT_REGIONAL_GROUND_PIPELINE_CONTRACT_VERSION
    assert result.regional_plan is plan
    assert result.regional_result is not None
    assert result.ground_result is not None
    assert result.ground_result.status is GroundResultStatus.COMPLETE
    assert result.regional_result.status is RegionalGroundExecutionStatus.COMPLETE
    assert (
        result.bounce_result.result.termination.reason
        is BounceTerminationReason.SETTLED_TO_SKID
    )
    assert result.ground_request_sha256 == (
        result.bounce_result.request.ground_request_sha256
    )
    assert result.repeated_bounce_execution_input_sha256 == (
        result.bounce_result.execution_input_sha256
    )
    assert (
        result.regional_plan_sha256
        == hashlib.sha256(plan.to_json().encode("utf-8")).hexdigest()
    )
    assert result.regional_result.regional_plan_sha256 == (result.regional_plan_sha256)
    assert result.regional_result.plan_provenance == plan.provenance
    assert result.ground_result.provenance == _settings().provenance
    assert result.ground_result.summary is not None
    assert result.ground_result.summary.total_distance_m > 0.0


def test_pipeline_validates_plan_base_before_bounce_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_bounce(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("bounce physics must not run")

    monkeypatch.setattr(
        subject, "execute_repeated_bounce_from_flight", forbidden_bounce
    )
    valid_plan = _plan()
    wrong_base = replace(valid_plan.base_surface, surface_id="wrong-base")
    mismatched_plan = replace(valid_plan, base_surface=wrong_base)

    with pytest.raises(ValueError, match="launch-relative transfer surface"):
        execute_regional_ground_from_flight(
            _crossing_result(),
            _launch(),
            _settings(),
            mismatched_plan,
        )
    with pytest.raises(ValueError, match="capture_speed_m_s"):
        execute_regional_ground_from_flight(
            _crossing_result(),
            _launch(),
            _settings(),
            valid_plan,
            capture_speed_m_s=0.0,
        )
    assert calls == 0


def test_pipeline_requires_exact_inputs_before_bounce_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_bounce(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("bounce physics must not run")

    monkeypatch.setattr(
        subject, "execute_repeated_bounce_from_flight", forbidden_bounce
    )
    flight = _crossing_result()
    launch = _launch()
    transfer = _settings()
    plan = _plan()
    invalid_calls = (
        (cast(FlightResult, object()), launch, transfer, plan, None, "FlightResult"),
        (
            flight,
            cast(LaunchConditions, object()),
            transfer,
            plan,
            None,
            "LaunchConditions",
        ),
        (
            flight,
            launch,
            cast(FlightGroundTransferSettings, object()),
            plan,
            None,
            "FlightGroundTransferSettings",
        ),
        (
            flight,
            launch,
            transfer,
            cast(GroundRegionalMaterialPlanRequest, object()),
            None,
            "GroundRegionalMaterialPlanRequest",
        ),
        (
            flight,
            launch,
            transfer,
            plan,
            cast(RegionalGroundExecutionOptions, object()),
            "RegionalGroundExecutionOptions",
        ),
    )
    for (
        bad_flight,
        bad_launch,
        bad_transfer,
        bad_plan,
        options,
        message,
    ) in invalid_calls:
        with pytest.raises(ValueError, match=f"exact {message}"):
            execute_regional_ground_from_flight(
                bad_flight,
                bad_launch,
                bad_transfer,
                bad_plan,
                options=options,
            )
    assert calls == 0


def test_preflight_cancellation_does_not_invoke_regional_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_regional(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("regional physics must not run")

    monkeypatch.setattr(subject, "execute_regional_ground", forbidden_regional)
    plan = _plan()

    result = execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        plan,
        options=RegionalGroundExecutionOptions(is_cancelled=lambda: True),
    )

    assert result.regional_result is None
    assert result.ground_result is None
    assert result.regional_plan is plan
    assert (
        result.bounce_result.result.termination.reason
        is BounceTerminationReason.CANCELLED
    )
    assert calls == 0


@pytest.mark.parametrize(
    "reason",
    [
        BounceTerminationReason.CANCELLED,
        BounceTerminationReason.TIME_LIMIT,
        BounceTerminationReason.EVENT_LIMIT,
        BounceTerminationReason.NO_RECONTACT,
        BounceTerminationReason.NUMERICAL_FAILURE,
    ],
)
def test_every_nonsettled_bounce_outcome_skips_regional_execution(
    monkeypatch: pytest.MonkeyPatch,
    reason: BounceTerminationReason,
) -> None:
    pair = (
        _time_limit_pair()
        if reason is BounceTerminationReason.TIME_LIMIT
        else _empty_termination_pair(reason)
    )
    regional_calls = 0

    def supplied_bounce(*_args: object, **_kwargs: object) -> object:
        return pair

    def forbidden_regional(*_args: object, **_kwargs: object) -> None:
        nonlocal regional_calls
        regional_calls += 1
        raise AssertionError("regional physics must not run")

    monkeypatch.setattr(subject, "execute_repeated_bounce_from_flight", supplied_bounce)
    monkeypatch.setattr(subject, "execute_regional_ground", forbidden_regional)
    plan = _plan()

    result = execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        plan,
    )

    assert result.bounce_result is pair
    assert result.bounce_result.result.termination.reason is reason
    assert result.regional_result is None
    assert result.ground_result is None
    assert result.regional_plan is plan
    assert result.ground_request_sha256 == pair.request.ground_request_sha256
    assert regional_calls == 0


def test_transfer_failure_does_not_invoke_regional_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_regional(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("regional physics must not run")

    monkeypatch.setattr(subject, "execute_regional_ground", forbidden_regional)
    with pytest.raises(FlightGroundTransferError):
        execute_regional_ground_from_flight(
            _no_contact_result(),
            _launch(),
            _settings(),
            _plan(),
        )
    assert calls == 0


def test_pipeline_result_rejects_fabricated_phase_combinations() -> None:
    plan = _plan()
    complete = execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        plan,
        capture_speed_m_s=3.0,
    )
    cancelled = execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        plan,
        options=RegionalGroundExecutionOptions(is_cancelled=lambda: True),
    )

    with pytest.raises(ValueError, match="settled bounce requires regional result"):
        replace(complete, regional_result=None)
    with pytest.raises(ValueError, match="non-settled bounce forbids regional result"):
        replace(cancelled, regional_result=complete.regional_result)
    with pytest.raises(ValueError, match="ground_request_sha256"):
        replace(complete, ground_request_sha256="c" * 64)
    with pytest.raises(ValueError, match="contract_version"):
        replace(complete, contract_version="unsupported/v2")


def test_pipeline_result_requires_exact_nested_records() -> None:
    complete = execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        _plan(),
        capture_speed_m_s=3.0,
    )

    with pytest.raises(ValueError, match="exact RepeatedBounceRequestResultPair"):
        replace(complete, bounce_result=cast(Any, object()))
    with pytest.raises(ValueError, match="exact GroundRegionalMaterialPlanRequest"):
        replace(complete, regional_plan=cast(Any, object()))
