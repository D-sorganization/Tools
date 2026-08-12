"""Fail-closed qualification tests for the production regional-ground runner."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rate_of_closure.application.regional_ground_execution_job import (
    FlightExecutionInput,
    build_regional_ground_execution_job,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
    GroundRegionalVariationHooks,
)
from rate_of_closure.web_authority.jobs import (
    AuthorityJobManager,
    AuthorityJobResultUnavailable,
    AuthorityJobStatus,
)
from rate_of_closure.web_authority.production_runner import (
    ProductionRunnerPreflightReason,
    RegionalGroundProductionPreflightError,
    preflight_regional_ground_production_job,
    run_regional_ground_production_job,
)
from tests.rate_of_closure.test_regional_ground_execution_job import _job

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_known_model_still_rejects_unregistered_versioned_execution_profile() -> None:
    job = _job()

    with pytest.raises(RegionalGroundProductionPreflightError) as raised:
        preflight_regional_ground_production_job(job)

    error = raised.value
    assert error.reason is ProductionRunnerPreflightReason.FLIGHT_PROFILE_UNREGISTERED
    assert error.model_id == "waterloo_penner"
    assert error.model_version == "tools-core/1.0.0"
    assert "trajectory_sha256" not in str(error)
    assert "result_sha256" not in str(error)


def test_unknown_model_identity_is_a_distinct_preflight_failure() -> None:
    source = _job()
    changed_flight = replace(source.flight, model_id="unknown-flight-authority")
    changed_request = replace(
        source.variation_request,
        plan=replace(
            source.variation_request.plan,
            flight_model="unknown-flight-authority",
        ),
    )
    job = build_regional_ground_execution_job(
        job_id="unknown-model-ground-study",
        launch=source.launch.launch,
        flight=changed_flight,
        transfer=source.transfer,
        capture_speed_m_s=source.capture_speed_m_s,
        execution_options=source.execution_options,
        regional_execution_options=source.regional_execution_options,
        variation_request=changed_request,
        producer=source.provenance.producer,
        producer_version=source.provenance.producer_version,
        source_revision=source.provenance.source_revision,
    )

    with pytest.raises(RegionalGroundProductionPreflightError) as raised:
        preflight_regional_ground_production_job(job)

    assert raised.value.reason is ProductionRunnerPreflightReason.FLIGHT_MODEL_UNKNOWN


def test_generic_flight_settings_contract_does_not_imply_executable_semantics() -> None:
    source = _job()
    arbitrary = FlightExecutionInput(
        model_id=source.flight.model_id,
        model_version=source.flight.model_version,
        settings={"arbitrary_numeric_setting": 17.0},
        trajectory_sha256=source.flight.trajectory_sha256,
        result_sha256=source.flight.result_sha256,
    )
    job = build_regional_ground_execution_job(
        job_id="arbitrary-flight-settings-study",
        launch=source.launch.launch,
        flight=arbitrary,
        transfer=source.transfer,
        capture_speed_m_s=source.capture_speed_m_s,
        execution_options=source.execution_options,
        regional_execution_options=source.regional_execution_options,
        variation_request=source.variation_request,
        producer=source.provenance.producer,
        producer_version=source.provenance.producer_version,
        source_revision=source.provenance.source_revision,
    )

    with pytest.raises(RegionalGroundProductionPreflightError) as raised:
        preflight_regional_ground_production_job(job)

    assert (
        raised.value.reason
        is ProductionRunnerPreflightReason.FLIGHT_PROFILE_UNREGISTERED
    )


def test_precancel_wins_before_preflight_or_any_physics() -> None:
    hooks = GroundRegionalVariationHooks(cancellation_requested=lambda: True)

    with pytest.raises(GroundRegionalVariationCancelled) as raised:
        run_regional_ground_production_job(_job(), hooks)

    assert (raised.value.completed, raised.value.total) == (0, 4)


def test_cancellation_callback_defect_remains_typed_and_chained() -> None:
    cause = RuntimeError("cancellation authority unavailable")

    def broken_cancellation() -> bool:
        raise cause

    hooks = GroundRegionalVariationHooks(cancellation_requested=broken_cancellation)
    with pytest.raises(GroundRegionalVariationFailed) as raised:
        run_regional_ground_production_job(_job(), hooks)

    failure = raised.value
    assert failure.stage is GroundRegionalVariationFailureStage.CANCELLATION_CALLBACK
    assert (failure.completed, failure.total) == (0, 4)
    assert failure.__cause__ is cause


def test_preflight_failure_is_typed_and_runs_no_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        calls.append("physics")
        raise AssertionError("physics must not run before profile qualification")

    monkeypatch.setattr(
        "shared.python.swing_sim.flight.pipeline.simulate",
        forbidden,
    )
    monkeypatch.setattr(
        "shared.python.swing_sim.flight.regional_ground_pipeline.execute_regional_ground_from_flight",
        forbidden,
    )

    with pytest.raises(GroundRegionalVariationFailed) as raised:
        run_regional_ground_production_job(_job(), GroundRegionalVariationHooks())

    failure = raised.value
    assert failure.stage is GroundRegionalVariationFailureStage.PREFLIGHT
    assert (failure.completed, failure.total) == (0, 4)
    assert isinstance(failure.__cause__, RegionalGroundProductionPreflightError)
    assert calls == []


def test_authority_manager_publishes_only_generic_preflight_failure() -> None:
    job = _job()
    manager = AuthorityJobManager(runner=run_regional_ground_production_job)

    manager.submit(job)
    terminal = manager.wait_for_terminal(job.job_id, timeout_s=2.0)

    assert terminal.status is AuthorityJobStatus.FAILED
    assert terminal.completed == 0
    assert terminal.result_available is False
    assert terminal.failure is not None
    assert terminal.failure.to_wire() == {
        "code": "execution_failed",
        "stage": "preflight",
    }
    with pytest.raises(AuthorityJobResultUnavailable):
        manager.result(job.job_id)
