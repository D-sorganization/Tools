"""Fail-closed qualification tests for the production regional-ground runner."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rate_of_closure.application.regional_ground_execution_job import (
    FlightExecutionInput,
    RegionalGroundExecutionJob,
    build_regional_ground_execution_job,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
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
from shared.python.swing_sim.flight import (
    FlightModelRegistry,
    execute_regional_ground_from_flight,
)
from tests.rate_of_closure.test_regional_ground_execution_job import _job

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _rebuild_with_flight(
    source: RegionalGroundExecutionJob,
    flight: FlightExecutionInput,
) -> RegionalGroundExecutionJob:
    return build_regional_ground_execution_job(
        job_id=source.job_id,
        launch=source.launch.launch,
        flight=flight,
        transfer=source.transfer,
        capture_speed_m_s=source.capture_speed_m_s,
        execution_options=source.execution_options,
        regional_execution_options=source.regional_execution_options,
        variation_request=source.variation_request,
        producer=source.provenance.producer,
        producer_version=source.provenance.producer_version,
        source_revision=source.provenance.source_revision,
    )


def _mismatched_job() -> RegionalGroundExecutionJob:
    source = _job()
    return _rebuild_with_flight(
        source,
        replace(source.flight, trajectory_sha256="0" * 64),
    )


def test_known_profile_rejects_mismatched_recomputed_flight_evidence() -> None:
    job = _mismatched_job()

    with pytest.raises(RegionalGroundProductionPreflightError) as raised:
        preflight_regional_ground_production_job(job)

    error = raised.value
    assert error.reason is ProductionRunnerPreflightReason.FLIGHT_EVIDENCE_MISMATCH
    assert error.model_id == "waterloo_penner"
    assert error.model_version == "tools-core/1.0.0"
    assert "trajectory_sha256" not in str(error)
    assert "result_sha256" not in str(error)


def test_canonical_fixture_is_profile_qualified() -> None:
    flight = preflight_regional_ground_production_job(_job())

    assert flight.trajectory
    assert flight.trajectory[-1].time <= 10.0


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


def test_known_model_with_unregistered_version_is_a_distinct_failure() -> None:
    source = _job()
    changed_flight = replace(source.flight, model_version="tools-core/2.0.0")
    job = build_regional_ground_execution_job(
        job_id="unregistered-version-ground-study",
        launch=source.launch.launch,
        flight=changed_flight,
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

    assert raised.value.reason is (
        ProductionRunnerPreflightReason.FLIGHT_PROFILE_UNREGISTERED
    )


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

    assert raised.value.reason is (
        ProductionRunnerPreflightReason.FLIGHT_SETTINGS_INVALID
    )


def test_solver_failure_is_mapped_without_exposing_internal_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenModel:
        def simulate_to_surface(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("solver-specific secret detail")

    monkeypatch.setattr(
        "rate_of_closure.application.flight_execution_profiles."
        "FlightModelRegistry.get_model",
        lambda _model_type: BrokenModel(),
    )

    with pytest.raises(RegionalGroundProductionPreflightError) as raised:
        preflight_regional_ground_production_job(_job())

    assert raised.value.reason is (
        ProductionRunnerPreflightReason.FLIGHT_RECOMPUTATION_FAILED
    )
    assert "solver-specific secret detail" not in str(raised.value)


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
        "shared.python.swing_sim.flight.regional_ground_pipeline.execute_regional_ground_from_flight",
        forbidden,
    )

    with pytest.raises(GroundRegionalVariationFailed) as raised:
        run_regional_ground_production_job(
            _mismatched_job(), GroundRegionalVariationHooks()
        )

    failure = raised.value
    assert failure.stage is GroundRegionalVariationFailureStage.PREFLIGHT
    assert (failure.completed, failure.total) == (0, 4)
    assert isinstance(failure.__cause__, RegionalGroundProductionPreflightError)
    assert calls == []


def test_authority_manager_publishes_only_generic_preflight_failure() -> None:
    job = _mismatched_job()
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


def test_qualified_job_reuses_one_flight_solve_and_publishes_bound_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    original_get_model = FlightModelRegistry.get_model
    model_resolutions = 0

    def counted_get_model(model_type: object):
        nonlocal model_resolutions
        model_resolutions += 1
        return original_get_model(model_type)

    monkeypatch.setattr(FlightModelRegistry, "get_model", counted_get_model)
    progress: list[tuple[int, int]] = []
    result = run_regional_ground_production_job(
        job,
        GroundRegionalVariationHooks(
            progress_callback=lambda value: progress.append(
                (value.completed, value.total)
            )
        ),
    )

    assert type(result) is RegionalGroundExecutionResult
    result.assert_matches_job(job)
    assert model_resolutions == 1
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert len(result.dataset.rows) == job.execution_options.max_trials


def test_authority_manager_publishes_qualified_complete_result() -> None:
    job = _job()
    manager = AuthorityJobManager(runner=run_regional_ground_production_job)

    manager.submit(job)
    terminal = manager.wait_for_terminal(job.job_id, timeout_s=30.0)

    assert terminal.status is AuthorityJobStatus.SUCCEEDED
    assert (terminal.completed, terminal.total) == (4, 4)
    assert terminal.result_available is True
    result = manager.result(job.job_id)
    result.assert_matches_job(job)


def test_qualified_job_forwards_cancellation_into_physics_and_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    observed_callbacks: list[object] = []
    checks = 0

    def cancellation_requested() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    def observe(*args: object, **kwargs: object):
        observed_callbacks.append(kwargs["options"].is_cancelled)
        return execute_regional_ground_from_flight(*args, **kwargs)

    monkeypatch.setattr(
        "rate_of_closure.web_authority.production_runner."
        "execute_regional_ground_from_flight",
        observe,
    )

    with pytest.raises(GroundRegionalVariationCancelled) as raised:
        run_regional_ground_production_job(
            job,
            GroundRegionalVariationHooks(cancellation_requested=cancellation_requested),
        )

    assert raised.value.total == job.execution_options.max_trials
    assert raised.value.completed < raised.value.total
    assert observed_callbacks
    assert all(callback is cancellation_requested for callback in observed_callbacks)


def test_qualified_executor_failure_retains_typed_counts_and_no_partial_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    cause = RuntimeError("injected regional executor failure")

    def broken(*_args: object, **_kwargs: object) -> object:
        raise cause

    monkeypatch.setattr(
        "rate_of_closure.web_authority.production_runner."
        "execute_regional_ground_from_flight",
        broken,
    )

    with pytest.raises(GroundRegionalVariationFailed) as raised:
        run_regional_ground_production_job(job, GroundRegionalVariationHooks())

    failure = raised.value
    assert failure.stage is GroundRegionalVariationFailureStage.EXECUTOR
    assert (failure.completed, failure.total) == (0, 4)
    assert failure.__cause__ is cause
    assert not hasattr(failure, "result")


def test_qualified_result_binding_failure_is_typed_complete_only_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    cause = ValueError("injected result identity mismatch")

    def reject_result(*_args: object, **_kwargs: object) -> object:
        raise cause

    monkeypatch.setattr(
        "rate_of_closure.web_authority.production_runner."
        "build_regional_ground_execution_result",
        reject_result,
    )

    with pytest.raises(GroundRegionalVariationFailed) as raised:
        run_regional_ground_production_job(job, GroundRegionalVariationHooks())

    failure = raised.value
    assert failure.stage is GroundRegionalVariationFailureStage.PUBLICATION
    assert (failure.completed, failure.total) == (4, 4)
    assert failure.__cause__ is cause
    assert not hasattr(failure, "result")
