"""Strict versioned flight execution-profile qualification tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rate_of_closure.application.flight_execution_profiles import (
    FLIGHT_EXECUTION_PROFILE_REGISTRY_SCHEMA_VERSION,
    FlightExecutionProfileQualificationError,
    FlightExecutionQualificationReason,
    build_qualified_flight_execution_input,
    qualify_flight_execution_input,
    recompute_qualified_flight_result,
    registered_flight_execution_profiles,
)
from rate_of_closure.application.regional_ground_execution_job import (
    FlightExecutionInput,
    canonical_flight_result_sha256,
    canonical_flight_trajectory_sha256,
)
from shared.python.swing_sim.flight import FlightModelRegistry, FlightResult
from tests.rate_of_closure.test_regional_ground_execution_job import _job

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_registry_exposes_one_exact_versioned_profile_and_settings_schema() -> None:
    profiles = registered_flight_execution_profiles()

    assert len(profiles) == 1
    assert profiles[0].schema_version == (
        FLIGHT_EXECUTION_PROFILE_REGISTRY_SCHEMA_VERSION
    )
    assert profiles[0].model_id == "waterloo_penner"
    assert profiles[0].model_version == "tools-core/1.0.0"
    assert profiles[0].setting_ids == (
        "max_time_s",
        "sample_every",
        "step_s",
    )
    assert profiles[0].recomputation_contract == (
        "waterloo-penner-adaptive-rk45-planar-contact/v1"
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"max_time_s": 10.0, "step_s": 0.001},
        {
            "max_time_s": 10.0,
            "sample_every": 10.0,
            "step_s": 0.001,
            "extra": 1.0,
        },
        {"max_time_s": 10.0, "sample_every": 1.5, "step_s": 0.001},
        {"max_time_s": 121.0, "sample_every": 10.0, "step_s": 0.001},
        {"max_time_s": 10.0, "sample_every": 10.0, "step_s": 0.0},
    ],
)
def test_schema_mismatch_fails_before_model_resolution(
    settings: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job()
    flight = FlightExecutionInput(
        job.flight.model_id,
        job.flight.model_version,
        settings,
        job.flight.trajectory_sha256,
        job.flight.result_sha256,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid settings must not resolve or run a model")

    monkeypatch.setattr(FlightModelRegistry, "get_model", forbidden)
    evidence = qualify_flight_execution_input(job.launch.launch, job.transfer, flight)

    assert evidence.qualified is False
    assert evidence.reason is FlightExecutionQualificationReason.SETTINGS_SCHEMA_INVALID
    assert evidence.recomputed_trajectory_sha256 is None
    assert evidence.recomputed_result_sha256 is None


def test_unregistered_exact_identity_fails_before_model_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    flight = replace(job.flight, model_version="tools-core/2.0.0")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unregistered identity must not resolve or run a model")

    monkeypatch.setattr(FlightModelRegistry, "get_model", forbidden)
    evidence = qualify_flight_execution_input(job.launch.launch, job.transfer, flight)

    assert evidence.reason is FlightExecutionQualificationReason.PROFILE_NOT_REGISTERED
    assert evidence.qualified is False


def test_model_failure_becomes_typed_non_digest_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()

    class BrokenModel:
        def simulate_to_surface(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("solver-specific secret detail")

    monkeypatch.setattr(
        FlightModelRegistry,
        "get_model",
        lambda _model_type: BrokenModel(),
    )
    evidence = qualify_flight_execution_input(
        job.launch.launch,
        job.transfer,
        job.flight,
    )

    assert evidence.reason is FlightExecutionQualificationReason.RECOMPUTATION_FAILED
    assert evidence.qualified is False
    assert evidence.recomputed_trajectory_sha256 is None
    assert evidence.recomputed_result_sha256 is None


def test_current_fixture_recomputes_deterministically_and_qualifies() -> None:
    job = _job()

    first = qualify_flight_execution_input(
        job.launch.launch,
        job.transfer,
        job.flight,
    )
    second = qualify_flight_execution_input(
        job.launch.launch,
        job.transfer,
        job.flight,
    )

    assert first == second
    assert first.reason is FlightExecutionQualificationReason.QUALIFIED
    assert first.qualified is True
    assert first.recomputed_trajectory_sha256 is not None
    assert first.recomputed_result_sha256 is not None
    assert first.recomputed_trajectory_sha256 == job.flight.trajectory_sha256
    assert first.recomputed_result_sha256 == job.flight.result_sha256


def test_qualified_boundary_returns_only_an_exact_digest_matched_result() -> None:
    job = _job()
    observed = qualify_flight_execution_input(
        job.launch.launch,
        job.transfer,
        job.flight,
    )
    assert observed.recomputed_trajectory_sha256 is not None
    assert observed.recomputed_result_sha256 is not None
    qualified_input = replace(
        job.flight,
        trajectory_sha256=observed.recomputed_trajectory_sha256,
        result_sha256=observed.recomputed_result_sha256,
    )

    result = recompute_qualified_flight_result(
        job.launch.launch,
        job.transfer,
        qualified_input,
    )

    assert type(result) is FlightResult
    assert canonical_flight_trajectory_sha256(result) == (
        qualified_input.trajectory_sha256
    )
    assert canonical_flight_result_sha256(result) == qualified_input.result_sha256


def test_builder_returns_a_fresh_digest_bound_registered_input() -> None:
    job = _job()

    built = build_qualified_flight_execution_input(
        job.launch.launch,
        job.transfer,
        model_id=job.flight.model_id,
        model_version=job.flight.model_version,
        settings=job.flight.settings,
    )

    evidence = qualify_flight_execution_input(job.launch.launch, job.transfer, built)
    assert evidence.reason is FlightExecutionQualificationReason.QUALIFIED
    assert built.trajectory_sha256 == evidence.recomputed_trajectory_sha256
    assert built.result_sha256 == evidence.recomputed_result_sha256


def test_unqualified_boundary_raises_typed_evidence_without_digest_text() -> None:
    job = _job()
    mismatched = replace(job.flight, trajectory_sha256="0" * 64)

    with pytest.raises(FlightExecutionProfileQualificationError) as raised:
        recompute_qualified_flight_result(
            job.launch.launch,
            job.transfer,
            mismatched,
        )

    assert raised.value.qualification.reason is (
        FlightExecutionQualificationReason.TRAJECTORY_DIGEST_MISMATCH
    )
    assert mismatched.trajectory_sha256 not in str(raised.value)
    assert mismatched.result_sha256 not in str(raised.value)
