"""Current-editor preparation of qualified regional-ground execution jobs."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from rate_of_closure.application.flight_execution_profiles import (
    FlightExecutionQualificationReason,
    qualify_flight_execution_input,
)
from rate_of_closure.application.regional_ground_job_preparation import (
    DEFAULT_REGIONAL_GROUND_JOB_PREPARATION_PROFILE,
    RegionalGroundJobPreparationProfile,
    prepare_regional_ground_execution_job,
    require_prepared_job_matches_request,
)
from rate_of_closure.application.regional_ground_job_preparation_request import (
    MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES,
    RegionalGroundJobPreparationRequest,
    regional_ground_job_preparation_request_from_json,
    regional_ground_job_preparation_request_to_json,
)
from shared.python.swing_sim.flight import WindScenario
from tests.rate_of_closure.test_regional_ground_execution_job import _job

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_preparation_binds_current_launch_request_and_recomputed_flight() -> None:
    source = _job()

    prepared = prepare_regional_ground_execution_job(
        job_id="editor-ground-study-001",
        launch=source.launch.launch,
        variation_request=source.variation_request,
    )

    assert prepared.job_id == "editor-ground-study-001"
    assert prepared.launch.launch == source.launch.launch
    assert prepared.variation_request == source.variation_request
    assert (
        prepared.transfer.surface == source.variation_request.regional_plan.base_surface
    )
    assert prepared.execution_options.max_trials == source.variation_request.plan.n_runs
    assert prepared.flight.model_id == source.variation_request.plan.flight_model
    assert prepared.provenance.source_revision == (
        DEFAULT_REGIONAL_GROUND_JOB_PREPARATION_PROFILE.source_revision
    )
    evidence = qualify_flight_execution_input(
        prepared.launch.launch,
        prepared.transfer,
        prepared.flight,
    )
    assert evidence.reason is FlightExecutionQualificationReason.QUALIFIED
    assert prepared.input_sha256 == prepared.expected_input_sha256
    assert prepared.job_sha256 == prepared.expected_job_sha256


def test_preparation_is_deterministic_for_the_same_exact_authorities() -> None:
    source = _job()

    first = prepare_regional_ground_execution_job(
        job_id="editor-ground-study-001",
        launch=source.launch.launch,
        variation_request=source.variation_request,
    )
    second = prepare_regional_ground_execution_job(
        job_id="editor-ground-study-001",
        launch=source.launch.launch,
        variation_request=source.variation_request,
    )

    assert first == second


def test_preparation_rejects_an_unsupported_editor_flight_model() -> None:
    source = _job()
    incompatible = replace(
        source.variation_request,
        plan=replace(source.variation_request.plan, flight_model="unregistered-model"),
    )

    with pytest.raises(ValueError, match="flight model"):
        prepare_regional_ground_execution_job(
            job_id="editor-ground-study-001",
            launch=source.launch.launch,
            variation_request=incompatible,
        )


def test_preparation_rejects_unresolved_variable_wind() -> None:
    source = _job()
    unresolved = replace(source.launch.launch, wind_scenario=WindScenario())

    with pytest.raises(ValueError, match="resolved constant wind"):
        prepare_regional_ground_execution_job(
            job_id="editor-ground-study-001",
            launch=unresolved,
            variation_request=source.variation_request,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", " "),
        ("model_version", ""),
        ("flight_max_time_s", math.inf),
        ("flight_step_s", 0.0),
        ("flight_sample_every", 0),
        ("transfer_max_time_s", -1.0),
        ("transfer_output_interval_s", math.nan),
        ("transfer_max_events", 0),
        ("rotational_inertia_factor", 1.1),
        ("capture_speed_m_s", 0.0),
        ("source_revision", " revision "),
    ],
)
def test_preparation_profile_rejects_invalid_numerical_and_identity_bounds(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RegionalGroundJobPreparationProfile(**{field: value})  # type: ignore[arg-type]


def test_prepared_job_postcondition_rejects_substituted_authority() -> None:
    source = _job()

    with pytest.raises(ValueError, match="job_id"):
        require_prepared_job_matches_request(
            source,
            job_id="different-job-id",
            launch=source.launch.launch,
            variation_request=source.variation_request,
        )


def test_preparation_request_round_trips_canonical_exact_editor_state() -> None:
    source = _job()
    request = RegionalGroundJobPreparationRequest(
        "editor-ground-study-001", source.launch, source.variation_request
    )

    text = regional_ground_job_preparation_request_to_json(request)

    assert regional_ground_job_preparation_request_from_json(text) == request
    assert (
        regional_ground_job_preparation_request_to_json(
            regional_ground_job_preparation_request_from_json(text)
        )
        == text
    )


def test_preparation_request_rejects_duplicate_and_multibyte_oversize_json() -> None:
    source = _job()
    request = RegionalGroundJobPreparationRequest(
        "editor-ground-study-001", source.launch, source.variation_request
    )
    duplicate = regional_ground_job_preparation_request_to_json(request).replace(
        '"job_id":', '"job_id":"duplicate","job_id":', 1
    )

    with pytest.raises(ValueError, match="duplicate"):
        regional_ground_job_preparation_request_from_json(duplicate)
    with pytest.raises(ValueError, match="maximum wire size"):
        regional_ground_job_preparation_request_from_json(
            "é" * (MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES // 2 + 1)
        )
