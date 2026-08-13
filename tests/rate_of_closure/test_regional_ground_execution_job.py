"""Strict execution-job contract tests for seeded regional-ground studies."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from rate_of_closure.application.regional_ground_execution_job import (
    MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
    FlightExecutionInput,
    GroundExecutionOptions,
    RegionalGroundExecutionJob,
    build_regional_ground_execution_job,
    canonical_flight_result_sha256,
    canonical_flight_trajectory_sha256,
    regional_ground_execution_job_from_json,
    regional_ground_execution_job_to_json,
)
from rate_of_closure.application.regional_ground_variation_request import (
    regional_ground_variation_request_from_json,
)
from shared.python.swing_sim.ball_setup import BallSetup, BallSupportMode
from shared.python.swing_sim.flight.tests._regional_ground_pipeline_support import (
    _crossing_result,
    _launch,
    _settings,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_FIXTURE = (
    Path(__file__).parents[2]
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "regional_ground_execution_job_golden_v1.json"
)


def _job() -> RegionalGroundExecutionJob:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return regional_ground_execution_job_from_json(
        json.dumps(fixture["job"], separators=(",", ":"), sort_keys=True)
    )


def test_shared_golden_round_trip_and_digest_parity() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    job = _job()
    text = regional_ground_execution_job_to_json(job)

    assert json.loads(text) == fixture["job"]
    assert job.input_sha256 == fixture["input_sha256"]
    assert job.job_sha256 == fixture["job_sha256"]
    assert job.canonical_sha256 == fixture["canonical_sha256"]
    assert regional_ground_execution_job_from_json(text) == job


def test_flight_digests_bind_every_canonical_result_field() -> None:
    result = _crossing_result()
    trajectory_digest = canonical_flight_trajectory_sha256(result)
    result_digest = canonical_flight_result_sha256(result)

    assert len(trajectory_digest) == 64
    assert len(result_digest) == 64
    assert result_digest != trajectory_digest
    changed = replace(result, landing_angle=1.25)
    assert canonical_flight_trajectory_sha256(changed) == trajectory_digest
    assert canonical_flight_result_sha256(changed) != result_digest


def test_builder_binds_exact_launch_setup_transfer_and_existing_request() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    request = regional_ground_variation_request_from_json(
        json.dumps(fixture["job"]["variation_request"], separators=(",", ":"))
    )
    launch = replace(
        _launch(),
        ball_setup=BallSetup(BallSupportMode.TEE, 0.0381),
        spin_rate=2500.0,
        spin_axis=(0.0, -1.0, 0.0),
    )
    result = _crossing_result()
    flight = FlightExecutionInput(
        model_id="waterloo_penner",
        model_version="tools-core/1.0.0",
        settings={"max_time_s": 10.0, "sample_every": 10.0, "step_s": 0.001},
        trajectory_sha256=canonical_flight_trajectory_sha256(result),
        result_sha256=canonical_flight_result_sha256(result),
    )
    job = build_regional_ground_execution_job(
        job_id="driver-ground-study-1729",
        launch=launch,
        flight=flight,
        transfer=_settings(),
        capture_speed_m_s=0.05,
        execution_options=GroundExecutionOptions(4, 2, 120.0, False),
        variation_request=request,
        producer="tools.rate_of_closure",
        producer_version="1.0.0",
        source_revision="fixture-4369",
    )

    assert job.launch.ball_setup == launch.ball_setup
    assert job.transfer.surface == _settings().surface
    assert job.variation_request == request
    assert job.provenance.input_sha256 == job.input_sha256
    assert job.job_sha256 == job.expected_job_sha256


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("schema_version",), "regional-ground-execution-job/v2", "schema_version"),
        (("unit_system",), "US", "unit_system"),
        (("execution_options", "max_trials"), True, "max_trials"),
        (("capture_speed_m_s",), True, "capture_speed"),
        (("launch", "ball_speed_m_s"), math.nan, "finite"),
        (("flight", "settings", "step_s"), math.inf, "finite"),
        (("flight", "settings", "unknown"), 9_007_199_254_740_992, "safe"),
        (("flight", "trajectory_sha256"), "A" * 64, "lowercase"),
        (("variation_request", "max_rows"), True, "max_rows"),
    ],
)
def test_nested_invalid_values_fail_closed(
    path: tuple[str, ...], value: object, message: str
) -> None:
    payload = json.loads(regional_ground_execution_job_to_json(_job()))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises((TypeError, ValueError), match=message):
        regional_ground_execution_job_from_json(
            json.dumps(payload, allow_nan=True, separators=(",", ":"))
        )


@pytest.mark.parametrize(
    "field",
    [
        "job_id",
        "launch",
        "flight",
        "transfer",
        "capture_speed_m_s",
        "execution_options",
        "variation_request",
        "provenance",
        "input_sha256",
    ],
)
def test_tampering_with_any_bound_input_rejects_job_digest(field: str) -> None:
    payload = json.loads(regional_ground_execution_job_to_json(_job()))
    if field == "job_id":
        payload[field] += "-changed"
    elif field == "capture_speed_m_s":
        payload[field] = 0.06
    elif field in {"input_sha256"}:
        payload[field] = "0" * 64
    elif field == "provenance":
        payload[field]["source_revision"] = "changed"
    elif field == "launch":
        payload[field]["spin_rate_rpm"] += 1.0
    elif field == "flight":
        payload[field]["settings"]["step_s"] = 0.002
    elif field == "transfer":
        payload[field]["max_events"] += 1
    elif field == "execution_options":
        payload[field]["timeout_s"] += 1.0
    else:
        payload[field]["result_id"] += "-changed"

    with pytest.raises(ValueError, match="sha256|digest|authority"):
        regional_ground_execution_job_from_json(json.dumps(payload))


def test_execution_bounds_and_cross_contract_invariants_fail_closed() -> None:
    payload = json.loads(regional_ground_execution_job_to_json(_job()))
    cases = (
        ("max_trials", 5, "n_runs"),
        ("max_parallelism", 33, "parallelism"),
        ("timeout_s", 3600.01, "timeout"),
    )
    for field, value, message in cases:
        changed = json.loads(json.dumps(payload))
        changed["execution_options"][field] = value
        with pytest.raises((TypeError, ValueError), match=message):
            regional_ground_execution_job_from_json(json.dumps(changed))

    mismatch = json.loads(json.dumps(payload))
    mismatch["transfer"]["surface"]["surface_id"] = "different-surface"
    with pytest.raises(ValueError, match="surface|sha256|digest"):
        regional_ground_execution_job_from_json(json.dumps(mismatch))


def test_model_and_launch_relative_surface_authorities_must_align() -> None:
    job = _job()
    mismatched_model = replace(job.flight, model_id="nathan")
    with pytest.raises(ValueError, match="model_id.*flight_model"):
        replace(job, flight=mismatched_model)

    shifted_surface = replace(job.transfer.surface, height_m=0.01)
    shifted_transfer = replace(job.transfer, surface=shifted_surface)
    with pytest.raises(ValueError, match="launch-relative transfer surface"):
        replace(job, transfer=shifted_transfer)


@pytest.mark.parametrize(
    "text,message",
    [
        ('{"schema_version":"one","schema_version":"two"}', "duplicate"),
        ('{"value":NaN}', "finite"),
        ('{"value":"\\ud800"}', "surrogate"),
    ],
)
def test_json_safety_failures_are_rejected(text: str, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        regional_ground_execution_job_from_json(text)


def test_extra_fields_and_wire_size_fail_closed() -> None:
    payload = json.loads(regional_ground_execution_job_to_json(_job()))
    payload["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        regional_ground_execution_job_from_json(json.dumps(payload))

    oversized = "é" * (MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES // 2 + 1)
    with pytest.raises(ValueError, match="maximum wire size"):
        regional_ground_execution_job_from_json(oversized)


def test_immutable_records_and_settings() -> None:
    job = _job()
    with pytest.raises(TypeError):
        job.flight.settings["step_s"] = 1.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        job.execution_options.max_trials = 5  # type: ignore[misc]
