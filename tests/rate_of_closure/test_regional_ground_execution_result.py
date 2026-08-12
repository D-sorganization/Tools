"""Cross-runtime contract tests for job-bound regional-ground results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest

from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
    regional_ground_execution_job_from_json,
)
from rate_of_closure.application.regional_ground_execution_result import (
    MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES,
    RegionalGroundExecutionResult,
    build_regional_ground_execution_result,
    regional_ground_execution_result_from_json,
    regional_ground_execution_result_to_json,
)
from rate_of_closure.variation.scalar_ensemble_wire import (
    scalar_ensemble_dataset_from_wire,
)
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_MODEL_FIXTURES = (
    Path(__file__).parents[2]
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
)
_FIXTURE = _MODEL_FIXTURES / "regional_ground_execution_result_golden_v1.json"
_JOB_FIXTURE = _MODEL_FIXTURES / "regional_ground_execution_job_golden_v1.json"


def _job() -> RegionalGroundExecutionJob:
    payload = json.loads(_JOB_FIXTURE.read_text(encoding="utf-8"))["job"]
    return regional_ground_execution_job_from_json(json.dumps(payload))


def _dataset_payload() -> dict[str, object]:
    fixture = cast(dict[str, Any], json.loads(_FIXTURE.read_text(encoding="utf-8")))
    result = cast(dict[str, Any], fixture["result"])
    return cast(dict[str, object], result["dataset"])


def _result() -> RegionalGroundExecutionResult:
    return build_regional_ground_execution_result(
        _job(), scalar_ensemble_dataset_from_wire(_dataset_payload())
    )


def _substitute_series(dataset: dict[str, Any]) -> None:
    row = cast(dict[str, Any], dataset["rows"][0])
    row["series_id"] = "substituted"
    row["row_id"] = "series:substituted/trial:0"


def test_python_authority_exactly_recreates_shared_golden() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    result = _result()
    text = regional_ground_execution_result_to_json(result)

    assert json.loads(text) == fixture["result"]
    assert result.dataset_sha256 == fixture["dataset_sha256"]
    assert result.canonical_sha256 == fixture["canonical_sha256"]
    assert (
        regional_ground_execution_result_from_json(text, expected_job=_job()) == result
    )
    assert result.dataset.rows[1].values["metric.carry_distance"] is None


@pytest.mark.parametrize("field", ["job_id", "job_sha256", "input_sha256"])
def test_job_identity_substitution_fails_expected_job_binding(field: str) -> None:
    payload = json.loads(regional_ground_execution_result_to_json(_result()))
    payload[field] = "substituted" if field == "job_id" else "0" * 64

    with pytest.raises(ValueError, match=field):
        regional_ground_execution_result_from_json(
            json.dumps(payload), expected_job=_job()
        )


def test_dataset_substitution_and_digest_tampering_fail_closed() -> None:
    payload = json.loads(regional_ground_execution_result_to_json(_result()))
    payload["dataset"]["rows"][0]["values"]["metric.carry_distance"] += 1.0
    with pytest.raises(ValueError, match="dataset_sha256"):
        regional_ground_execution_result_from_json(json.dumps(payload))

    payload = json.loads(regional_ground_execution_result_to_json(_result()))
    payload["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="dataset_sha256"):
        regional_ground_execution_result_from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda dataset: dataset.update(result_id="other-study"), "result_id"),
        (lambda dataset: dataset["rows"].pop(), "trial count"),
        (lambda dataset: dataset["rows"].reverse(), "trial ordering"),
        (_substitute_series, "series_id"),
    ],
)
def test_self_consistent_dataset_substitution_fails_job_binding(
    mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    payload = json.loads(regional_ground_execution_result_to_json(_result()))
    dataset = cast(dict[str, Any], payload["dataset"])
    mutate(dataset)
    canonical = str(canonical_numeric_json(dataset))
    payload["dataset_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    parsed = regional_ground_execution_result_from_json(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        parsed.assert_matches_job(_job())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(extra=True), "fields"),
        (
            lambda value: value["dataset"].update(extra=True),
            "scalar ensemble result fields",
        ),
        (
            lambda value: value["dataset"]["rows"][1]["values"].update(
                {"metric.carry_distance": "0"}
            ),
            "finite or null",
        ),
        (
            lambda value: value["dataset"]["rows"][0]["values"].pop(
                "metric.carry_distance"
            ),
            "variable keys",
        ),
        (
            lambda value: value["dataset"]["rows"][0].update(trial_index=True),
            "trial_index",
        ),
    ],
)
def test_nested_extra_and_typed_null_adversaries_fail_closed(
    mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    payload = json.loads(regional_ground_execution_result_to_json(_result()))
    mutate(payload)
    with pytest.raises((TypeError, ValueError), match=message):
        regional_ground_execution_result_from_json(json.dumps(payload))


def test_duplicate_fields_and_wire_size_fail_closed() -> None:
    text = regional_ground_execution_result_to_json(_result())
    duplicate = text.replace(
        '"job_id":"driver-ground-study-1729"',
        '"job_id":"driver-ground-study-1729","job_id":"duplicate"',
    )
    with pytest.raises(ValueError, match="duplicate"):
        regional_ground_execution_result_from_json(duplicate)

    nested_duplicate = text.replace(
        '"result_id":"seeded-ground-study"',
        '"result_id":"seeded-ground-study","result_id":"duplicate"',
    )
    with pytest.raises(ValueError, match="duplicate"):
        regional_ground_execution_result_from_json(nested_duplicate)

    oversized = "é" * (MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES // 2 + 1)
    with pytest.raises(ValueError, match="maximum wire size"):
        regional_ground_execution_result_from_json(oversized)


def test_result_and_nested_dataset_are_immutable() -> None:
    result = _result()
    with pytest.raises(FrozenInstanceError):
        result.job_id = "changed"
    with pytest.raises(TypeError):
        result.dataset.rows[0].values["metric.carry_distance"] = 0.0


def test_direct_construction_revalidates_dataset_and_job_digests() -> None:
    result = _result()
    with pytest.raises(ValueError, match="dataset_sha256"):
        replace(result, dataset_sha256="0" * 64)
