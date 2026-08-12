"""Cross-runtime authority job-status wire contract tests."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.application.regional_ground_authority_status import (
    MAX_AUTHORITY_JOB_STATUS_BYTES,
    AuthorityJobFailure,
    AuthorityJobSnapshot,
    AuthorityJobStatus,
    regional_ground_authority_job_status_from_json,
    regional_ground_authority_job_status_to_json,
)
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from tests.rate_of_closure.test_regional_ground_execution_result import (
    _job as golden_job,
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
    / "regional_ground_authority_job_status_golden_v1.json"
)
_FIXTURE_SCHEMA = "rate-of-closure/regional-ground-authority-job-status-golden/v1"
_FAILURE_STAGES = (
    "cancellation_callback",
    "preflight",
    "executor",
    "validation",
    "progress_callback",
    "publication",
    "runner",
    "result_validation",
)


def _statuses() -> tuple[AuthorityJobSnapshot, ...]:
    job = golden_job()
    common = {"job_id": job.job_id, "job_sha256": job.job_sha256, "total": 4}
    normal = (
        AuthorityJobSnapshot(**common, status=AuthorityJobStatus.QUEUED, completed=0),
        AuthorityJobSnapshot(**common, status=AuthorityJobStatus.RUNNING, completed=1),
        AuthorityJobSnapshot(
            **common, status=AuthorityJobStatus.CANCEL_REQUESTED, completed=2
        ),
        AuthorityJobSnapshot(
            **common,
            status=AuthorityJobStatus.SUCCEEDED,
            completed=4,
            result_available=True,
        ),
        AuthorityJobSnapshot(
            **common, status=AuthorityJobStatus.CANCELLED, completed=3
        ),
    )
    failures = tuple(
        AuthorityJobSnapshot(
            **common,
            status=AuthorityJobStatus.FAILED,
            completed=1,
            failure=AuthorityJobFailure(
                (
                    "result_rejected"
                    if stage == "result_validation"
                    else "execution_failed"
                ),
                stage,
            ),
        )
        for stage in _FAILURE_STAGES
    )
    return normal + failures


def _golden_payload() -> dict[str, object]:
    return {
        "fixture_schema": _FIXTURE_SCHEMA,
        "cases": [status.to_wire() for status in _statuses()],
    }


def test_python_authority_recreates_shared_golden_bytes_and_semantics() -> None:
    fixture_text = _FIXTURE.read_text(encoding="utf-8").strip()
    assert fixture_text == canonical_numeric_json(_golden_payload())
    fixture = json.loads(fixture_text)
    job = golden_job()

    for expected, wire in zip(_statuses(), fixture["cases"], strict=True):
        case_text = canonical_numeric_json(wire)
        parsed = regional_ground_authority_job_status_from_json(case_text, job)
        assert parsed == expected
        assert regional_ground_authority_job_status_to_json(parsed, job) == case_text


@pytest.mark.parametrize(
    "change,message",
    [
        ({"extra": True}, "fields"),
        ({"completed": True}, "completed"),
        ({"total": 9_007_199_254_740_992}, "safe"),
        ({"completed": math.nan}, "finite"),
        ({"job_id": "other-job"}, "job_id"),
        ({"job_sha256": "0" * 64}, "job_sha256"),
        ({"total": 5}, "total"),
        ({"status": "complete"}, "status"),
        ({"result_available": 1}, "result_available"),
    ],
)
def test_invalid_typed_identity_and_numeric_status_values_fail_closed(
    change: dict[str, object], message: str
) -> None:
    wire = _statuses()[0].to_wire() | change
    with pytest.raises((TypeError, ValueError), match=message):
        regional_ground_authority_job_status_from_json(
            json.dumps(wire, allow_nan=True, separators=(",", ":")), golden_job()
        )


@pytest.mark.parametrize(
    "changed,message",
    [
        ({"status": "queued", "completed": 1}, "queued"),
        (
            {"status": "succeeded", "completed": 3, "result_available": True},
            "succeeded",
        ),
        (
            {"status": "succeeded", "completed": 4, "result_available": False},
            "succeeded",
        ),
        ({"status": "running", "result_available": True}, "result"),
        ({"status": "failed", "failure": None}, "failure"),
        (
            {
                "status": "failed",
                "failure": {
                    "code": "execution_failed",
                    "stage": "runner",
                    "extra": True,
                },
            },
            "fields",
        ),
        (
            {
                "status": "failed",
                "failure": {"code": "internal_exception", "stage": "runner"},
            },
            "code",
        ),
        (
            {
                "status": "failed",
                "failure": {"code": "execution_failed", "stage": "internal"},
            },
            "stage",
        ),
    ],
)
def test_impossible_terminal_and_failure_semantics_fail_closed(
    changed: dict[str, object], message: str
) -> None:
    wire = _statuses()[0].to_wire() | changed
    with pytest.raises((TypeError, ValueError), match=message):
        regional_ground_authority_job_status_from_json(
            json.dumps(wire, separators=(",", ":")), golden_job()
        )


def test_duplicate_and_oversized_documents_fail_before_publication() -> None:
    source = regional_ground_authority_job_status_to_json(_statuses()[0], golden_job())
    duplicate = source.replace('"job_id":', '"job_id":"duplicate","job_id":', 1)
    with pytest.raises(ValueError, match="duplicate"):
        regional_ground_authority_job_status_from_json(duplicate, golden_job())
    with pytest.raises(ValueError, match="maximum wire size"):
        regional_ground_authority_job_status_from_json(
            source + (" " * MAX_AUTHORITY_JOB_STATUS_BYTES), golden_job()
        )


def test_python_serializer_revalidates_mutated_instances() -> None:
    changed = replace(_statuses()[0], total=5)
    with pytest.raises(ValueError, match="total"):
        regional_ground_authority_job_status_to_json(changed, golden_job())
