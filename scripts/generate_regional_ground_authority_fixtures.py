"""Regenerate the canonical regional-ground authority fixture family.

The execution job is the root authority. Its flight digests are recomputed by
the registered production profile, then every dependent job/result/status
identity is rebuilt from that qualified job. Use ``--check`` in CI to prevent
hand-edited or stale fixture relationships.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rate_of_closure.application.flight_execution_profiles import (  # noqa: E402
    qualify_flight_execution_input,
)
from rate_of_closure.application.regional_ground_authority_status import (  # noqa: E402
    AuthorityFailureStage,
    AuthorityJobFailure,
    AuthorityJobSnapshot,
    AuthorityJobStatus,
)
from rate_of_closure.application.regional_ground_execution_job import (  # noqa: E402
    build_regional_ground_execution_job,
    regional_ground_execution_job_from_json,
    regional_ground_execution_job_to_json,
)
from rate_of_closure.application.regional_ground_execution_result import (  # noqa: E402
    build_regional_ground_execution_result,
    regional_ground_execution_result_from_json,
    regional_ground_execution_result_to_json,
)
from shared.python.swing_sim.canonical_numeric_json import (  # noqa: E402
    canonical_numeric_json,
)

_FIXTURES = _ROOT / "src" / "rate_of_closure" / "web" / "src" / "model" / "__fixtures__"
_JOB = _FIXTURES / "regional_ground_execution_job_golden_v1.json"
_STATUS = _FIXTURES / "regional_ground_authority_job_status_golden_v1.json"
_RESULT = _FIXTURES / "regional_ground_execution_result_golden_v1.json"
_FAILURE_STAGES: tuple[AuthorityFailureStage, ...] = (
    "authority_restart",
    "cancellation_callback",
    "preflight",
    "executor",
    "validation",
    "progress_callback",
    "publication",
    "runner",
    "result_validation",
)


def _canonical_fixture(value: object) -> str:
    return f"{canonical_numeric_json(value)}\n"


def generated_fixture_texts() -> dict[Path, str]:
    """Return the deterministic, mutually bound canonical fixture family."""
    current_job_fixture = json.loads(_JOB.read_text(encoding="utf-8"))
    source = regional_ground_execution_job_from_json(
        str(canonical_numeric_json(current_job_fixture["job"]))
    )
    qualification = qualify_flight_execution_input(
        source.launch.launch,
        source.transfer,
        source.flight,
    )
    trajectory_sha256 = qualification.recomputed_trajectory_sha256
    result_sha256 = qualification.recomputed_result_sha256
    if trajectory_sha256 is None or result_sha256 is None:
        raise RuntimeError("registered flight profile did not produce exact digests")
    qualified_flight = replace(
        source.flight,
        trajectory_sha256=trajectory_sha256,
        result_sha256=result_sha256,
    )
    job = build_regional_ground_execution_job(
        job_id=source.job_id,
        launch=source.launch.launch,
        flight=qualified_flight,
        transfer=source.transfer,
        capture_speed_m_s=source.capture_speed_m_s,
        execution_options=source.execution_options,
        regional_execution_options=source.regional_execution_options,
        variation_request=source.variation_request,
        producer=source.provenance.producer,
        producer_version=source.provenance.producer_version,
        source_revision=source.provenance.source_revision,
    )
    job_text = regional_ground_execution_job_to_json(job)
    job_fixture = {
        "canonical_sha256": hashlib.sha256(job_text.encode("utf-8")).hexdigest(),
        "input_sha256": job.input_sha256,
        "job": json.loads(job_text),
        "job_sha256": job.job_sha256,
    }

    status_common = {
        "job_id": job.job_id,
        "job_sha256": job.job_sha256,
        "total": job.execution_options.max_trials,
    }
    normal_statuses = (
        AuthorityJobSnapshot(
            **status_common, status=AuthorityJobStatus.QUEUED, completed=0
        ),
        AuthorityJobSnapshot(
            **status_common, status=AuthorityJobStatus.RUNNING, completed=1
        ),
        AuthorityJobSnapshot(
            **status_common, status=AuthorityJobStatus.CANCEL_REQUESTED, completed=2
        ),
        AuthorityJobSnapshot(
            **status_common,
            status=AuthorityJobStatus.SUCCEEDED,
            completed=job.execution_options.max_trials,
            result_available=True,
        ),
        AuthorityJobSnapshot(
            **status_common, status=AuthorityJobStatus.CANCELLED, completed=3
        ),
    )
    failed_statuses = tuple(
        AuthorityJobSnapshot(
            **status_common,
            status=AuthorityJobStatus.FAILED,
            completed=1,
            failure=AuthorityJobFailure(
                "result_rejected"
                if stage == "result_validation"
                else "execution_failed",
                stage,
            ),
        )
        for stage in _FAILURE_STAGES
    )
    current_status_fixture = json.loads(_STATUS.read_text(encoding="utf-8"))
    status_fixture = {
        "fixture_schema": current_status_fixture["fixture_schema"],
        "cases": [status.to_wire() for status in normal_statuses + failed_statuses],
    }

    current_result_fixture = json.loads(_RESULT.read_text(encoding="utf-8"))
    source_result = regional_ground_execution_result_from_json(
        str(canonical_numeric_json(current_result_fixture["result"]))
    )
    result = build_regional_ground_execution_result(job, source_result.dataset)
    result_text = regional_ground_execution_result_to_json(result)
    result_fixture = {
        "canonical_sha256": hashlib.sha256(result_text.encode("utf-8")).hexdigest(),
        "dataset_sha256": result.dataset_sha256,
        "result": json.loads(result_text),
    }
    return {
        _JOB: _canonical_fixture(job_fixture),
        _STATUS: _canonical_fixture(status_fixture),
        _RESULT: _canonical_fixture(result_fixture),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when a canonical fixture differs from deterministic output",
    )
    args = parser.parse_args()
    expected = generated_fixture_texts()
    stale = [path for path, text in expected.items() if path.read_text() != text]
    if args.check:
        if stale:
            print("stale regional-ground authority fixtures:")
            for path in stale:
                print(path.relative_to(_ROOT))
            return 1
        return 0
    for path, text in expected.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
