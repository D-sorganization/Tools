"""Persistence contracts for imported regional-ground execution artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from rate_of_closure.application.regional_ground_execution_files import (
    read_regional_ground_execution_job,
    write_regional_ground_execution_job_atomic,
    write_regional_ground_execution_result_atomic,
    write_regional_ground_execution_rows_csv_atomic,
)
from rate_of_closure.application.regional_ground_execution_job import (
    MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
    regional_ground_execution_job_from_json,
    regional_ground_execution_job_to_json,
)
from rate_of_closure.application.regional_ground_execution_result import (
    regional_ground_execution_result_from_json,
)
from tests.rate_of_closure.test_regional_ground_execution_result import _job, _result

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_job_read_and_atomic_artifact_exports_round_trip(tmp_path: Path) -> None:
    job = _job()
    result = _result()
    job_path = tmp_path / "job.json"
    result_path = tmp_path / "result.json"
    csv_path = tmp_path / "rows.csv"

    assert write_regional_ground_execution_job_atomic(job, job_path)
    assert read_regional_ground_execution_job(job_path) == job
    assert write_regional_ground_execution_result_atomic(result, result_path)
    assert (
        regional_ground_execution_result_from_json(
            result_path.read_text(encoding="utf-8"), expected_job=job
        )
        == result
    )
    assert write_regional_ground_execution_rows_csv_atomic(result, csv_path)
    csv = csv_path.read_text(encoding="utf-8")
    assert "trial_index" in csv
    assert len(csv.splitlines()) == len(result.dataset.rows) + 1


def test_invalid_or_oversized_import_fails_without_mutating_prior_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "job.json"
    path.write_text("prior", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON is invalid"):
        read_regional_ground_execution_job(path)
    path.write_bytes(b"x" * (MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES + 1))
    with pytest.raises(ValueError, match="maximum wire size"):
        read_regional_ground_execution_job(path)


def test_atomic_writer_validates_before_replacing_destination(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    path.write_text("prior", encoding="utf-8")
    invalid = object.__new__(type(_job()))
    with pytest.raises((AttributeError, TypeError, ValueError)):
        write_regional_ground_execution_job_atomic(invalid, path)
    assert path.read_text(encoding="utf-8") == "prior"


def test_cancelled_destinations_do_not_write() -> None:
    assert not write_regional_ground_execution_job_atomic(_job(), None)
    assert not write_regional_ground_execution_result_atomic(_result(), None)
    assert not write_regional_ground_execution_rows_csv_atomic(_result(), None)


def test_import_rejects_duplicate_root_key(tmp_path: Path) -> None:
    text = regional_ground_execution_job_to_json(_job())
    duplicate = text.replace("{", '{"schema_version":"duplicate",', 1)
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_regional_ground_execution_job(path)


def test_file_adapter_matches_direct_parser(tmp_path: Path) -> None:
    text = regional_ground_execution_job_to_json(_job())
    path = tmp_path / "job.json"
    path.write_text(text, encoding="utf-8")
    assert read_regional_ground_execution_job(path) == (
        regional_ground_execution_job_from_json(text)
    )
