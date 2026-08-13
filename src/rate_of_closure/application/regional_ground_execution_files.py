"""Bounded, canonical file adapters for regional-ground execution artifacts."""

from __future__ import annotations

from pathlib import Path

from rate_of_closure.variation.scalar_ensemble_io import scalar_ensemble_csv

from .atomic_text_files import write_utf8_text_atomic
from .bounded_text_files import read_bounded_utf8
from .regional_ground_execution_job import (
    MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
    RegionalGroundExecutionJob,
    regional_ground_execution_job_from_json,
    regional_ground_execution_job_to_json,
)
from .regional_ground_execution_result import (
    RegionalGroundExecutionResult,
    regional_ground_execution_result_to_json,
)


def read_regional_ground_execution_job(
    source: str | Path,
) -> RegionalGroundExecutionJob:
    """Read one bounded UTF-8 snapshot and parse the exact execution job."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"regional-ground execution job does not exist: {path}")
    return regional_ground_execution_job_from_json(
        read_bounded_utf8(
            path,
            MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
            "regional-ground execution job",
        )
    )


def write_regional_ground_execution_job_atomic(
    job: RegionalGroundExecutionJob,
    destination: str | Path | None,
) -> bool:
    """Validate and atomically write one canonical execution job."""
    text = regional_ground_execution_job_to_json(job)
    written: bool = write_utf8_text_atomic(
        text, destination, document_name="regional-ground execution job"
    )
    return written


def write_regional_ground_execution_result_atomic(
    result: RegionalGroundExecutionResult,
    destination: str | Path | None,
) -> bool:
    """Validate and atomically write one canonical job-bound result."""
    text = regional_ground_execution_result_to_json(result)
    written: bool = write_utf8_text_atomic(
        text, destination, document_name="regional-ground execution result"
    )
    return written


def write_regional_ground_execution_rows_csv_atomic(
    result: RegionalGroundExecutionResult,
    destination: str | Path | None,
) -> bool:
    """Atomically export every retained scalar row from an exact result."""
    if type(result) is not RegionalGroundExecutionResult:
        raise TypeError("result must be an exact RegionalGroundExecutionResult")
    result.__post_init__()
    text = scalar_ensemble_csv(result.dataset) + "\n"
    written: bool = write_utf8_text_atomic(
        text, destination, document_name="regional-ground execution rows CSV"
    )
    return written


__all__ = [
    "read_regional_ground_execution_job",
    "write_regional_ground_execution_job_atomic",
    "write_regional_ground_execution_result_atomic",
    "write_regional_ground_execution_rows_csv_atomic",
]
