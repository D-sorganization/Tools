"""Job-bound result envelopes for regional-ground scalar ensembles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

from rate_of_closure.application._regional_ground_execution_job_values import (
    digest,
    sha256,
)
from rate_of_closure.application._workspace_validation import exact_mapping, stable_id
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from rate_of_closure.variation.scalar_ensemble_contract import ScalarEnsembleDataset
from rate_of_closure.variation.scalar_ensemble_wire import (
    scalar_ensemble_dataset_from_wire,
)
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.ground.strict_json import strict_json_object

REGIONAL_GROUND_EXECUTION_RESULT_SCHEMA_VERSION = (
    "rate-of-closure/regional-ground-execution-result/v1"
)
MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES = 8_388_608

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "job_id",
        "job_sha256",
        "input_sha256",
        "dataset_sha256",
        "dataset",
    }
)


@dataclass(frozen=True)
class RegionalGroundExecutionResult:
    """Immutable identity binding between one job and its complete dataset."""

    job_id: str
    job_sha256: str
    input_sha256: str
    dataset_sha256: str
    dataset: ScalarEnsembleDataset

    def __post_init__(self) -> None:
        stable_id(self.job_id, "job_id")
        digest(self.job_sha256, "job_sha256")
        digest(self.input_sha256, "input_sha256")
        digest(self.dataset_sha256, "dataset_sha256")
        if type(self.dataset) is not ScalarEnsembleDataset:
            raise TypeError("dataset must be an exact ScalarEnsembleDataset")
        if self.dataset_sha256 != self.expected_dataset_sha256:
            raise ValueError("dataset_sha256 must match the complete dataset authority")

    @property
    def expected_dataset_sha256(self) -> str:
        """Return the digest of the complete canonical scalar-ensemble object."""
        return cast(str, sha256(self.dataset.to_wire()))

    @property
    def canonical_sha256(self) -> str:
        """Return the digest of the complete canonical result envelope bytes."""
        text = regional_ground_execution_result_to_json(self)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def assert_matches_job(self, job: RegionalGroundExecutionJob) -> None:
        """Fail unless all carried job identities match one validated authority."""
        if type(job) is not RegionalGroundExecutionJob:
            raise TypeError("expected_job must be an exact RegionalGroundExecutionJob")
        job.__post_init__()
        for field in ("job_id", "job_sha256", "input_sha256"):
            if getattr(self, field) != getattr(job, field):
                raise ValueError(f"{field} must match the expected execution job")
        request = job.variation_request
        if self.dataset.result_id != request.result_id:
            raise ValueError("dataset result_id must match the expected execution job")
        if len(self.dataset.rows) != job.execution_options.max_trials:
            raise ValueError(
                "dataset trial count must match the expected execution job"
            )
        indexes = tuple(row.trial_index for row in self.dataset.rows)
        if indexes != tuple(range(job.execution_options.max_trials)):
            raise ValueError(
                "dataset trial ordering must match the expected execution job"
            )
        if any(row.series_id != request.series_id for row in self.dataset.rows):
            raise ValueError("dataset series_id must match the expected execution job")

    def to_dict(self) -> dict[str, Any]:
        """Return one detached exact v1 wire mapping."""
        return {
            "schema_version": REGIONAL_GROUND_EXECUTION_RESULT_SCHEMA_VERSION,
            "job_id": self.job_id,
            "job_sha256": self.job_sha256,
            "input_sha256": self.input_sha256,
            "dataset_sha256": self.dataset_sha256,
            "dataset": self.dataset.to_wire(),
        }


def build_regional_ground_execution_result(
    job: RegionalGroundExecutionJob,
    dataset: ScalarEnsembleDataset,
) -> RegionalGroundExecutionResult:
    """Build a complete result envelope from validated job and dataset authorities."""
    if type(job) is not RegionalGroundExecutionJob:
        raise TypeError("job must be an exact RegionalGroundExecutionJob")
    if type(dataset) is not ScalarEnsembleDataset:
        raise TypeError("dataset must be an exact ScalarEnsembleDataset")
    job.__post_init__()
    result = RegionalGroundExecutionResult(
        job.job_id,
        job.job_sha256,
        job.input_sha256,
        sha256(dataset.to_wire()),
        dataset,
    )
    result.assert_matches_job(job)
    return result


def regional_ground_execution_result_to_json(
    result: RegionalGroundExecutionResult,
) -> str:
    """Serialize one validated result as bounded canonical numeric JSON."""
    if type(result) is not RegionalGroundExecutionResult:
        raise TypeError("result must be an exact RegionalGroundExecutionResult")
    result.__post_init__()
    text = str(canonical_numeric_json(result.to_dict()))
    if len(text.encode("utf-8")) > MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES:
        raise ValueError("regional-ground execution result exceeds maximum wire size")
    return text


def regional_ground_execution_result_from_json(
    text: str,
    *,
    expected_job: RegionalGroundExecutionJob | None = None,
) -> RegionalGroundExecutionResult:
    """Parse one bounded exact envelope and optionally bind it to its job."""
    if type(text) is not str:
        raise TypeError("regional-ground execution result JSON must be text")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "regional-ground execution result must be valid UTF-8"
        ) from exc
    if len(encoded) > MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES:
        raise ValueError("regional-ground execution result exceeds maximum wire size")
    payload = strict_json_object(text)
    canonical_numeric_json(payload)
    item = exact_mapping(payload, _ROOT_FIELDS, "regional-ground execution result")
    if item["schema_version"] != REGIONAL_GROUND_EXECUTION_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    result = RegionalGroundExecutionResult(
        stable_id(item["job_id"], "job_id"),
        digest(item["job_sha256"], "job_sha256"),
        digest(item["input_sha256"], "input_sha256"),
        digest(item["dataset_sha256"], "dataset_sha256"),
        scalar_ensemble_dataset_from_wire(item["dataset"]),
    )
    if expected_job is not None:
        result.assert_matches_job(expected_job)
    return result


__all__ = [
    "MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES",
    "REGIONAL_GROUND_EXECUTION_RESULT_SCHEMA_VERSION",
    "RegionalGroundExecutionResult",
    "build_regional_ground_execution_result",
    "regional_ground_execution_result_from_json",
    "regional_ground_execution_result_to_json",
]
