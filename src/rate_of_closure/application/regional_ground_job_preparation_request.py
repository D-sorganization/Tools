"""Strict wire request for preparing, but not running, a regional-ground job."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rate_of_closure.application._regional_ground_execution_job_values import (
    FlightLaunchInput,
)
from rate_of_closure.application._workspace_validation import exact_mapping, stable_id
from rate_of_closure.application.regional_ground_variation_request import (
    GroundRegionalVariationRequest,
    regional_ground_variation_request_from_json,
    regional_ground_variation_request_to_json,
)
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.ground.strict_json import strict_json_object

REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA_VERSION = (
    "rate-of-closure/regional-ground-job-preparation-request/v1"
)
MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES = 1_048_576
_ROOT_FIELDS = frozenset(
    {"schema_version", "unit_system", "job_id", "launch", "variation_request"}
)


@dataclass(frozen=True)
class RegionalGroundJobPreparationRequest:
    """Exact current-editor snapshot submitted to the Python authority."""

    job_id: str
    launch: FlightLaunchInput
    variation_request: GroundRegionalVariationRequest

    def __post_init__(self) -> None:
        """Enforce exact immutable request members at construction."""
        stable_id(self.job_id, "job_id")
        if type(self.launch) is not FlightLaunchInput:
            raise TypeError("launch must be an exact FlightLaunchInput")
        if type(self.variation_request) is not GroundRegionalVariationRequest:
            raise TypeError("variation_request must be exact")

    def to_dict(self) -> dict[str, Any]:
        """Return the detached strict SI preparation mapping."""
        variation = strict_json_object(
            regional_ground_variation_request_to_json(self.variation_request)
        )
        return {
            "schema_version": REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA_VERSION,
            "unit_system": "SI",
            "job_id": self.job_id,
            "launch": self.launch.to_dict(),
            "variation_request": variation,
        }


def regional_ground_job_preparation_request_to_json(
    request: RegionalGroundJobPreparationRequest,
) -> str:
    """Serialize one exact request as bounded canonical numeric JSON."""
    if type(request) is not RegionalGroundJobPreparationRequest:
        raise TypeError("request must be an exact RegionalGroundJobPreparationRequest")
    text = str(canonical_numeric_json(request.to_dict()))
    if len(text.encode("utf-8")) > MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES:
        raise ValueError(
            "regional-ground job preparation request exceeds maximum wire size"
        )
    return text


def regional_ground_job_preparation_request_from_json(
    text: str,
) -> RegionalGroundJobPreparationRequest:
    """Parse one bounded exact request without preparing or running a job."""
    if type(text) is not str:
        raise TypeError("regional-ground job preparation request JSON must be text")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "regional-ground job preparation request must be valid UTF-8"
        ) from exc
    if len(encoded) > MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES:
        raise ValueError(
            "regional-ground job preparation request exceeds maximum wire size"
        )
    payload = strict_json_object(text)
    canonical_numeric_json(payload)
    data = exact_mapping(
        payload, _ROOT_FIELDS, "regional-ground job preparation request"
    )
    if data["schema_version"] != REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if data["unit_system"] != "SI":
        raise ValueError("unsupported unit_system")
    variation_text = str(canonical_numeric_json(data["variation_request"]))
    return RegionalGroundJobPreparationRequest(
        stable_id(data["job_id"], "job_id"),
        FlightLaunchInput.from_dict(data["launch"]),
        regional_ground_variation_request_from_json(variation_text),
    )


__all__ = [
    "MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES",
    "REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA_VERSION",
    "RegionalGroundJobPreparationRequest",
    "regional_ground_job_preparation_request_from_json",
    "regional_ground_job_preparation_request_to_json",
]
