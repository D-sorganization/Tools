"""Canonical UI-neutral execution jobs for seeded regional-ground studies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

from rate_of_closure.application._regional_ground_execution_job_values import (
    MAX_CAPTURE_SPEED_M_S,
    FlightExecutionInput,
    FlightLaunchInput,
    GroundExecutionOptions,
    canonical_flight_evidence_sha256,
    canonical_flight_result_sha256,
    canonical_flight_trajectory_sha256,
    digest,
    positive,
    sha256,
)
from rate_of_closure.application._regional_ground_execution_physics import (
    regional_execution_options_from_dict,
    regional_execution_options_to_dict,
)
from rate_of_closure.application._regional_ground_execution_transfer import (
    transfer_from_dict,
    transfer_payload,
)
from rate_of_closure.application._workspace_validation import exact_mapping, stable_id
from rate_of_closure.application.regional_ground_execution_qualification import (
    qualify_regional_plan_for_launch,
)
from rate_of_closure.application.regional_ground_variation_request import (
    GroundRegionalVariationRequest,
    regional_ground_variation_request_from_json,
    regional_ground_variation_request_to_json,
)
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.flight import (
    FlightGroundTransferSettings,
    LaunchConditions,
)
from shared.python.swing_sim.ground import (
    GroundProvenance,
    GroundRegionalMaterialPlanRequest,
    RegionalGroundExecutionOptions,
)
from shared.python.swing_sim.ground.contract_wire import (
    record_from_dict,
    record_to_dict,
)
from shared.python.swing_sim.ground.regional_plan_records import (
    regional_plan_request_sha256,
)
from shared.python.swing_sim.ground.strict_json import strict_json_object

REGIONAL_GROUND_EXECUTION_JOB_SCHEMA_VERSION = (
    "rate-of-closure/regional-ground-execution-job/v1"
)
MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES = 1_048_576
UNIT_SYSTEM_SI = "SI"

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "unit_system",
        "job_id",
        "launch",
        "flight",
        "transfer",
        "capture_speed_m_s",
        "execution_options",
        "regional_execution_options",
        "qualified_regional_plan",
        "qualified_plan_sha256",
        "variation_request",
        "input_sha256",
        "provenance",
        "job_sha256",
    }
)


def _variation_payload(request: GroundRegionalVariationRequest) -> dict[str, Any]:
    payload: dict[str, Any] = strict_json_object(
        regional_ground_variation_request_to_json(request)
    )
    return payload


@dataclass(frozen=True)
class RegionalGroundExecutionJob:
    """Immutable v1 orchestration authority for one seeded ground study."""

    job_id: str
    launch: FlightLaunchInput
    flight: FlightExecutionInput
    transfer: FlightGroundTransferSettings
    capture_speed_m_s: float
    execution_options: GroundExecutionOptions
    regional_execution_options: RegionalGroundExecutionOptions
    qualified_regional_plan: GroundRegionalMaterialPlanRequest
    qualified_plan_sha256: str
    variation_request: GroundRegionalVariationRequest
    input_sha256: str
    provenance: GroundProvenance
    job_sha256: str

    def __post_init__(self) -> None:
        self._validate_types_and_bounds()
        self._validate_cross_contract_authority()
        if self.provenance.input_sha256 != self.input_sha256:
            raise ValueError("provenance input_sha256 must match input authority")
        if self.input_sha256 != self.expected_input_sha256:
            raise ValueError("input_sha256 must match the embedded input authority")
        if self.job_sha256 != self.expected_job_sha256:
            raise ValueError("job_sha256 must match the complete job authority")

    def _validate_types_and_bounds(self) -> None:
        stable_id(self.job_id, "job_id")
        if type(self.launch) is not FlightLaunchInput:
            raise TypeError("launch must be an exact execution-job record")
        if type(self.flight) is not FlightExecutionInput:
            raise TypeError("flight must be an exact execution-job record")
        if type(self.transfer) is not FlightGroundTransferSettings:
            raise TypeError("transfer must be an exact FlightGroundTransferSettings")
        if type(self.execution_options) is not GroundExecutionOptions:
            raise TypeError("execution_options must be exact")
        if type(self.regional_execution_options) is not RegionalGroundExecutionOptions:
            raise TypeError("regional_execution_options must be exact")
        regional_execution_options_to_dict(self.regional_execution_options)
        if type(self.qualified_regional_plan) is not GroundRegionalMaterialPlanRequest:
            raise TypeError("qualified_regional_plan must be exact")
        digest(self.qualified_plan_sha256, "qualified_plan_sha256")
        if type(self.variation_request) is not GroundRegionalVariationRequest:
            raise TypeError("variation_request must be exact")
        positive(self.capture_speed_m_s, "capture_speed_m_s", MAX_CAPTURE_SPEED_M_S)
        digest(self.input_sha256, "input_sha256")
        digest(self.job_sha256, "job_sha256")

    def _validate_cross_contract_authority(self) -> None:
        options = self.execution_options
        request = self.variation_request
        if options.max_trials != request.plan.n_runs:
            raise ValueError("max_trials must equal variation request n_runs")
        if self.flight.model_id != request.plan.flight_model:
            raise ValueError(
                "flight model_id must match variation request flight_model"
            )
        if request.regional_plan.base_surface != self.transfer.surface:
            raise ValueError("source regional base surface must match transfer surface")
        expected_plan = qualify_regional_plan_for_launch(
            request.regional_plan,
            self.launch.launch,
            self.transfer.surface,
            source_revision=self.regional_execution_options.source_revision,
        )
        if self.qualified_regional_plan != expected_plan:
            raise ValueError(
                "qualified regional plan must match launch-origin translation"
            )
        if self.qualified_plan_sha256 != regional_plan_request_sha256(expected_plan):
            raise ValueError("qualified_plan_sha256 must match the qualified plan")

    def _inputs(self) -> dict[str, Any]:
        return {
            "launch": self.launch.to_dict(),
            "flight": self.flight.to_dict(),
            "transfer": transfer_payload(self.transfer),
            "capture_speed_m_s": self.capture_speed_m_s,
            "execution_options": self.execution_options.to_dict(),
            "regional_execution_options": regional_execution_options_to_dict(
                self.regional_execution_options
            ),
            "qualified_regional_plan": self.qualified_regional_plan.to_dict(),
            "qualified_plan_sha256": self.qualified_plan_sha256,
            "variation_request": _variation_payload(self.variation_request),
        }

    @property
    def expected_input_sha256(self) -> str:
        """Return the canonical digest of every executable input."""
        input_digest: str = sha256(self._inputs())
        return input_digest

    @property
    def expected_job_sha256(self) -> str:
        """Return the canonical digest of the whole envelope except itself."""
        job_digest: str = sha256(self.to_dict(include_job_sha256=False))
        return job_digest

    @property
    def canonical_sha256(self) -> str:
        """Return the SHA-256 of the final canonical wire document."""
        text = regional_ground_execution_job_to_json(self)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_job_sha256: bool = True) -> dict[str, Any]:
        """Return one detached strict v1 mapping."""
        payload = {
            "schema_version": REGIONAL_GROUND_EXECUTION_JOB_SCHEMA_VERSION,
            "unit_system": UNIT_SYSTEM_SI,
            "job_id": self.job_id,
            **self._inputs(),
            "input_sha256": self.input_sha256,
            "provenance": record_to_dict(self.provenance),
        }
        if include_job_sha256:
            payload["job_sha256"] = self.job_sha256
        return payload


def _unchecked_job(
    *,
    job_id: str,
    launch: FlightLaunchInput,
    flight: FlightExecutionInput,
    transfer: FlightGroundTransferSettings,
    capture_speed_m_s: float,
    execution_options: GroundExecutionOptions,
    regional_execution_options: RegionalGroundExecutionOptions,
    qualified_regional_plan: GroundRegionalMaterialPlanRequest,
    qualified_plan_sha256: str,
    variation_request: GroundRegionalVariationRequest,
    provenance: GroundProvenance,
) -> RegionalGroundExecutionJob:
    placeholder = "0" * 64
    job = object.__new__(RegionalGroundExecutionJob)
    values = locals() | {"input_sha256": placeholder, "job_sha256": placeholder}
    for name in RegionalGroundExecutionJob.__dataclass_fields__:
        object.__setattr__(job, name, values[name])
    return job


def build_regional_ground_execution_job(
    *,
    job_id: str,
    launch: LaunchConditions,
    flight: FlightExecutionInput,
    transfer: FlightGroundTransferSettings,
    capture_speed_m_s: float,
    execution_options: GroundExecutionOptions,
    regional_execution_options: RegionalGroundExecutionOptions,
    variation_request: GroundRegionalVariationRequest,
    producer: str,
    producer_version: str,
    source_revision: str,
) -> RegionalGroundExecutionJob:
    """Build a job and derive both canonical identities without physics."""
    placeholder = "0" * 64
    qualified_plan = qualify_regional_plan_for_launch(
        variation_request.regional_plan,
        launch,
        transfer.surface,
        source_revision=regional_execution_options.source_revision,
    )
    job = _unchecked_job(
        job_id=job_id,
        launch=FlightLaunchInput(launch),
        flight=flight,
        transfer=transfer,
        capture_speed_m_s=capture_speed_m_s,
        execution_options=execution_options,
        regional_execution_options=regional_execution_options,
        qualified_regional_plan=qualified_plan,
        qualified_plan_sha256=regional_plan_request_sha256(qualified_plan),
        variation_request=variation_request,
        provenance=GroundProvenance(
            producer, producer_version, source_revision, placeholder
        ),
    )
    input_digest = job.expected_input_sha256
    object.__setattr__(job, "input_sha256", input_digest)
    object.__setattr__(
        job,
        "provenance",
        GroundProvenance(producer, producer_version, source_revision, input_digest),
    )
    object.__setattr__(job, "job_sha256", job.expected_job_sha256)
    job.__post_init__()
    return job


def regional_ground_execution_job_to_json(job: RegionalGroundExecutionJob) -> str:
    """Serialize one validated job with bounded canonical numeric JSON."""
    if type(job) is not RegionalGroundExecutionJob:
        raise TypeError("job must be an exact RegionalGroundExecutionJob")
    job.__post_init__()
    text = str(canonical_numeric_json(job.to_dict()))
    if len(text.encode("utf-8")) > MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES:
        raise ValueError("regional-ground execution job exceeds maximum wire size")
    return text


def regional_ground_execution_job_from_json(text: str) -> RegionalGroundExecutionJob:
    """Parse one bounded exact job without executing any physics."""
    if type(text) is not str:
        raise TypeError("regional-ground execution job JSON must be text")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("regional-ground execution job must be valid UTF-8") from exc
    if len(encoded) > MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES:
        raise ValueError("regional-ground execution job exceeds maximum wire size")
    payload = strict_json_object(text)
    canonical_numeric_json(payload)
    data = exact_mapping(payload, _ROOT_FIELDS, "regional-ground execution job")
    if data["schema_version"] != REGIONAL_GROUND_EXECUTION_JOB_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if data["unit_system"] != UNIT_SYSTEM_SI:
        raise ValueError("unsupported unit_system")
    variation_text = str(canonical_numeric_json(data["variation_request"]))
    return RegionalGroundExecutionJob(
        stable_id(data["job_id"], "job_id"),
        FlightLaunchInput.from_dict(data["launch"]),
        FlightExecutionInput.from_dict(data["flight"]),
        transfer_from_dict(data["transfer"]),
        positive(data["capture_speed_m_s"], "capture_speed_m_s", MAX_CAPTURE_SPEED_M_S),
        GroundExecutionOptions.from_dict(data["execution_options"]),
        regional_execution_options_from_dict(data["regional_execution_options"]),
        GroundRegionalMaterialPlanRequest.from_dict(data["qualified_regional_plan"]),
        digest(data["qualified_plan_sha256"], "qualified_plan_sha256"),
        regional_ground_variation_request_from_json(variation_text),
        digest(data["input_sha256"], "input_sha256"),
        cast(GroundProvenance, record_from_dict(GroundProvenance, data["provenance"])),
        digest(data["job_sha256"], "job_sha256"),
    )


__all__ = [
    "MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES",
    "REGIONAL_GROUND_EXECUTION_JOB_SCHEMA_VERSION",
    "FlightExecutionInput",
    "FlightLaunchInput",
    "GroundExecutionOptions",
    "RegionalGroundExecutionJob",
    "build_regional_ground_execution_job",
    "canonical_flight_evidence_sha256",
    "canonical_flight_result_sha256",
    "canonical_flight_trajectory_sha256",
    "regional_ground_execution_job_from_json",
    "regional_ground_execution_job_to_json",
]
