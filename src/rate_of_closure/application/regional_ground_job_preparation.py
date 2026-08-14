"""Python-authoritative preparation of current-editor ground-study jobs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from rate_of_closure.application._regional_ground_execution_job_values import (
    MAX_CAPTURE_SPEED_M_S,
    MAX_EXECUTION_TIMEOUT_S,
    canonical_text,
    integer,
    positive,
)
from rate_of_closure.application.flight_execution_profiles import (
    build_qualified_flight_execution_input,
)
from rate_of_closure.application.regional_ground_execution_job import (
    GroundExecutionOptions,
    RegionalGroundExecutionJob,
    build_regional_ground_execution_job,
)
from rate_of_closure.application.regional_ground_variation_request import (
    regional_ground_variation_request_to_json,
)
from rate_of_closure.variation.regional_ground_variation import (
    GroundRegionalVariationRequest,
)
from shared.python.swing_sim.flight import (
    FlightGroundTransferSettings,
    LaunchConditions,
)
from shared.python.swing_sim.ground import (
    CalibrationKind,
    GroundCalibration,
    GroundProvenance,
    RegionalGroundExecutionOptions,
)


@dataclass(frozen=True, slots=True)
class RegionalGroundJobPreparationProfile:
    """Explicit qualified numerical profile used by current-editor preparation."""

    model_id: str = "waterloo_penner"
    model_version: str = "tools-core/1.0.0"
    flight_max_time_s: float = 10.0
    flight_step_s: float = 0.001
    flight_sample_every: int = 10
    transfer_max_time_s: float = 12.0
    transfer_output_interval_s: float = 0.01
    transfer_max_events: int = 32
    rotational_inertia_factor: float = 0.4
    capture_speed_m_s: float = 0.05
    source_revision: str = "interactive-editor-preparation-v1"
    regional_execution_options: RegionalGroundExecutionOptions = field(
        default_factory=RegionalGroundExecutionOptions
    )

    def __post_init__(self) -> None:
        """Reject non-exact or callback-bearing preparation profiles."""
        canonical_text(self.model_id, "model_id")
        canonical_text(self.model_version, "model_version")
        flight_max_time_s = positive(
            self.flight_max_time_s, "flight_max_time_s", MAX_EXECUTION_TIMEOUT_S
        )
        positive(self.flight_step_s, "flight_step_s", flight_max_time_s)
        integer(self.flight_sample_every, "flight_sample_every", 1, 1_000_000)
        transfer_max_time_s = positive(
            self.transfer_max_time_s,
            "transfer_max_time_s",
            MAX_EXECUTION_TIMEOUT_S,
        )
        positive(
            self.transfer_output_interval_s,
            "transfer_output_interval_s",
            transfer_max_time_s,
        )
        integer(self.transfer_max_events, "transfer_max_events", 1, 10_000)
        positive(self.rotational_inertia_factor, "rotational_inertia_factor", 1.0)
        positive(self.capture_speed_m_s, "capture_speed_m_s", MAX_CAPTURE_SPEED_M_S)
        canonical_text(self.source_revision, "source_revision")
        if type(self.regional_execution_options) is not RegionalGroundExecutionOptions:
            raise TypeError("regional_execution_options must be exact")
        if self.regional_execution_options.is_cancelled is not None:
            raise ValueError(
                "preparation profile cannot retain a cancellation callback"
            )


DEFAULT_REGIONAL_GROUND_JOB_PREPARATION_PROFILE = RegionalGroundJobPreparationProfile()


class RegionalGroundJobPreparer(Protocol):
    """Exact dependency-injection port for UI and transport adapters."""

    def __call__(
        self,
        *,
        job_id: str,
        launch: LaunchConditions,
        variation_request: GroundRegionalVariationRequest,
    ) -> RegionalGroundExecutionJob:
        """Prepare one immutable job without executing its ground trials."""


def _request_sha256(request: GroundRegionalVariationRequest) -> str:
    """Return the canonical identity of the exact variation request."""
    text = regional_ground_variation_request_to_json(request)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _transfer(
    job_id: str,
    request: GroundRegionalVariationRequest,
    profile: RegionalGroundJobPreparationProfile,
) -> FlightGroundTransferSettings:
    """Build the bounded, provenance-bound flight-to-ground transfer request."""
    source_digest = _request_sha256(request)
    return FlightGroundTransferSettings(
        request_id=f"{job_id}:flight-ground-transfer",
        surface=request.regional_plan.base_surface,
        calibration=GroundCalibration(
            calibration_id="editor-surface-unvalidated-v1",
            kind=CalibrationKind.UNVALIDATED,
            source="User-edited regional surface; no measured calibration supplied.",
            confidence=0.0,
        ),
        provenance=GroundProvenance(
            producer="tools.rate_of_closure.editor-job-preparation",
            producer_version="1.0.0",
            source_revision=profile.source_revision,
            input_sha256=source_digest,
        ),
        max_time_s=profile.transfer_max_time_s,
        output_interval_s=profile.transfer_output_interval_s,
        max_events=profile.transfer_max_events,
        rotational_inertia_factor=profile.rotational_inertia_factor,
    )


def prepare_regional_ground_execution_job(
    *,
    job_id: str,
    launch: LaunchConditions,
    variation_request: GroundRegionalVariationRequest,
    profile: RegionalGroundJobPreparationProfile = (
        DEFAULT_REGIONAL_GROUND_JOB_PREPARATION_PROFILE
    ),
) -> RegionalGroundExecutionJob:
    """Prepare one exact job from current validated editors without executing it."""
    if type(launch) is not LaunchConditions:
        raise TypeError("launch must be an exact LaunchConditions")
    if type(variation_request) is not GroundRegionalVariationRequest:
        raise TypeError("variation_request must be exact")
    if type(profile) is not RegionalGroundJobPreparationProfile:
        raise TypeError("profile must be exact")
    if launch.wind_scenario is not None:
        raise ValueError("preparation requires resolved constant wind")
    if variation_request.plan.flight_model != profile.model_id:
        raise ValueError("variation request flight model is not registered")
    transfer = _transfer(job_id, variation_request, profile)
    flight = build_qualified_flight_execution_input(
        launch,
        transfer,
        model_id=profile.model_id,
        model_version=profile.model_version,
        settings={
            "max_time_s": profile.flight_max_time_s,
            "sample_every": profile.flight_sample_every,
            "step_s": profile.flight_step_s,
        },
    )
    return build_regional_ground_execution_job(
        job_id=job_id,
        launch=launch,
        flight=flight,
        transfer=transfer,
        capture_speed_m_s=profile.capture_speed_m_s,
        execution_options=GroundExecutionOptions(variation_request.plan.n_runs),
        regional_execution_options=profile.regional_execution_options,
        variation_request=variation_request,
        producer="tools.rate_of_closure.editor-job-preparation",
        producer_version="1.0.0",
        source_revision=profile.source_revision,
    )


def require_prepared_job_matches_request(
    job: RegionalGroundExecutionJob,
    *,
    job_id: str,
    launch: LaunchConditions,
    variation_request: GroundRegionalVariationRequest,
) -> RegionalGroundExecutionJob:
    """Enforce the exact postcondition of an injected preparation authority."""
    if type(job) is not RegionalGroundExecutionJob:
        raise TypeError("job_preparer must return an exact RegionalGroundExecutionJob")
    if job.job_id != job_id:
        raise ValueError("prepared job_id does not match the request")
    if job.launch.launch != launch:
        raise ValueError("prepared launch does not match the request")
    if job.variation_request != variation_request:
        raise ValueError("prepared variation request does not match the request")
    job.__post_init__()
    return job


__all__ = [
    "DEFAULT_REGIONAL_GROUND_JOB_PREPARATION_PROFILE",
    "RegionalGroundJobPreparer",
    "RegionalGroundJobPreparationProfile",
    "prepare_regional_ground_execution_job",
    "require_prepared_job_matches_request",
]
