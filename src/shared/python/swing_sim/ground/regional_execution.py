"""Authoritative regional-plan execution binding for ground simulation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .bounce_types import RepeatedBounceResult
from .contract_records import GroundSimulationRequest, GroundSimulationResult
from .contract_types import GroundProvenance, _text
from .ground_result_composer import GroundCompositionError, compose_ground_result
from .regional_execution_records import (
    RegionalGroundExecutionFailureReason,
    RegionalGroundExecutionResult,
    RegionalGroundExecutionStatus,
    execution_input_sha256,
)
from .regional_plan_records import GroundRegionalMaterialPlanRequest
from .request_identity import ground_request_fingerprint
from .skid_roll_result_types import SkidRollResult
from .skid_roll_simulation import SkidRollExecution, simulate_skid_roll
from .surface_motion_types import (
    CancellationCheck,
    SkidRollSettings,
    SkidRollTerminationReason,
)

REGIONAL_GROUND_EXECUTOR_ID = "tools-ground-regional-executor"
REGIONAL_GROUND_EXECUTOR_VERSION = "1.0.0"
REGIONAL_GROUND_EXECUTOR_SOURCE = "ground-regional-execution-v1"
MAX_REGIONAL_EXECUTION_STEPS = 1_000_000
MAX_REGIONAL_EXECUTION_TRANSITIONS = 4_096


@dataclass(frozen=True)
class RegionalGroundExecutionOptions:
    """Bounded numerical, cancellation, and source-identity options."""

    settings: SkidRollSettings = field(default_factory=SkidRollSettings)
    is_cancelled: CancellationCheck | None = None
    source_revision: str = REGIONAL_GROUND_EXECUTOR_SOURCE

    def __post_init__(self) -> None:
        if type(self.settings) is not SkidRollSettings:
            raise ValueError("settings must be an exact SkidRollSettings record")
        if self.settings.max_steps > MAX_REGIONAL_EXECUTION_STEPS:
            raise ValueError("settings exceed the regional execution step bound")
        if self.settings.max_surface_transitions > MAX_REGIONAL_EXECUTION_TRANSITIONS:
            raise ValueError("settings exceed the regional transition bound")
        if self.is_cancelled is not None and not callable(self.is_cancelled):
            raise ValueError("is_cancelled must be callable")
        object.__setattr__(
            self,
            "source_revision",
            _text(self.source_revision, "source_revision"),
        )


_FAILURE_MAP = {
    SkidRollTerminationReason.CANCELLED: (
        RegionalGroundExecutionStatus.CANCELLED,
        RegionalGroundExecutionFailureReason.CANCELLED,
    ),
    SkidRollTerminationReason.STEP_LIMIT: (
        RegionalGroundExecutionStatus.FAILED,
        RegionalGroundExecutionFailureReason.STEP_LIMIT,
    ),
    SkidRollTerminationReason.SURFACE_TRANSITION_LIMIT: (
        RegionalGroundExecutionStatus.FAILED,
        RegionalGroundExecutionFailureReason.SURFACE_TRANSITION_LIMIT,
    ),
    SkidRollTerminationReason.UNSUPPORTED_SURFACE: (
        RegionalGroundExecutionStatus.FAILED,
        RegionalGroundExecutionFailureReason.UNSUPPORTED_SURFACE,
    ),
    SkidRollTerminationReason.NUMERICAL_FAILURE: (
        RegionalGroundExecutionStatus.FAILED,
        RegionalGroundExecutionFailureReason.NUMERICAL_FAILURE,
    ),
}


def _validate_inputs(
    request: GroundSimulationRequest,
    prefix: RepeatedBounceResult,
    plan: GroundRegionalMaterialPlanRequest,
    options: RegionalGroundExecutionOptions,
) -> None:
    if type(request) is not GroundSimulationRequest:
        raise ValueError("request must be an exact GroundSimulationRequest")
    if type(prefix) is not RepeatedBounceResult:
        raise ValueError("prefix must be an exact RepeatedBounceResult")
    if type(plan) is not GroundRegionalMaterialPlanRequest:
        raise ValueError("plan must be an exact GroundRegionalMaterialPlanRequest")
    if type(options) is not RegionalGroundExecutionOptions:
        raise ValueError("options must be an exact RegionalGroundExecutionOptions")
    if plan.base_surface != request.surface:
        raise ValueError("plan.base_surface must equal the ground request surface")
    identities = (request.request_id, request.surface.surface_id)
    if identities != (prefix.request_id, prefix.surface_id):
        raise ValueError("ground request and bounce prefix identities must match")
    if prefix.request_fingerprint_sha256 != ground_request_fingerprint(request):
        raise ValueError("bounce prefix fingerprint must match the ground request")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _executor_provenance(
    options: RegionalGroundExecutionOptions,
    ground_digest: str,
    plan_digest: str,
) -> GroundProvenance:
    return GroundProvenance(
        REGIONAL_GROUND_EXECUTOR_ID,
        REGIONAL_GROUND_EXECUTOR_VERSION,
        options.source_revision,
        execution_input_sha256(ground_digest, plan_digest),
    )


def _compose_outcome(
    request: GroundSimulationRequest,
    prefix: RepeatedBounceResult,
    suffix: SkidRollResult,
) -> tuple[
    GroundSimulationResult | None,
    RegionalGroundExecutionStatus,
    RegionalGroundExecutionFailureReason | None,
]:
    failure = _FAILURE_MAP.get(suffix.termination.reason)
    ground_result = None
    if failure is None:
        try:
            ground_result = compose_ground_result(request, prefix, suffix)
        except GroundCompositionError:
            failure = (
                RegionalGroundExecutionStatus.FAILED,
                RegionalGroundExecutionFailureReason.COMPOSITION_FAILURE,
            )
    if ground_result is not None:
        return (
            ground_result,
            RegionalGroundExecutionStatus(ground_result.status.value),
            None,
        )
    if failure is None:
        raise RuntimeError("regional execution requires a terminal outcome")
    return None, *failure


def execute_regional_ground(
    request: GroundSimulationRequest,
    prefix: RepeatedBounceResult,
    plan: GroundRegionalMaterialPlanRequest,
    options: RegionalGroundExecutionOptions | None = None,
) -> RegionalGroundExecutionResult:
    """Run the existing solver/composer using only the plan-owned resolver.

    Preconditions:
        Inputs are exact records and ``plan.base_surface == request.surface``.
    Postconditions:
        Representable outcomes embed frozen ground-result v1; internal bounds
        return a typed null-result status without fabricated physics.
    """
    selected = RegionalGroundExecutionOptions() if options is None else options
    _validate_inputs(request, prefix, plan, selected)
    ground_digest = ground_request_fingerprint(request)
    plan_digest = _sha256(plan.to_json())
    resolver = plan.to_surface_resolver()
    suffix = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(selected.settings, resolver, selected.is_cancelled),
    )
    model_id = f"{prefix.model_id}+{suffix.model_id}"
    model_version = f"{prefix.model_version}+{suffix.model_version}"
    ground_result, status, failure_reason = _compose_outcome(request, prefix, suffix)
    return RegionalGroundExecutionResult(
        request.request_id,
        request.surface.surface_id,
        plan.request_id,
        ground_digest,
        plan_digest,
        status,
        failure_reason,
        ground_result,
        plan.provenance,
        _executor_provenance(selected, ground_digest, plan_digest),
        model_id,
        model_version,
        suffix.surface_transitions,
    )


__all__ = [
    "MAX_REGIONAL_EXECUTION_STEPS",
    "MAX_REGIONAL_EXECUTION_TRANSITIONS",
    "REGIONAL_GROUND_EXECUTOR_ID",
    "REGIONAL_GROUND_EXECUTOR_SOURCE",
    "REGIONAL_GROUND_EXECUTOR_VERSION",
    "RegionalGroundExecutionOptions",
    "execute_regional_ground",
]
