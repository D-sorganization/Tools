"""Fail-closed production-runner qualification for regional-ground jobs.

The v1 execution job records generic flight evidence. A separate exact
versioned profile binds eligible evidence to solver semantics. This module is
the production boundary: it releases recomputed flight only after both digests
match and preserves typed cancellation/failure behavior.
"""

from __future__ import annotations

from enum import StrEnum

from rate_of_closure.application.flight_execution_profiles import (
    FlightExecutionProfileQualificationError,
    FlightExecutionQualificationReason,
    recompute_qualified_flight_result,
)
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
    GroundRegionalVariationHooks,
)
from shared.python.swing_sim.flight import FlightModelType, FlightResult


class ProductionRunnerPreflightReason(StrEnum):
    """Stable reason why a job cannot enter physical execution."""

    FLIGHT_MODEL_UNKNOWN = "flight_model_unknown"
    FLIGHT_PROFILE_UNREGISTERED = "flight_profile_unregistered"
    FLIGHT_SETTINGS_INVALID = "flight_settings_invalid"
    FLIGHT_RECOMPUTATION_FAILED = "flight_recomputation_failed"
    FLIGHT_EVIDENCE_MISMATCH = "flight_evidence_mismatch"


class RegionalGroundProductionPreflightError(RuntimeError):
    """Typed, non-sensitive rejection of an unqualified execution profile."""

    def __init__(
        self,
        reason: ProductionRunnerPreflightReason,
        model_id: str,
        model_version: str,
    ) -> None:
        if type(reason) is not ProductionRunnerPreflightReason:
            raise TypeError("reason must be an exact ProductionRunnerPreflightReason")
        self.reason = reason
        self.model_id = model_id
        self.model_version = model_version
        super().__init__(
            "regional-ground production preflight rejected flight profile "
            f"{model_id!r} version {model_version!r}: {reason.value}"
        )


def preflight_regional_ground_production_job(
    job: RegionalGroundExecutionJob,
) -> FlightResult:
    """Return flight only when a registered profile reproduces both digests.

    The generic numeric ``flight.settings`` mapping alone is not execution
    authority. The separate registry must recognize its exact model/version,
    validate its exact schema, recompute with bound solver/surface semantics,
    and match both declared digests.
    """
    if type(job) is not RegionalGroundExecutionJob:
        raise TypeError("job must be an exact RegionalGroundExecutionJob")
    job.__post_init__()

    try:
        return recompute_qualified_flight_result(
            job.launch.launch,
            job.transfer,
            job.flight,
        )
    except FlightExecutionProfileQualificationError as error:
        reason = _preflight_reason(job, error.qualification.reason)
        raise RegionalGroundProductionPreflightError(
            reason,
            job.flight.model_id,
            job.flight.model_version,
        ) from error


def _preflight_reason(
    job: RegionalGroundExecutionJob,
    reason: FlightExecutionQualificationReason,
) -> ProductionRunnerPreflightReason:
    if reason is FlightExecutionQualificationReason.PROFILE_NOT_REGISTERED:
        try:
            FlightModelType(job.flight.model_id)
        except ValueError:
            return ProductionRunnerPreflightReason.FLIGHT_MODEL_UNKNOWN
        return ProductionRunnerPreflightReason.FLIGHT_PROFILE_UNREGISTERED
    if reason is FlightExecutionQualificationReason.SETTINGS_SCHEMA_INVALID:
        return ProductionRunnerPreflightReason.FLIGHT_SETTINGS_INVALID
    if reason is FlightExecutionQualificationReason.RECOMPUTATION_FAILED:
        return ProductionRunnerPreflightReason.FLIGHT_RECOMPUTATION_FAILED
    if reason in {
        FlightExecutionQualificationReason.TRAJECTORY_DIGEST_MISMATCH,
        FlightExecutionQualificationReason.RESULT_DIGEST_MISMATCH,
    }:
        return ProductionRunnerPreflightReason.FLIGHT_EVIDENCE_MISMATCH
    raise AssertionError("qualified flight cannot produce a preflight error")


def _raise_if_cancelled(
    hooks: GroundRegionalVariationHooks,
    total: int,
) -> None:
    callback = hooks.cancellation_requested
    if callback is None:
        return
    try:
        requested = callback()
        if type(requested) is not bool:
            raise TypeError("cancellation callback must return an exact bool")
    except Exception as error:
        failure = GroundRegionalVariationFailed(
            GroundRegionalVariationFailureStage.CANCELLATION_CALLBACK,
            0,
            total,
            error,
        )
        raise failure from error
    if requested:
        raise GroundRegionalVariationCancelled(0, total)


def run_regional_ground_production_job(
    job: RegionalGroundExecutionJob,
    hooks: GroundRegionalVariationHooks,
) -> RegionalGroundExecutionResult:
    """Run only a qualified job, publishing no partial result on rejection."""
    if type(job) is not RegionalGroundExecutionJob:
        raise TypeError("job must be an exact RegionalGroundExecutionJob")
    if type(hooks) is not GroundRegionalVariationHooks:
        raise TypeError("hooks must be exact GroundRegionalVariationHooks")
    job.__post_init__()
    hooks.__post_init__()
    total = job.execution_options.max_trials
    _raise_if_cancelled(hooks, total)
    try:
        preflight_regional_ground_production_job(job)
    except RegionalGroundProductionPreflightError as error:
        failure = GroundRegionalVariationFailed(
            GroundRegionalVariationFailureStage.PREFLIGHT,
            0,
            total,
            error,
        )
        raise failure from error

    raise AssertionError("qualified production execution is not yet registered")


__all__ = [
    "ProductionRunnerPreflightReason",
    "RegionalGroundProductionPreflightError",
    "preflight_regional_ground_production_job",
    "run_regional_ground_production_job",
]
