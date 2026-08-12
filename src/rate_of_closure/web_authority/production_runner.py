"""Fail-closed production-runner qualification for regional-ground jobs.

The v1 execution job records flight evidence, but it does not yet bind a
versioned model input profile to executable solver semantics.  This module is
the production boundary: it rejects unqualified profiles before flight or
ground physics can run and preserves typed cancellation/failure behavior.
"""

from __future__ import annotations

from enum import StrEnum
from typing import NoReturn

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
from shared.python.swing_sim.flight import FlightModelType


class ProductionRunnerPreflightReason(StrEnum):
    """Stable reason why a job cannot enter physical execution."""

    FLIGHT_MODEL_UNKNOWN = "flight_model_unknown"
    FLIGHT_PROFILE_UNREGISTERED = "flight_profile_unregistered"


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
) -> NoReturn:
    """Reject until a versioned, recomputable flight profile is registered.

    A future qualified profile must bind the exact model ID and version,
    settings schema, solver/surface semantics, and recomputable trajectory and
    result digests.  The generic numeric ``flight.settings`` mapping does not
    establish any of those semantics, so recognizing a model ID alone is not
    permission to execute it.
    """
    if type(job) is not RegionalGroundExecutionJob:
        raise TypeError("job must be an exact RegionalGroundExecutionJob")
    job.__post_init__()

    try:
        FlightModelType(job.flight.model_id)
    except ValueError as error:
        raise RegionalGroundProductionPreflightError(
            ProductionRunnerPreflightReason.FLIGHT_MODEL_UNKNOWN,
            job.flight.model_id,
            job.flight.model_version,
        ) from error

    raise RegionalGroundProductionPreflightError(
        ProductionRunnerPreflightReason.FLIGHT_PROFILE_UNREGISTERED,
        job.flight.model_id,
        job.flight.model_version,
    )


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
