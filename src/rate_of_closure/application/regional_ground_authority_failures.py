"""Typed terminal conversion for the regional-ground authority client."""

from __future__ import annotations

from typing import NoReturn

from rate_of_closure.application.regional_ground_authority_status import (
    AuthorityJobSnapshot,
)
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
)


def authority_failure_stage(
    status: AuthorityJobSnapshot,
) -> GroundRegionalVariationFailureStage:
    """Map the public authority stage onto the existing PyQt terminal contract."""
    failure = status.failure
    if failure is None:
        return GroundRegionalVariationFailureStage.VALIDATION
    mapping = {
        "cancellation_callback": (
            GroundRegionalVariationFailureStage.CANCELLATION_CALLBACK
        ),
        "executor": GroundRegionalVariationFailureStage.EXECUTOR,
        "preflight": GroundRegionalVariationFailureStage.PREFLIGHT,
        "validation": GroundRegionalVariationFailureStage.VALIDATION,
        "progress_callback": GroundRegionalVariationFailureStage.PROGRESS_CALLBACK,
        "publication": GroundRegionalVariationFailureStage.PUBLICATION,
        "runner": GroundRegionalVariationFailureStage.EXECUTOR,
        "result_validation": GroundRegionalVariationFailureStage.VALIDATION,
    }
    return mapping[failure.stage]


def fail_regional_ground_authority(
    stage: GroundRegionalVariationFailureStage,
    completed: int,
    job: RegionalGroundExecutionJob,
    cause: Exception,
) -> NoReturn:
    """Raise one existing typed terminal without publishing partial data."""
    failure = GroundRegionalVariationFailed(
        stage, completed, job.execution_options.max_trials, cause
    )
    failure.__cause__ = cause
    raise failure from cause


__all__ = ["authority_failure_stage", "fail_regional_ground_authority"]
