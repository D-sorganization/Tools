"""One-way compatibility projection into the existing flight metric DTO."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from shared.python.swing_sim.flight.result_metrics import GroundModelResult

from .contract_records import GroundSimulationResult
from .contract_types import GroundResultStatus, GroundTerminationReason
from .result_types import GroundSummary

if TYPE_CHECKING:
    from .study_record import GroundStudyProjection


def _metric_result(
    model_id: str,
    model_version: str,
    summary: GroundSummary,
) -> GroundModelResult:
    return GroundModelResult(
        model_id=f"{model_id}@{model_version}",
        total_distance_m=summary.total_distance_m,
        roll_distance_m=summary.roll_distance_m,
        bounce_count=summary.bounce_count,
        final_offline_m=summary.final_offline_m,
    )


def to_ground_model_result(result: GroundSimulationResult) -> GroundModelResult:
    """Compatibility-only projection without material-profile qualification."""
    warnings.warn(
        "to_ground_model_result is unqualified compatibility output; "
        "use the study adapter for qualification-sensitive consumers",
        DeprecationWarning,
        stacklevel=2,
    )
    if result is None:
        raise ValueError("result must be provided")
    rest_terminated = result.termination.reason is GroundTerminationReason.REST
    if (
        result.status is not GroundResultStatus.COMPLETE
        or not rest_terminated
        or result.summary is None
    ):
        raise ValueError(
            "only complete, rest-terminated qualified ground results can be projected"
        )
    summary = result.summary
    return _metric_result(result.model_id, result.model_version, summary)


def qualified_study_to_ground_model_result(
    study: GroundStudyProjection,
) -> GroundModelResult:
    """Populate the legacy DTO only from an objective-qualified study."""
    from .study_record import GroundStudyProjection

    if type(study) is not GroundStudyProjection:
        raise TypeError("study must use the exact GroundStudyProjection type")
    if not study.solver_eligibility.eligible or study.metrics is None:
        raise ValueError("study must be solver-eligible with numeric metrics")
    return _metric_result(study.model_id, study.model_version, study.metrics.summary)


__all__ = ["qualified_study_to_ground_model_result", "to_ground_model_result"]
