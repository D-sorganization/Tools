"""Project qualified final ground endpoints onto canonical spatial targets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rate_of_closure.variation.regional_ground_study_adapter import (
    RegionalGroundStudyOutcome,
    qualified_regional_ground_result,
    regional_ground_evidence_attributes,
)
from rate_of_closure.variation.scalar_ensemble_contract import (
    SCALAR_ENSEMBLE_SCHEMA_VERSION,
    ScalarCohortDefinition,
    ScalarEnsembleDataset,
    ScalarEnsembleProvenance,
    ScalarEnsembleRow,
    ScalarEnsembleStage,
    ScalarVariableCategory,
    ScalarVariableDefinition,
    scalar_ensemble_row_id,
)
from shared.python.contracts import require
from shared.python.swing_sim.flight import (
    FLIGHT_REGIONAL_GROUND_PIPELINE_CONTRACT_VERSION,
    FlightGroundTransferError,
    FlightRegionalGroundPipelineResult,
)
from shared.python.swing_sim.ground import (
    BounceTerminationReason,
    GroundFrame,
    GroundTerminationReason,
    RegionalGroundExecutionStatus,
)
from shared.python.swing_sim.solver import SpatialTarget, TargetMiss

if TYPE_CHECKING:
    from enum import StrEnum

    from shared.python.swing_sim.solver._target_validation import Vector3
else:
    from shared.python.compatibility import StrEnum

MAX_REGIONAL_GROUND_TARGET_ROWS = 100_000
REGIONAL_GROUND_TARGET_ADAPTER_ID = "regional-ground-target/scalar-ensemble/v1"


class RegionalGroundTargetAvailability(StrEnum):
    """Typed availability for a post-ground target projection."""

    AVAILABLE = "AVAILABLE"
    AERIAL_REQUIRES_FLIGHT_TRAJECTORY = "AERIAL_REQUIRES_FLIGHT_TRAJECTORY"
    TRANSFER_ERROR = "TRANSFER_ERROR"
    BOUNCE_NOT_SETTLED = "BOUNCE_NOT_SETTLED"
    REGIONAL_CANCELLED = "REGIONAL_CANCELLED"
    REGIONAL_FAILED = "REGIONAL_FAILED"
    REGIONAL_INCOMPLETE = "REGIONAL_INCOMPLETE"
    GROUND_NOT_REST = "GROUND_NOT_REST"
    SUMMARY_UNAVAILABLE = "SUMMARY_UNAVAILABLE"


@dataclass(frozen=True)
class RegionalGroundTargetProjection:
    """Nullable target evidence with exact phase and input identities."""

    availability: RegionalGroundTargetAvailability
    frame: GroundFrame
    phase: str
    reason: str
    endpoint_app_m: Vector3 | None
    miss: TargetMiss | None
    ground_request_sha256: str | None
    bounce_execution_input_sha256: str | None
    regional_plan_sha256: str | None

    @property
    def hold(self) -> bool | None:
        """Return target acceptance only when a qualified endpoint exists."""
        return None if self.miss is None else self.miss.accepted


@dataclass(frozen=True)
class _Unavailable:
    availability: RegionalGroundTargetAvailability
    phase: str
    reason: str


_STAGES = (ScalarEnsembleStage("ground_target", "Ground Target"),)
_CATEGORIES = (ScalarVariableCategory("target", "Target Outcome"),)
_VARIABLES = tuple(
    ScalarVariableDefinition(key, label, unit, "ground_target", "target")
    for key, label, unit in (
        ("target.hold", "Target Hold", "1"),
        ("target.miss_distance", "Target Miss Distance", "m"),
        ("target.miss_downrange", "Target Downrange Miss", "m"),
        ("target.miss_elevation", "Target Elevation Miss", "m"),
        ("target.miss_lateral", "Target Lateral Miss", "m"),
    )
)
_COHORTS = (
    ScalarCohortDefinition("hold", "Target Hold"),
    ScalarCohortDefinition("miss", "Target Miss"),
    ScalarCohortDefinition("unavailable", "Unavailable"),
)
_NULL_VALUES: dict[str, float | None] = {item.key: None for item in _VARIABLES}


def _require_inputs(outcome: object, target: object) -> None:
    require(
        type(outcome)
        in (FlightRegionalGroundPipelineResult, FlightGroundTransferError),
        "outcome must be a pipeline result or transfer failure",
    )
    require(type(target) is SpatialTarget, "target must be an exact SpatialTarget")


def _unavailable(outcome: RegionalGroundStudyOutcome) -> _Unavailable:
    if isinstance(outcome, FlightGroundTransferError):
        return _Unavailable(
            RegionalGroundTargetAvailability.TRANSFER_ERROR,
            "flight_transfer",
            outcome.reason.value,
        )
    bounce_reason = outcome.bounce_result.result.termination.reason
    if bounce_reason is not BounceTerminationReason.SETTLED_TO_SKID:
        return _Unavailable(
            RegionalGroundTargetAvailability.BOUNCE_NOT_SETTLED,
            "bounce",
            bounce_reason.value,
        )
    return _regional_unavailable(outcome)


def _regional_unavailable(
    outcome: FlightRegionalGroundPipelineResult,
) -> _Unavailable:
    regional = outcome.regional_result
    if regional is None:
        raise RuntimeError("settled pipeline must retain regional evidence")
    if regional.status is RegionalGroundExecutionStatus.CANCELLED:
        reason = regional.failure_reason
        return _Unavailable(
            RegionalGroundTargetAvailability.REGIONAL_CANCELLED,
            "regional_ground",
            "cancelled" if reason is None else reason.value,
        )
    if regional.status is RegionalGroundExecutionStatus.FAILED:
        reason = regional.failure_reason
        return _Unavailable(
            RegionalGroundTargetAvailability.REGIONAL_FAILED,
            "regional_ground",
            "failed" if reason is None else reason.value,
        )
    if regional.status is not RegionalGroundExecutionStatus.COMPLETE:
        return _Unavailable(
            RegionalGroundTargetAvailability.REGIONAL_INCOMPLETE,
            "regional_ground",
            regional.status.value,
        )
    ground = outcome.ground_result
    if ground is None:
        raise RuntimeError("complete regional evidence must retain a ground result")
    if ground.termination.reason is not GroundTerminationReason.REST:
        return _Unavailable(
            RegionalGroundTargetAvailability.GROUND_NOT_REST,
            "ground",
            ground.termination.reason.value,
        )
    return _Unavailable(
        RegionalGroundTargetAvailability.SUMMARY_UNAVAILABLE,
        "ground",
        "summary_unavailable",
    )


def _digests(
    outcome: RegionalGroundStudyOutcome,
) -> tuple[str | None, str | None, str | None]:
    if isinstance(outcome, FlightGroundTransferError):
        return (None, None, None)
    return (
        outcome.ground_request_sha256,
        outcome.repeated_bounce_execution_input_sha256,
        outcome.regional_plan_sha256,
    )


def _unavailable_projection(
    outcome: RegionalGroundStudyOutcome,
    unavailable: _Unavailable,
) -> RegionalGroundTargetProjection:
    ground_digest, bounce_digest, plan_digest = _digests(outcome)
    return RegionalGroundTargetProjection(
        unavailable.availability,
        GroundFrame.TARGET,
        unavailable.phase,
        unavailable.reason,
        None,
        None,
        ground_digest,
        bounce_digest,
        plan_digest,
    )


def _available_projection(
    outcome: FlightRegionalGroundPipelineResult,
    endpoint: Vector3,
    miss: TargetMiss,
) -> RegionalGroundTargetProjection:
    ground_digest, bounce_digest, plan_digest = _digests(outcome)
    return RegionalGroundTargetProjection(
        RegionalGroundTargetAvailability.AVAILABLE,
        GroundFrame.TARGET,
        "complete",
        "complete_rest",
        endpoint,
        miss,
        ground_digest,
        bounce_digest,
        plan_digest,
    )


def project_regional_ground_target(
    outcome: RegionalGroundStudyOutcome,
    target: SpatialTarget,
) -> RegionalGroundTargetProjection:
    """Evaluate one target only after qualified complete-rest ground evidence."""
    _require_inputs(outcome, target)
    if target.kind == "aerial_waypoint":
        code = RegionalGroundTargetAvailability.AERIAL_REQUIRES_FLIGHT_TRAJECTORY
        return _unavailable_projection(
            outcome, _Unavailable(code, "target", code.value)
        )
    ground = qualified_regional_ground_result(outcome)
    if ground is None:
        return _unavailable_projection(outcome, _unavailable(outcome))
    endpoint = _landing_endpoint(target, ground.trajectory[-1].position_m)
    if not isinstance(outcome, FlightRegionalGroundPipelineResult):
        raise RuntimeError("qualified ground evidence requires a pipeline result")
    return _available_projection(outcome, endpoint, target.miss(endpoint))


def _landing_endpoint(target: SpatialTarget, ball_center: Vector3) -> Vector3:
    """Return the surface-contact point in the sole v1 target frame.

    Ground v1 and the app target contract share x-downrange/y-up/z-right.
    The terminal trajectory stores the ball centre, so replacing only y with
    the declared course-surface elevation removes radius/support height once;
    ``SpatialTarget.miss`` then applies the target geometry without a second
    terrain or frame adjustment.
    """
    return (ball_center[0], target.point.elevation_m, ball_center[2])


def _values(projection: RegionalGroundTargetProjection) -> dict[str, float | None]:
    miss = projection.miss
    if miss is None:
        return dict(_NULL_VALUES)
    return {
        "target.hold": float(miss.accepted),
        "target.miss_distance": miss.distance_m,
        "target.miss_downrange": miss.downrange_m,
        "target.miss_elevation": miss.elevation_m,
        "target.miss_lateral": miss.right_m,
    }


def _cohort(projection: RegionalGroundTargetProjection) -> str:
    hold = projection.hold
    if hold is None:
        return "unavailable"
    return "hold" if hold else "miss"


def _attributes(
    outcome: RegionalGroundStudyOutcome,
    target: SpatialTarget,
    projection: RegionalGroundTargetProjection,
) -> dict[str, str | None]:
    attributes: dict[str, str | None] = dict(
        regional_ground_evidence_attributes(outcome)
    )
    attributes.update(
        {
            "availability": projection.availability.value,
            "frame": projection.frame.value,
            "phase": projection.phase,
            "reason": projection.reason,
            "target_label": target.label,
            "target_kind": target.kind,
            "target_ground_source": target.ground_source,
        }
    )
    return attributes


def _retained_outcomes(
    outcomes: Iterable[RegionalGroundStudyOutcome],
    target: SpatialTarget,
    max_rows: int,
) -> tuple[RegionalGroundStudyOutcome, ...]:
    retained: list[RegionalGroundStudyOutcome] = []
    for outcome in outcomes:
        require(
            len(retained) < max_rows,
            f"regional-ground target outcome exceeds max_rows {max_rows}",
        )
        _require_inputs(outcome, target)
        retained.append(outcome)
    require(bool(retained), "regional-ground target outcomes must be nonempty")
    return tuple(retained)


def _rows(
    outcomes: tuple[RegionalGroundStudyOutcome, ...],
    target: SpatialTarget,
    series_id: str | None,
) -> tuple[ScalarEnsembleRow, ...]:
    rows = []
    for trial_index, outcome in enumerate(outcomes):
        projection = project_regional_ground_target(outcome, target)
        rows.append(
            ScalarEnsembleRow(
                scalar_ensemble_row_id(trial_index, series_id),
                trial_index,
                _cohort(projection),
                _values(projection),
                series_id,
                _attributes(outcome, target, projection),
            )
        )
    return tuple(rows)


def build_regional_ground_target_ensemble(
    outcomes: Iterable[RegionalGroundStudyOutcome],
    target: SpatialTarget,
    result_id: str,
    source_provenance: str,
    max_rows: int,
    *,
    series_id: str | None = None,
) -> ScalarEnsembleDataset:
    """Build a bounded ordered target dataset without promoting censored rows."""
    require(type(target) is SpatialTarget, "target must be an exact SpatialTarget")
    require(
        type(max_rows) is int and 1 <= max_rows <= MAX_REGIONAL_GROUND_TARGET_ROWS,
        f"max_rows must be within [1, {MAX_REGIONAL_GROUND_TARGET_ROWS}]",
    )
    require(
        isinstance(result_id, str) and bool(result_id.strip()),
        "result_id must be nonempty",
    )
    require(
        isinstance(source_provenance, str) and bool(source_provenance.strip()),
        "source_provenance must be nonempty",
    )
    retained = _retained_outcomes(outcomes, target, max_rows)
    return ScalarEnsembleDataset(
        SCALAR_ENSEMBLE_SCHEMA_VERSION,
        result_id,
        ScalarEnsembleProvenance(
            REGIONAL_GROUND_TARGET_ADAPTER_ID,
            FLIGHT_REGIONAL_GROUND_PIPELINE_CONTRACT_VERSION,
            source_provenance,
        ),
        _STAGES,
        _CATEGORIES,
        _VARIABLES,
        _COHORTS,
        _rows(retained, target, series_id),
    )


__all__ = [
    "MAX_REGIONAL_GROUND_TARGET_ROWS",
    "REGIONAL_GROUND_TARGET_ADAPTER_ID",
    "RegionalGroundTargetAvailability",
    "RegionalGroundTargetProjection",
    "build_regional_ground_target_ensemble",
    "project_regional_ground_target",
]
