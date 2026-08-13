"""Adapt qualified regional-ground outcomes to existing metric study contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TypeAlias

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
    FlightMetricInputs,
    FlightRegionalGroundPipelineResult,
)
from shared.python.swing_sim.ground import (
    BounceTerminationReason,
    GroundResultStatus,
    GroundSimulationResult,
    GroundTerminationReason,
    RegionalGroundExecutionStatus,
    to_ground_model_result,
)

MAX_REGIONAL_GROUND_STUDY_ROWS = 100_000
_ADAPTER_ID = "flight-regional-ground/scalar-ensemble/v1"

RegionalGroundStudyOutcome: TypeAlias = (
    FlightRegionalGroundPipelineResult | FlightGroundTransferError
)

_STAGES = (ScalarEnsembleStage("ground_stop", "Ground Stop"),)
_CATEGORIES = (
    ScalarVariableCategory("flight_metric", "Canonical Flight Metrics"),
    ScalarVariableCategory("ground_detail", "Ground Phase Detail"),
)
_VARIABLES = (
    ScalarVariableDefinition(
        "metric.carry_distance",
        "Carry Distance",
        "m",
        "ground_stop",
        "flight_metric",
    ),
    ScalarVariableDefinition(
        "ground.bounce_air_distance",
        "Bounce Air Distance",
        "m",
        "ground_stop",
        "ground_detail",
    ),
    ScalarVariableDefinition(
        "ground.skid_distance",
        "Skid Distance",
        "m",
        "ground_stop",
        "ground_detail",
    ),
    ScalarVariableDefinition(
        "metric.roll_distance",
        "Roll Distance",
        "m",
        "ground_stop",
        "flight_metric",
    ),
    ScalarVariableDefinition(
        "ground.surface_path_distance",
        "Surface Path Distance",
        "m",
        "ground_stop",
        "ground_detail",
    ),
    ScalarVariableDefinition(
        "metric.total_distance",
        "Total Distance",
        "m",
        "ground_stop",
        "flight_metric",
    ),
    ScalarVariableDefinition(
        "ground.final_downrange",
        "Final Downrange",
        "m",
        "ground_stop",
        "ground_detail",
    ),
    ScalarVariableDefinition(
        "metric.final_offline",
        "Final Offline",
        "m",
        "ground_stop",
        "flight_metric",
    ),
    ScalarVariableDefinition(
        "metric.bounce_count",
        "Bounce Count",
        "count",
        "ground_stop",
        "flight_metric",
    ),
)
_COHORTS = (
    ScalarCohortDefinition("complete", "Complete at Rest"),
    ScalarCohortDefinition("partial", "Partial or Censored"),
    ScalarCohortDefinition("cancelled", "Cancelled"),
    ScalarCohortDefinition("failed", "Failed"),
    ScalarCohortDefinition("unavailable", "Unavailable"),
)
_NULL_VALUES: dict[str, float | None] = {variable.key: None for variable in _VARIABLES}


def _qualified_ground_result(
    outcome: RegionalGroundStudyOutcome,
) -> GroundSimulationResult | None:
    if not isinstance(outcome, FlightRegionalGroundPipelineResult):
        return None
    regional = outcome.regional_result
    ground = outcome.ground_result
    if (
        regional is None
        or regional.status is not RegionalGroundExecutionStatus.COMPLETE
        or type(ground) is not GroundSimulationResult
        or ground.status is not GroundResultStatus.COMPLETE
        or ground.termination.reason is not GroundTerminationReason.REST
        or ground.summary is None
    ):
        return None
    return ground


def apply_regional_ground_metrics(
    inputs: FlightMetricInputs,
    outcome: RegionalGroundStudyOutcome,
) -> FlightMetricInputs:
    """Attach final ground metrics only for complete, rest-terminated evidence."""
    require(
        type(inputs) is FlightMetricInputs,
        "inputs must be an exact FlightMetricInputs",
    )
    _require_outcome(outcome)
    ground = _qualified_ground_result(outcome)
    if ground is None:
        return replace(inputs, ground_result=None)
    return inputs.with_ground_result(to_ground_model_result(ground))


def _require_outcome(outcome: object) -> None:
    require(
        type(outcome)
        in (FlightRegionalGroundPipelineResult, FlightGroundTransferError),
        "outcome must be a pipeline result or transfer failure",
    )


def _cohort(outcome: RegionalGroundStudyOutcome) -> str:
    if isinstance(outcome, FlightGroundTransferError):
        return "unavailable"
    ground = outcome.ground_result
    if _qualified_ground_result(outcome) is not None:
        return "complete"
    regional = outcome.regional_result
    if regional is not None:
        if regional.status is RegionalGroundExecutionStatus.CANCELLED:
            return "cancelled"
        if regional.status is RegionalGroundExecutionStatus.FAILED:
            return "failed"
        if ground is None or ground.summary is None:
            return "unavailable"
        return "partial"
    reason = outcome.bounce_result.result.termination.reason
    if reason is BounceTerminationReason.CANCELLED:
        return "cancelled"
    if reason is BounceTerminationReason.NUMERICAL_FAILURE:
        return "failed"
    return "partial"


def _qualification(outcome: RegionalGroundStudyOutcome) -> str:
    if isinstance(outcome, FlightGroundTransferError):
        return "unavailable"
    ground = outcome.ground_result
    if _qualified_ground_result(outcome) is not None:
        return "complete_rest"
    regional = outcome.regional_result
    if regional is not None and regional.status in (
        RegionalGroundExecutionStatus.CANCELLED,
        RegionalGroundExecutionStatus.FAILED,
    ):
        return str(regional.status.value)
    if ground is not None and ground.summary is None:
        return "summary_unavailable"
    if ground is not None or regional is None:
        return "censored"
    return "unavailable"


def _values(outcome: RegionalGroundStudyOutcome) -> dict[str, float | None]:
    ground = _qualified_ground_result(outcome)
    if ground is None or ground.summary is None:
        return dict(_NULL_VALUES)
    summary = ground.summary
    return {
        "metric.carry_distance": summary.carry_distance_m,
        "ground.bounce_air_distance": summary.bounce_air_distance_m,
        "ground.skid_distance": summary.skid_distance_m,
        "metric.roll_distance": summary.roll_distance_m,
        "ground.surface_path_distance": summary.surface_path_distance_m,
        "metric.total_distance": summary.total_distance_m,
        "ground.final_downrange": summary.final_downrange_m,
        "metric.final_offline": summary.final_offline_m,
        "metric.bounce_count": float(summary.bounce_count),
    }


def _attributes(outcome: RegionalGroundStudyOutcome) -> dict[str, str | None]:
    if isinstance(outcome, FlightGroundTransferError):
        return {
            "source_kind": "transfer_failure",
            "endpoint_qualification": "unavailable",
            "transfer_field_id": outcome.field_id.value,
            "transfer_reason": outcome.reason.value,
            "bounce_termination": None,
            "regional_status": None,
            "regional_failure_reason": None,
            "ground_status": None,
            "ground_termination": None,
            "ground_request_sha256": None,
            "bounce_execution_input_sha256": None,
            "regional_plan_sha256": None,
            "ground_model_id": None,
            "ground_model_version": None,
        }
    regional = outcome.regional_result
    ground = outcome.ground_result
    return {
        "source_kind": "pipeline",
        "endpoint_qualification": _qualification(outcome),
        "transfer_field_id": None,
        "transfer_reason": None,
        "bounce_termination": outcome.bounce_result.result.termination.reason.value,
        "regional_status": None if regional is None else regional.status.value,
        "regional_failure_reason": (
            None
            if regional is None or regional.failure_reason is None
            else regional.failure_reason.value
        ),
        "ground_status": None if ground is None else ground.status.value,
        "ground_termination": (
            None if ground is None else ground.termination.reason.value
        ),
        "ground_request_sha256": outcome.ground_request_sha256,
        "bounce_execution_input_sha256": (
            outcome.repeated_bounce_execution_input_sha256
        ),
        "regional_plan_sha256": outcome.regional_plan_sha256,
        "ground_model_id": None if ground is None else ground.model_id,
        "ground_model_version": None if ground is None else ground.model_version,
    }


def build_regional_ground_study_ensemble(
    outcomes: Iterable[RegionalGroundStudyOutcome],
    result_id: str,
    source_provenance: str,
    max_rows: int,
    *,
    series_id: str | None = None,
) -> ScalarEnsembleDataset:
    """Build a bounded nullable study without promoting censored endpoints."""
    require(
        type(max_rows) is int and 1 <= max_rows <= MAX_REGIONAL_GROUND_STUDY_ROWS,
        f"max_rows must be within [1, {MAX_REGIONAL_GROUND_STUDY_ROWS}]",
    )
    require(bool(result_id.strip()), "result_id must be nonempty")
    require(bool(source_provenance.strip()), "source_provenance must be nonempty")
    retained: list[RegionalGroundStudyOutcome] = []
    for outcome in outcomes:
        require(
            len(retained) < max_rows,
            f"regional-ground outcome exceeds max_rows {max_rows}",
        )
        _require_outcome(outcome)
        retained.append(outcome)
    require(bool(retained), "regional-ground outcomes must be nonempty")
    rows = tuple(
        ScalarEnsembleRow(
            scalar_ensemble_row_id(trial_index, series_id),
            trial_index,
            _cohort(outcome),
            _values(outcome),
            series_id,
            _attributes(outcome),
        )
        for trial_index, outcome in enumerate(retained)
    )
    return ScalarEnsembleDataset(
        SCALAR_ENSEMBLE_SCHEMA_VERSION,
        result_id,
        ScalarEnsembleProvenance(
            _ADAPTER_ID,
            FLIGHT_REGIONAL_GROUND_PIPELINE_CONTRACT_VERSION,
            source_provenance,
        ),
        _STAGES,
        _CATEGORIES,
        _VARIABLES,
        _COHORTS,
        rows,
    )


__all__ = [
    "MAX_REGIONAL_GROUND_STUDY_ROWS",
    "RegionalGroundStudyOutcome",
    "apply_regional_ground_metrics",
    "build_regional_ground_study_ensemble",
]
