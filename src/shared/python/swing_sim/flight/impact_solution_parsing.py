"""Strict wire parsers for impact solution-family result records."""

from __future__ import annotations

from typing import Any

from .impact_solution_contract import ForwardStatus, ModelManifest, _exact
from .impact_solution_result import (
    FamilyMember,
    ImpactSolutionResult,
    MetricSensitivity,
    MetricValue,
    ParameterCorrelation,
    ParameterInterval,
    RejectedCandidate,
    SolutionFamily,
)
from .inverse_contract import (
    ObjectiveMode,
    ObjectiveResidual,
    ParameterValue,
    SolverStatus,
)
from .result_contract import FlightMetricId


def _parameter(payload: dict[str, Any]) -> ParameterValue:
    _exact(payload, {"parameter_id", "unit", "value"}, "parameter value")
    return ParameterValue(
        str(payload["parameter_id"]), str(payload["unit"]), float(payload["value"])
    )


def _residual(payload: dict[str, Any]) -> ObjectiveResidual:
    _exact(
        payload,
        {
            "actual_value",
            "constraint_violation",
            "metric_id",
            "mode",
            "normalized_residual",
            "provenance",
            "target_value",
            "unit",
        },
        "objective residual",
    )
    target = payload["target_value"]
    return ObjectiveResidual(
        FlightMetricId(payload["metric_id"]),
        str(payload["unit"]),
        ObjectiveMode(payload["mode"]),
        float(payload["actual_value"]),
        None if target is None else float(target),
        float(payload["normalized_residual"]),
        float(payload["constraint_violation"]),
        str(payload["provenance"]),
    )


def _metric(payload: dict[str, Any]) -> MetricValue:
    _exact(
        payload,
        {"metric_id", "provenance", "reference_event", "unit", "value"},
        "metric value",
    )
    return MetricValue(
        FlightMetricId(payload["metric_id"]),
        str(payload["unit"]),
        float(payload["value"]),
        str(payload["reference_event"]),
        str(payload["provenance"]),
    )


def _member(payload: dict[str, Any]) -> FamilyMember:
    _exact(
        payload,
        {
            "evaluation_index",
            "feasible",
            "flight_residuals",
            "launch_residuals",
            "launch_values",
            "parameters",
            "score",
        },
        "family member",
    )
    return FamilyMember(
        payload["evaluation_index"],
        payload["feasible"],
        float(payload["score"]),
        tuple(_parameter(item) for item in payload["parameters"]),
        tuple(_metric(item) for item in payload["launch_values"]),
        tuple(_residual(item) for item in payload["launch_residuals"]),
        tuple(_residual(item) for item in payload["flight_residuals"]),
    )


def _family(payload: dict[str, Any]) -> SolutionFamily:
    _exact(
        payload,
        {
            "correlations",
            "family_id",
            "flight_residuals",
            "intervals",
            "launch_residuals",
            "members",
            "rank",
            "representative_evaluation_index",
            "sensitivities",
        },
        "solution family",
    )
    intervals = tuple(
        ParameterInterval(
            str(item["parameter_id"]),
            str(item["unit"]),
            float(item["lower_bound"]),
            float(item["upper_bound"]),
        )
        for item in payload["intervals"]
    )
    correlations = tuple(
        ParameterCorrelation(
            str(item["left_parameter_id"]),
            str(item["right_parameter_id"]),
            float(item["coefficient"]),
            item["sample_count"],
        )
        for item in payload["correlations"]
    )
    sensitivities = tuple(
        MetricSensitivity(
            str(item["parameter_id"]),
            str(item["parameter_unit"]),
            FlightMetricId(item["metric_id"]),
            str(item["metric_unit"]),
            float(item["derivative"]),
            str(item["method"]),
        )
        for item in payload["sensitivities"]
    )
    return SolutionFamily(
        str(payload["family_id"]),
        payload["rank"],
        payload["representative_evaluation_index"],
        tuple(_member(item) for item in payload["members"]),
        intervals,
        correlations,
        sensitivities,
        tuple(_residual(item) for item in payload["launch_residuals"]),
        tuple(_residual(item) for item in payload["flight_residuals"]),
    )


def _rejected(payload: dict[str, Any]) -> RejectedCandidate:
    _exact(
        payload,
        {"evaluation_index", "parameters", "reason", "status"},
        "rejected candidate",
    )
    return RejectedCandidate(
        payload["evaluation_index"],
        ForwardStatus(payload["status"]),
        str(payload["reason"]),
        tuple(_parameter(item) for item in payload["parameters"]),
    )


def parse_impact_solution_result(payload: dict[str, Any]) -> ImpactSolutionResult:
    """Parse a strict v1 impact solution result."""
    _exact(
        payload,
        {
            "evaluations_attempted",
            "families",
            "model_manifest",
            "problem_id",
            "provenance",
            "rejected_candidates",
            "schema_version",
            "status",
            "termination_reason",
        },
        "impact solution result",
    )
    if (
        not isinstance(payload["families"], list)
        or not isinstance(payload["rejected_candidates"], list)
        or not isinstance(payload["model_manifest"], dict)
        or not isinstance(payload["provenance"], dict)
    ):
        raise ValueError("impact solution result collections have invalid types")
    return ImpactSolutionResult(
        str(payload["problem_id"]),
        SolverStatus(payload["status"]),
        str(payload["termination_reason"]),
        payload["evaluations_attempted"],
        tuple(_family(item) for item in payload["families"]),
        tuple(_rejected(item) for item in payload["rejected_candidates"]),
        ModelManifest.from_dict(payload["model_manifest"]),
        tuple((str(key), str(value)) for key, value in payload["provenance"].items()),
        str(payload["schema_version"]),
    )


__all__ = ["parse_impact_solution_result"]
