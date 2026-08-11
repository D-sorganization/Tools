"""Strict, presentation-neutral readback for regional execution evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.python.swing_sim.ground import (
    MAX_REGIONAL_GROUND_EXECUTION_WIRE_BYTES,
    GroundRegionalMaterialPlanRequest,
    RegionalGroundExecutionResult,
    regional_ground_execution_result_from_json,
)

from .bounded_text_files import read_bounded_utf8


@dataclass(frozen=True)
class RegionalExecutionReadback:
    """Small UI-neutral projection of one frozen execution envelope."""

    status: str
    failure_reason: str | None
    plan_id: str
    surface_id: str
    model_id: str
    model_version: str
    termination_reason: str | None
    transition_count: int
    skid_distance_m: float | None
    roll_distance_m: float | None
    total_distance_m: float | None
    executor_source_revision: str
    executor_input_sha256: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class RegionalExecutionEvidence:
    """Validated evidence plus its derived display projection."""

    result: RegionalGroundExecutionResult
    readback: RegionalExecutionReadback


def regional_execution_readback(
    result: RegionalGroundExecutionResult,
    current_plan: GroundRegionalMaterialPlanRequest,
) -> RegionalExecutionReadback:
    """Bind executor evidence to the exact current plan and summarize it."""
    if type(result) is not RegionalGroundExecutionResult:
        raise TypeError("result must be an exact RegionalGroundExecutionResult")
    if type(current_plan) is not GroundRegionalMaterialPlanRequest:
        raise TypeError(
            "current_plan must be an exact GroundRegionalMaterialPlanRequest"
        )
    if result.regional_plan != current_plan:
        raise ValueError("execution evidence does not match the current regional plan")
    ground = result.ground_result
    summary = None if ground is None else ground.summary
    return RegionalExecutionReadback(
        status=result.status.value,
        failure_reason=None
        if result.failure_reason is None
        else result.failure_reason.value,
        plan_id=result.plan_id,
        surface_id=result.surface_id,
        model_id=result.model_id,
        model_version=result.model_version,
        termination_reason=None if ground is None else ground.termination.reason.value,
        transition_count=len(result.transitions),
        skid_distance_m=None if summary is None else summary.skid_distance_m,
        roll_distance_m=None if summary is None else summary.roll_distance_m,
        total_distance_m=None if summary is None else summary.total_distance_m,
        executor_source_revision=result.executor_provenance.source_revision,
        executor_input_sha256=result.executor_provenance.input_sha256,
        limitations=result.limitations,
    )


def read_regional_execution_evidence(
    source: str | Path,
    current_plan: GroundRegionalMaterialPlanRequest,
) -> RegionalExecutionEvidence:
    """Read one bounded UTF-8 snapshot, parse it strictly, and bind its plan."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"regional execution evidence does not exist: {path}")
    text = read_bounded_utf8(
        path,
        MAX_REGIONAL_GROUND_EXECUTION_WIRE_BYTES,
        "regional execution evidence",
    )
    result = regional_ground_execution_result_from_json(text)
    return RegionalExecutionEvidence(
        result=result,
        readback=regional_execution_readback(result, current_plan),
    )


__all__ = [
    "RegionalExecutionEvidence",
    "RegionalExecutionReadback",
    "read_regional_execution_evidence",
    "regional_execution_readback",
]
