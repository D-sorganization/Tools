"""One-way compatibility projection into the existing flight metric DTO."""

from __future__ import annotations

from shared.python.swing_sim.flight.result_metrics import GroundModelResult

from .contract_records import GroundSimulationResult
from .contract_types import GroundResultStatus


def to_ground_model_result(result: GroundSimulationResult) -> GroundModelResult:
    """Project qualified summary values without inferring missing distances."""
    if result is None:
        raise ValueError("result must be provided")
    if result.status is not GroundResultStatus.COMPLETE or result.summary is None:
        raise ValueError("only complete qualified ground results can be projected")
    summary = result.summary
    return GroundModelResult(
        model_id=f"{result.model_id}@{result.model_version}",
        total_distance_m=summary.total_distance_m,
        roll_distance_m=summary.roll_distance_m,
        bounce_count=summary.bounce_count,
        final_offline_m=summary.final_offline_m,
    )


__all__ = ["to_ground_model_result"]
