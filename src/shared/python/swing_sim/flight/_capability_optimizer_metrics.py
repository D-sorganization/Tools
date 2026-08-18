"""Private landing extraction and risk metrics for capability optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from shared.python.swing_sim.solver.targets import TargetRegion

from .capability_contract import CapabilityObjective, OptimizationRequest
from .inverse_contract import EvaluationStatus, SolverEvaluation
from .result_contract import FlightMetricId


@dataclass(frozen=True)
class _Landing:
    carry_m: float
    offline_m: float


@dataclass(frozen=True)
class _RiskMetrics:
    mean_carry_m: float
    expected_miss_m: float
    hold_probability: float
    dispersion_rms_m: float
    cvar_miss_m: float
    downside_carry_m: float


def _landing(evaluation: SolverEvaluation) -> _Landing | None:
    if evaluation.status is not EvaluationStatus.COMPLETE:
        return None
    metrics = {item.metric_id: item.value for item in evaluation.metrics}
    carry = metrics.get(FlightMetricId.CARRY_DISTANCE)
    offline = metrics.get(FlightMetricId.CARRY_OFFLINE)
    if (
        carry is None
        or offline is None
        or not math.isfinite(carry)
        or not math.isfinite(offline)
    ):
        return None
    return _Landing(carry, offline)


def _target(request: OptimizationRequest) -> TargetRegion:
    from shared.python.swing_sim.solver.targets import TargetRegion

    value = request.target
    return TargetRegion(
        cast(Literal["green", "fairway"], value.kind),
        value.distance_m,
        value.radius_m,
        value.lateral_m,
        value.band_half_length_m,
        value.half_width_m,
    )


def _tail_mean(values: list[float], alpha: float, *, reverse: bool) -> float:
    count = max(1, math.ceil(len(values) * (1.0 - alpha)))
    return sum(sorted(values, reverse=reverse)[:count]) / count


def _score(
    objective: CapabilityObjective,
    risk: _RiskMetrics,
    target_distance: float,
) -> float:
    if objective is CapabilityObjective.MAXIMIZE_CARRY:
        return -risk.mean_carry_m
    if objective is CapabilityObjective.MINIMIZE_EXPECTED_MISS:
        return risk.expected_miss_m
    if objective is CapabilityObjective.MAXIMIZE_TARGET_HOLD:
        return -risk.hold_probability + risk.expected_miss_m * 1e-6
    if objective is CapabilityObjective.MINIMIZE_VARIABILITY:
        return risk.dispersion_rms_m
    if objective is CapabilityObjective.MINIMIZE_DOWNSIDE:
        return risk.cvar_miss_m + risk.downside_carry_m
    return abs(risk.mean_carry_m - target_distance) + risk.dispersion_rms_m


def _risk_metrics(
    landings: tuple[_Landing, ...], request: OptimizationRequest
) -> _RiskMetrics:
    target = _target(request)
    center_carry, center_offline = target.center
    carries = [item.carry_m for item in landings]
    offlines = [item.offline_m for item in landings]
    misses = [
        math.hypot(item.carry_m - center_carry, item.offline_m - center_offline)
        for item in landings
    ]
    mean_carry = sum(carries) / len(carries)
    mean_offline = sum(offlines) / len(offlines)
    dispersion = math.sqrt(
        sum(
            (carry - mean_carry) ** 2 + (offline - mean_offline) ** 2
            for carry, offline in zip(carries, offlines, strict=True)
        )
        / len(carries)
    )
    hold = sum(
        target.contains(item.carry_m, item.offline_m) for item in landings
    ) / len(landings)
    return _RiskMetrics(
        mean_carry,
        sum(misses) / len(misses),
        hold,
        dispersion,
        _tail_mean(misses, request.cvar_alpha, reverse=True),
        max(
            0.0,
            center_carry - _tail_mean(carries, request.cvar_alpha, reverse=False),
        ),
    )


__all__: list[str] = []
