"""All-trial decision statistics for conditional chip-shot forgiveness studies.

The module deliberately owns no swing or turf physics.  It summarizes typed
trial records produced by an adapter and keeps misses and numerical failures in
every probability, loss, confidence, and convergence denominator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import numpy as np

_WILSON_Z_95 = 1.959963984540054
_BOOTSTRAP_CI = (0.025, 0.975)
_MIN_BOOTSTRAP_SAMPLES = 64
_UINT32_MASK = 0xFFFFFFFF
_MULBERRY32_INCREMENT = 0x6D2B79F5


class ChipTrialCohort(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Mutually exclusive contact or evaluation result for one chip trial."""

    BALL_FIRST = "ball_first"
    BALL_ONLY = "ball_only"
    GROUND_FIRST = "ground_first"
    SIMULTANEOUS = "simultaneous_or_grazing"
    GROUND_ONLY_MISS = "ground_only_miss"
    NO_CONTACT_MISS = "no_contact_miss"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True)
class ChipStudyMetadata:
    """Provenance and qualification boundary for one candidate study."""

    candidate_id: str
    plan_schema: str
    coordinate_frame: str
    seed: int
    noise_model_id: str
    objective_id: str
    turf_profile_id: str
    turf_calibration_status: str
    solver_id: str
    sampling_design: str
    inference_method_id: str
    limitations: str

    def __post_init__(self) -> None:
        """Require stable, nonempty identifiers and a reproducible seed."""
        for name in (
            "candidate_id",
            "plan_schema",
            "coordinate_frame",
            "noise_model_id",
            "objective_id",
            "turf_profile_id",
            "turf_calibration_status",
            "solver_id",
            "sampling_design",
            "inference_method_id",
            "limitations",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")


@dataclass(frozen=True)
class ChipTrialRecord:
    """One retained cohort, scalar metrics, constraint state, and loss."""

    trial_index: int
    cohort: ChipTrialCohort
    loss: float
    constraint_violated: bool
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    diagnostic: str | None = None
    turf_contact_status: str | None = None

    def __post_init__(self) -> None:
        """Validate the trial boundary without fabricating unavailable metrics."""
        if self.trial_index < 0:
            raise ValueError("trial_index must be >= 0")
        if not isinstance(self.cohort, ChipTrialCohort):
            raise TypeError("cohort must be a ChipTrialCohort")
        loss = float(self.loss)
        if not math.isfinite(loss):
            raise ValueError("loss must be finite")
        if loss < 0.0:
            raise ValueError("loss must be >= 0")
        normalized = dict(self.metrics)
        if not all(isinstance(name, str) and name for name in normalized):
            raise ValueError("metric names must be nonempty strings")
        if not all(
            value is None or math.isfinite(float(value))
            for value in normalized.values()
        ):
            raise ValueError("available metrics must be finite")
        if self.turf_contact_status is not None and (
            not isinstance(self.turf_contact_status, str)
            or not self.turf_contact_status.strip()
        ):
            raise ValueError("turf_contact_status must be a nonempty string or None")
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "metrics", MappingProxyType(normalized))


@dataclass(frozen=True)
class BinomialEstimate:
    """Count, all-trial probability, and two-sided 95% Wilson interval."""

    count: int
    probability: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class ConvergencePoint:
    """Running all-trial loss estimate at a declared prefix size."""

    sample_count: int
    mean_loss: float
    standard_error: float | None


@dataclass(frozen=True)
class MetricDistribution:
    """Availability-aware quantiles for one optional scalar metric."""

    name: str
    support_count: int
    unavailable_count: int
    p05: float | None
    p50: float | None
    p95: float | None


@dataclass(frozen=True)
class ChipStudySummary:
    """Decision-ready result for one declared candidate and population."""

    metadata: ChipStudyMetadata
    sample_count: int
    cohorts: Mapping[ChipTrialCohort, BinomialEstimate]
    expected_loss: float
    expected_loss_ci_low: float
    expected_loss_ci_high: float
    cvar_loss: float
    cvar_tail_fraction: float
    constraint_violation_rate: float
    clean_contact_probability: float
    metric_distributions: tuple[MetricDistribution, ...]
    convergence: tuple[ConvergencePoint, ...]
    supports_turf_rankings: bool
    ranking_scope: str

    def __post_init__(self) -> None:
        """Freeze cohort estimates so callers cannot mutate study evidence."""
        object.__setattr__(self, "cohorts", MappingProxyType(dict(self.cohorts)))


def _wilson_interval(count: int, sample_count: int) -> BinomialEstimate:
    probability = count / sample_count
    z_squared = _WILSON_Z_95**2
    denominator = 1.0 + z_squared / sample_count
    center = (probability + z_squared / (2.0 * sample_count)) / denominator
    spread = (
        _WILSON_Z_95
        * math.sqrt(
            probability * (1.0 - probability) / sample_count
            + z_squared / (4.0 * sample_count**2)
        )
        / denominator
    )
    return BinomialEstimate(
        count=count,
        probability=probability,
        ci_low=max(0.0, center - spread),
        ci_high=min(1.0, center + spread),
    )


def _bootstrap_mean_interval(
    losses: np.ndarray, *, seed: int, sample_count: int
) -> tuple[float, float]:
    state = (seed & _UINT32_MASK) or _MULBERRY32_INCREMENT
    means: np.ndarray = np.empty(sample_count, dtype=float)
    for sample_index in range(sample_count):
        total = 0.0
        for _ in losses:
            state = (state + _MULBERRY32_INCREMENT) & _UINT32_MASK
            value = ((state ^ (state >> 15)) * (state | 1)) & _UINT32_MASK
            value ^= (
                value + (((value ^ (value >> 7)) * (value | 61)) & _UINT32_MASK)
            ) & _UINT32_MASK
            random_fraction = ((value ^ (value >> 14)) & _UINT32_MASK) / 4_294_967_296
            total += float(losses[math.floor(random_fraction * len(losses))])
        means[sample_index] = total / len(losses)
    bounds = np.asarray(np.quantile(means, _BOOTSTRAP_CI), dtype=float)
    return float(bounds[0]), float(bounds[1])


def _cvar(losses: np.ndarray, tail_fraction: float) -> float:
    tail_count = max(1, math.ceil(tail_fraction * len(losses)))
    worst = np.partition(losses, len(losses) - tail_count)[-tail_count:]
    return float(np.mean(worst))


def _checkpoint_counts(sample_count: int) -> tuple[int, ...]:
    checkpoints = {sample_count}
    power = 1
    while power < sample_count:
        checkpoints.add(power)
        power *= 2
    return tuple(sorted(checkpoints))


def _convergence(losses: np.ndarray) -> tuple[ConvergencePoint, ...]:
    points = []
    for count in _checkpoint_counts(len(losses)):
        prefix = losses[:count]
        standard_error = (
            None if count < 2 else float(np.std(prefix, ddof=1) / math.sqrt(count))
        )
        points.append(ConvergencePoint(count, float(np.mean(prefix)), standard_error))
    return tuple(points)


def _metric_distributions(
    records: Sequence[ChipTrialRecord],
) -> tuple[MetricDistribution, ...]:
    names = sorted({name for record in records for name in record.metrics})
    distributions = []
    for name in names:
        values = [
            float(value)
            for record in records
            if (value := record.metrics.get(name)) is not None
        ]
        quantiles: tuple[float | None, float | None, float | None]
        if values:
            raw_quantiles = np.asarray(
                np.quantile(values, (0.05, 0.5, 0.95)), dtype=float
            )
            quantiles = (
                float(raw_quantiles[0]),
                float(raw_quantiles[1]),
                float(raw_quantiles[2]),
            )
        else:
            quantiles = (None, None, None)
        distributions.append(
            MetricDistribution(
                name=name,
                support_count=len(values),
                unavailable_count=len(records) - len(values),
                p05=quantiles[0],
                p50=quantiles[1],
                p95=quantiles[2],
            )
        )
    return tuple(distributions)


def _ranking_scope(metadata: ChipStudyMetadata) -> tuple[bool, str]:
    calibrated = metadata.turf_calibration_status == "calibrated"
    if calibrated:
        return True, (
            "Conditional ranking for the declared noise model, calibrated turf "
            "profile, objective, and solver only."
        )
    return False, (
        f"Turf ranking disabled because profile status is "
        f"{metadata.turf_calibration_status}; non-turf comparisons remain "
        "conditional on the declared noise model and objective."
    )


def summarize_chip_trials(
    metadata: ChipStudyMetadata,
    records: Sequence[ChipTrialRecord],
    *,
    cvar_tail_fraction: float = 0.1,
    bootstrap_samples: int = 2_000,
) -> ChipStudySummary:
    """Summarize all trials with Wilson, bootstrap, CVaR, and convergence evidence."""
    if not isinstance(metadata, ChipStudyMetadata):
        raise TypeError("metadata must be ChipStudyMetadata")
    retained = tuple(records)
    if not retained:
        raise ValueError("records must not be empty")
    if tuple(record.trial_index for record in retained) != tuple(range(len(retained))):
        raise ValueError("records must be in canonical trial order")
    if not 0.0 < cvar_tail_fraction <= 1.0:
        raise ValueError("cvar_tail_fraction must be in (0, 1]")
    if bootstrap_samples < _MIN_BOOTSTRAP_SAMPLES:
        raise ValueError(f"bootstrap_samples must be >= {_MIN_BOOTSTRAP_SAMPLES}")
    losses = np.asarray([record.loss for record in retained], dtype=float)
    cohort_estimates = {
        cohort: _wilson_interval(
            sum(record.cohort is cohort for record in retained), len(retained)
        )
        for cohort in ChipTrialCohort
    }
    ci_low, ci_high = _bootstrap_mean_interval(
        losses, seed=metadata.seed, sample_count=bootstrap_samples
    )
    clean_count = sum(
        record.cohort in (ChipTrialCohort.BALL_FIRST, ChipTrialCohort.BALL_ONLY)
        for record in retained
    )
    supports_rankings, ranking_scope = _ranking_scope(metadata)
    return ChipStudySummary(
        metadata=metadata,
        sample_count=len(retained),
        cohorts=cohort_estimates,
        expected_loss=float(np.mean(losses)),
        expected_loss_ci_low=ci_low,
        expected_loss_ci_high=ci_high,
        cvar_loss=_cvar(losses, cvar_tail_fraction),
        cvar_tail_fraction=cvar_tail_fraction,
        constraint_violation_rate=(
            sum(record.constraint_violated for record in retained) / len(retained)
        ),
        clean_contact_probability=clean_count / len(retained),
        metric_distributions=_metric_distributions(retained),
        convergence=_convergence(losses),
        supports_turf_rankings=supports_rankings,
        ranking_scope=ranking_scope,
    )


__all__ = [
    "BinomialEstimate",
    "ChipStudyMetadata",
    "ChipStudySummary",
    "ChipTrialCohort",
    "ChipTrialRecord",
    "ConvergencePoint",
    "MetricDistribution",
    "summarize_chip_trials",
]
