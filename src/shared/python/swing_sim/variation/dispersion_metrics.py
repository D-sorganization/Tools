"""Confidence-scaled dispersion and ranked quiet-zone analysis.

The immutable results are plot-ready so UI adapters do not need to recreate
the statistical conventions.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from shared.python.contracts import require

from .dispersion_metric_types import (
    ELLIPSOID_VOLUME,
    ESTIMABLE,
    GAUSSIAN_POSITION_CONTENT_REGION,
    INSUFFICIENT_SAMPLES,
    INVALID_COVARIANCE,
    LARGEST_PRINCIPAL_SIGMA,
    MIN_SAMPLES_FOR_FULL_3D_COVARIANCE,
    RANK_DEFICIENT,
    RMS_RADIUS,
    ConfidenceEllipsoidSeries,
    DispersionMetricSeries,
    LowVariabilityMetricCriteria,
    RankedLowVariabilityInterval,
    validated_confidence_level,
)
from .ensemble_types import (
    CARTESIAN_DIMENSIONS,
    MIN_TRIALS_FOR_COVARIANCE,
    PositionDispersion,
)

_FOUR_THIRDS_PI = 4.0 * math.pi / 3.0
_QUANTILE_ITERATIONS = 96


def _chi_square_three_cdf(value: float) -> float:
    """Return the exact chi-square CDF for three degrees of freedom."""
    root = math.sqrt(value / 2.0)
    return math.erf(root) - math.sqrt(2.0 * value / math.pi) * math.exp(-value / 2.0)


def _chi_square_three_quantile(probability: float) -> float:
    """Invert the 3-D chi-square CDF by deterministic monotone bisection."""
    lower = 0.0
    upper = 1.0
    while _chi_square_three_cdf(upper) < probability:
        upper *= 2.0
    for _ in range(_QUANTILE_ITERATIONS):
        midpoint = (lower + upper) / 2.0
        if _chi_square_three_cdf(midpoint) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _adequacy(count: int, eigenvalues: np.ndarray) -> str:
    """Classify whether a sample supports a full three-dimensional ellipsoid."""
    if count < MIN_TRIALS_FOR_COVARIANCE:
        return INSUFFICIENT_SAMPLES
    if eigenvalues.shape != (CARTESIAN_DIMENSIONS,) or not np.all(
        np.isfinite(eigenvalues)
    ):
        return INVALID_COVARIANCE
    largest = max(float(np.max(eigenvalues)), 0.0)
    tolerance = np.finfo(float).eps * CARTESIAN_DIMENSIONS * largest
    if (
        count < MIN_SAMPLES_FOR_FULL_3D_COVARIANCE
        or largest == 0.0
        or np.count_nonzero(eigenvalues > tolerance) < CARTESIAN_DIMENSIONS
    ):
        return RANK_DEFICIENT
    return ESTIMABLE


def build_confidence_ellipsoids(
    dispersion: PositionDispersion,
    point_id: str,
    confidence_level: float = 0.95,
) -> ConfidenceEllipsoidSeries:
    """Scale covariance eigenaxes by the 3-D chi-square content quantile."""
    require(point_id in dispersion.point_ids, "unknown point_id", point_id)
    confidence = validated_confidence_level(confidence_level)
    quantile = _chi_square_three_quantile(confidence)
    radius_scale = math.sqrt(quantile)
    point_index = dispersion.point_index(point_id)
    eigenvalues = dispersion.eigenvalues_m2[:, point_index]
    counts = dispersion.count[:, point_index]
    adequacy = tuple(
        _adequacy(int(count), values)
        for count, values in zip(counts, eigenvalues, strict=True)
    )
    with np.errstate(invalid="ignore"):
        semi_axes = radius_scale * np.sqrt(np.maximum(eigenvalues, 0.0))
    volume = _FOUR_THIRDS_PI * np.prod(semi_axes, axis=1)
    volume = np.where(np.asarray(adequacy) == ESTIMABLE, volume, np.nan)
    return ConfidenceEllipsoidSeries(
        point_id=point_id,
        coordinate_frame=dispersion.coordinate_frame,
        interpretation=GAUSSIAN_POSITION_CONTENT_REGION,
        confidence_level=confidence,
        degrees_of_freedom=CARTESIAN_DIMENSIONS,
        chi_square_quantile=quantile,
        radius_scale=radius_scale,
        minimum_samples_for_full_rank=MIN_SAMPLES_FOR_FULL_3D_COVARIANCE,
        sample_times_s=dispersion.sample_times_s,
        valid_trial_count=counts,
        centers_m=dispersion.mean_positions_m[:, point_index],
        principal_axes=dispersion.principal_axes[:, point_index],
        semi_axis_lengths_m=semi_axes,
        volume_m3=volume,
        adequacy=adequacy,
    )


def build_dispersion_metric_series(
    dispersion: PositionDispersion,
    point_id: str,
    metric: str,
    confidence_level: float = 0.95,
) -> DispersionMetricSeries:
    """Return a selectable metric with explicit units and adequacy."""
    require(
        metric in (RMS_RADIUS, LARGEST_PRINCIPAL_SIGMA, ELLIPSOID_VOLUME),
        "unknown metric",
        metric,
    )
    ellipsoids = build_confidence_ellipsoids(dispersion, point_id, confidence_level)
    point_index = dispersion.point_index(point_id)
    if metric == RMS_RADIUS:
        values = dispersion.rms_radius_m[:, point_index]
        unit = "m"
    elif metric == LARGEST_PRINCIPAL_SIGMA:
        with np.errstate(invalid="ignore"):
            values = np.sqrt(
                np.maximum(dispersion.eigenvalues_m2[:, point_index, 0], 0.0)
            )
        unit = "m"
    else:
        require(metric == ELLIPSOID_VOLUME, "unknown metric", metric)
        values = ellipsoids.volume_m3
        unit = "m^3"
    return DispersionMetricSeries(
        point_id=point_id,
        metric=metric,
        unit=unit,
        confidence_level=(
            ellipsoids.confidence_level if metric == ELLIPSOID_VOLUME else None
        ),
        sample_times_s=dispersion.sample_times_s,
        values=values,
        adequacy=ellipsoids.adequacy,
    )


def _true_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return inclusive bounds of contiguous true regions."""
    padded = np.pad(mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    return tuple(
        zip(
            np.flatnonzero(transitions == 1).tolist(),
            (np.flatnonzero(transitions == -1) - 1).tolist(),
            strict=True,
        )
    )


def _eligible(metric: str, series: DispersionMetricSeries) -> np.ndarray:
    """Return samples whose metric is statistically usable."""
    adequacy = np.asarray(series.adequacy)
    if metric == ELLIPSOID_VOLUME:
        return np.asarray(adequacy == ESTIMABLE, dtype=bool)
    return np.asarray(
        (adequacy == ESTIMABLE) | (adequacy == RANK_DEFICIENT), dtype=bool
    )


def find_ranked_low_variability_intervals(
    dispersion: PositionDispersion,
    criteria: LowVariabilityMetricCriteria,
) -> tuple[RankedLowVariabilityInterval, ...]:
    """Find and dense-rank quiet intervals by mean/threshold score.

    Lower scores rank first. Exactly equal IEEE-754 scores share a dense rank;
    point ID and start/end indices provide deterministic presentation order.
    """
    point_ids = criteria.point_ids or dispersion.point_ids
    for point_id in point_ids:
        require(point_id in dispersion.point_ids, "unknown point_id", point_id)
    candidates: list[RankedLowVariabilityInterval] = []
    for point_id in point_ids:
        series = build_dispersion_metric_series(
            dispersion, point_id, criteria.metric, criteria.confidence_level
        )
        qualifying = (
            _eligible(criteria.metric, series)
            & np.isfinite(series.values)
            & (series.values <= criteria.max_value)
        )
        for start, end in _true_runs(qualifying):
            n_samples = end - start + 1
            duration = float(series.sample_times_s[end] - series.sample_times_s[start])
            if n_samples < criteria.min_samples or duration < criteria.min_duration_s:
                continue
            selected = series.values[start : end + 1]
            mean_value = float(np.mean(selected))
            candidates.append(
                RankedLowVariabilityInterval(
                    point_id=point_id,
                    metric=series.metric,
                    unit=series.unit,
                    confidence_level=series.confidence_level,
                    start_index=start,
                    end_index=end,
                    start_time_s=float(series.sample_times_s[start]),
                    end_time_s=float(series.sample_times_s[end]),
                    n_samples=n_samples,
                    mean_value=mean_value,
                    max_value=float(np.max(selected)),
                    score=mean_value / float(criteria.max_value),
                    rank=1,
                )
            )
    ordered = sorted(
        candidates,
        key=lambda item: (item.score, item.point_id, item.start_index, item.end_index),
    )
    ranked: list[RankedLowVariabilityInterval] = []
    rank = 0
    previous_score: float | None = None
    for item in ordered:
        if previous_score is None or item.score != previous_score:
            rank += 1
            previous_score = item.score
        ranked.append(replace(item, rank=rank))
    return tuple(ranked)


__all__ = [
    "build_confidence_ellipsoids",
    "build_dispersion_metric_series",
    "find_ranked_low_variability_intervals",
]
