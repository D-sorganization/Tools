"""Immutable contracts for selectable geometric dispersion metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from shared.python.contracts import require

from .ensemble_types import CARTESIAN_DIMENSIONS, immutable_array

RMS_RADIUS = "rms-radius"
LARGEST_PRINCIPAL_SIGMA = "largest-principal-sigma"
ELLIPSOID_VOLUME = "confidence-ellipsoid-volume"
LOW_VARIABILITY_METRICS = (
    RMS_RADIUS,
    LARGEST_PRINCIPAL_SIGMA,
    ELLIPSOID_VOLUME,
)

ESTIMABLE = "estimable"
RANK_DEFICIENT = "rank-deficient"
INSUFFICIENT_SAMPLES = "insufficient-samples"
INVALID_COVARIANCE = "invalid-covariance"
MIN_SAMPLES_FOR_FULL_3D_COVARIANCE = CARTESIAN_DIMENSIONS + 1
GAUSSIAN_POSITION_CONTENT_REGION = "gaussian-position-content-region"
CHI_SQUARE_DEGREES_OF_FREEDOM = CARTESIAN_DIMENSIONS
_ADEQUACY_STATES = (
    ESTIMABLE,
    RANK_DEFICIENT,
    INSUFFICIENT_SAMPLES,
    INVALID_COVARIANCE,
)


def validated_confidence_level(confidence_level: float) -> float:
    """Return a finite open-interval confidence level, rejecting booleans."""
    require(
        not isinstance(confidence_level, (bool, np.bool_))
        and isinstance(confidence_level, Real),
        "confidence_level must be a real number in (0, 1)",
        confidence_level,
    )
    value = float(confidence_level)
    require(
        math.isfinite(value) and 0.0 < value < 1.0,
        "confidence_level must be finite and in (0, 1)",
        confidence_level,
    )
    return value


@dataclass(frozen=True)
class ConfidenceEllipsoidSeries:
    """Plot-ready Gaussian position-content ellipsoids for one point.

    This is a plug-in sample-covariance description of position dispersion,
    not a confidence region for the unknown population mean.
    """

    point_id: str
    coordinate_frame: str
    interpretation: str
    confidence_level: float
    degrees_of_freedom: int
    chi_square_quantile: float
    radius_scale: float
    minimum_samples_for_full_rank: int
    sample_times_s: np.ndarray = field(repr=False)
    valid_trial_count: np.ndarray = field(repr=False)
    centers_m: np.ndarray = field(repr=False)
    principal_axes: np.ndarray = field(repr=False)
    semi_axis_lengths_m: np.ndarray = field(repr=False)
    volume_m3: np.ndarray = field(repr=False)
    adequacy: tuple[str, ...]

    def __post_init__(self) -> None:
        samples = np.asarray(self.sample_times_s).size
        require(bool(self.point_id), "point_id must be non-empty")
        require(bool(self.coordinate_frame), "coordinate_frame must be non-empty")
        require(
            self.interpretation == GAUSSIAN_POSITION_CONTENT_REGION,
            "invalid ellipsoid interpretation",
        )
        validated_confidence_level(self.confidence_level)
        require(
            self.degrees_of_freedom == CHI_SQUARE_DEGREES_OF_FREEDOM,
            "invalid degrees_of_freedom",
        )
        require(
            math.isfinite(self.chi_square_quantile) and self.chi_square_quantile > 0.0,
            "chi_square_quantile must be finite and positive",
        )
        require(
            math.isfinite(self.radius_scale)
            and self.radius_scale > 0.0
            and math.isclose(
                self.radius_scale**2,
                self.chi_square_quantile,
                rel_tol=32.0 * np.finfo(float).eps,
            ),
            "radius_scale must equal sqrt(chi_square_quantile)",
        )
        require(
            self.minimum_samples_for_full_rank == MIN_SAMPLES_FOR_FULL_3D_COVARIANCE,
            "invalid minimum_samples_for_full_rank",
        )
        expected = {
            "valid_trial_count": (samples,),
            "centers_m": (samples, CARTESIAN_DIMENSIONS),
            "principal_axes": (
                samples,
                CARTESIAN_DIMENSIONS,
                CARTESIAN_DIMENSIONS,
            ),
            "semi_axis_lengths_m": (samples, CARTESIAN_DIMENSIONS),
            "volume_m3": (samples,),
        }
        for name, shape in expected.items():
            require(np.asarray(getattr(self, name)).shape == shape, f"invalid {name}")
        require(len(self.adequacy) == samples, "invalid adequacy")
        require(
            all(item in _ADEQUACY_STATES for item in self.adequacy),
            "unknown adequacy state",
        )
        require(
            np.all(np.asarray(self.valid_trial_count) >= 0),
            "valid_trial_count must be non-negative",
        )
        estimable = np.asarray(self.adequacy) == ESTIMABLE
        require(
            np.all(np.isfinite(np.asarray(self.volume_m3)[estimable]))
            and np.all(np.asarray(self.volume_m3)[estimable] > 0.0),
            "estimable ellipsoids require positive finite volume",
        )
        require(
            np.all(np.isnan(np.asarray(self.volume_m3)[~estimable])),
            "non-estimable ellipsoid volume must be unavailable",
        )
        object.__setattr__(
            self, "sample_times_s", immutable_array(self.sample_times_s, float)
        )
        object.__setattr__(
            self, "valid_trial_count", immutable_array(self.valid_trial_count, int)
        )
        for name in (
            "centers_m",
            "principal_axes",
            "semi_axis_lengths_m",
            "volume_m3",
        ):
            object.__setattr__(self, name, immutable_array(getattr(self, name), float))


@dataclass(frozen=True)
class DispersionMetricSeries:
    """One selected quiet-zone metric with units and adequacy per sample."""

    point_id: str
    metric: str
    unit: str
    confidence_level: float | None
    sample_times_s: np.ndarray = field(repr=False)
    values: np.ndarray = field(repr=False)
    adequacy: tuple[str, ...]

    def __post_init__(self) -> None:
        samples = np.asarray(self.sample_times_s).size
        require(self.metric in LOW_VARIABILITY_METRICS, "unknown metric", self.metric)
        expected_unit = "m^3" if self.metric == ELLIPSOID_VOLUME else "m"
        require(self.unit == expected_unit, "invalid metric unit", self.unit)
        if self.metric == ELLIPSOID_VOLUME:
            confidence = self.confidence_level
            require(confidence is not None, "volume requires confidence")
            if confidence is not None:
                validated_confidence_level(confidence)
        else:
            require(self.confidence_level is None, "metric does not use confidence")
        require(np.asarray(self.values).shape == (samples,), "invalid metric values")
        require(len(self.adequacy) == samples, "invalid adequacy")
        require(
            all(item in _ADEQUACY_STATES for item in self.adequacy),
            "unknown adequacy state",
        )
        object.__setattr__(
            self, "sample_times_s", immutable_array(self.sample_times_s, float)
        )
        object.__setattr__(self, "values", immutable_array(self.values, float))


@dataclass(frozen=True)
class LowVariabilityMetricCriteria:
    """Declared metric, threshold, confidence, and continuity requirements."""

    metric: str
    max_value: float
    confidence_level: float = 0.95
    min_duration_s: float = 0.0
    min_samples: int = 1
    point_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require(self.metric in LOW_VARIABILITY_METRICS, "unknown metric", self.metric)
        require(
            not isinstance(self.max_value, (bool, np.bool_))
            and math.isfinite(float(self.max_value))
            and float(self.max_value) > 0.0,
            "max_value must be finite and > 0",
            self.max_value,
        )
        validated_confidence_level(self.confidence_level)
        require(
            not isinstance(self.min_duration_s, (bool, np.bool_))
            and math.isfinite(float(self.min_duration_s))
            and float(self.min_duration_s) >= 0.0,
            "min_duration_s must be finite and >= 0",
            self.min_duration_s,
        )
        require(
            not isinstance(self.min_samples, (bool, np.bool_))
            and isinstance(self.min_samples, int)
            and self.min_samples >= 1,
            "min_samples must be an integer >= 1",
            self.min_samples,
        )
        points = tuple(self.point_ids)
        require(len(set(points)) == len(points), "point_ids must be unique", points)
        require(
            all(bool(item) and item == item.strip() for item in points),
            "point_ids must be non-empty and trimmed",
            points,
        )
        object.__setattr__(self, "point_ids", points)


@dataclass(frozen=True)
class RankedLowVariabilityInterval:
    """One qualifying interval ranked by its dimensionless mean score."""

    point_id: str
    metric: str
    unit: str
    confidence_level: float | None
    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    n_samples: int
    mean_value: float
    max_value: float
    score: float
    rank: int

    def __post_init__(self) -> None:
        """Validate interval bounds, units, finite metrics, and dense rank."""
        require(self.metric in LOW_VARIABILITY_METRICS, "unknown metric", self.metric)
        expected_unit = "m^3" if self.metric == ELLIPSOID_VOLUME else "m"
        require(self.unit == expected_unit, "invalid metric unit", self.unit)
        require(
            0 <= self.start_index <= self.end_index,
            "invalid interval indices",
        )
        require(
            self.n_samples == self.end_index - self.start_index + 1,
            "n_samples must match interval indices",
        )
        require(
            self.start_time_s <= self.end_time_s,
            "invalid interval times",
        )
        require(
            all(
                math.isfinite(value) and value >= 0.0
                for value in (self.mean_value, self.max_value, self.score)
            ),
            "interval metrics must be finite and non-negative",
        )
        require(self.mean_value <= self.max_value, "mean_value cannot exceed max_value")
        require(
            isinstance(self.rank, int)
            and not isinstance(self.rank, bool)
            and self.rank >= 1,
            "rank must be a positive integer",
        )


__all__ = [
    "ELLIPSOID_VOLUME",
    "ESTIMABLE",
    "GAUSSIAN_POSITION_CONTENT_REGION",
    "INSUFFICIENT_SAMPLES",
    "INVALID_COVARIANCE",
    "LARGEST_PRINCIPAL_SIGMA",
    "LOW_VARIABILITY_METRICS",
    "MIN_SAMPLES_FOR_FULL_3D_COVARIANCE",
    "RANK_DEFICIENT",
    "RMS_RADIUS",
    "ConfidenceEllipsoidSeries",
    "DispersionMetricSeries",
    "LowVariabilityMetricCriteria",
    "RankedLowVariabilityInterval",
]
