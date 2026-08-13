"""Plot-ready geometric variability and quiet-zone data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from shared.python.contracts import require
from shared.python.swing_sim.variation import (
    ELLIPSOID_VOLUME,
    ESTIMABLE,
    GAUSSIAN_POSITION_CONTENT_REGION,
    INSUFFICIENT_SAMPLES,
    INVALID_COVARIANCE,
    RANK_DEFICIENT,
    LowVariabilityCriteria,
    LowVariabilityInterval,
    LowVariabilityMetricCriteria,
    PositionDispersion,
    RankedLowVariabilityInterval,
    build_dispersion_metric_series,
    compute_position_dispersion_view,
    find_low_variability_intervals,
    find_ranked_low_variability_intervals,
)
from shared.python.swing_sim.variation.ensemble_types import (
    EnsemblePositionTraces,
    immutable_array,
)

COMMON_TIME_ALIGNMENT = "common_simulation_time_s"
_DISPLAY_SCALE = {"m": 1_000.0, "m^3": 1_000_000_000.0}
_DISPLAY_UNIT = {"m": "mm", "m^3": "mm³"}
_ADEQUACY_STATES = (
    ESTIMABLE,
    RANK_DEFICIENT,
    INSUFFICIENT_SAMPLES,
    INVALID_COVARIANCE,
)


@dataclass(frozen=True)
class GeometricVariabilityData:
    """One point's covariance, RMS envelope, and declared quiet zones."""

    point_id: str
    coordinate_frame: str
    alignment_basis: str
    position_unit: str
    sample_times_s: np.ndarray = field(repr=False)
    valid_trial_count: np.ndarray = field(repr=False)
    mean_positions_m: np.ndarray = field(repr=False)
    rms_radius_m: np.ndarray = field(repr=False)
    principal_sigma_m: np.ndarray = field(repr=False)
    principal_axes: np.ndarray = field(repr=False)
    quiet_mask: np.ndarray = field(repr=False)
    quiet_intervals: tuple[LowVariabilityInterval, ...]
    criteria: LowVariabilityCriteria

    def __post_init__(self) -> None:
        samples = np.asarray(self.sample_times_s).size
        require(
            self.alignment_basis == COMMON_TIME_ALIGNMENT, "invalid alignment basis"
        )
        require(self.position_unit == "m", "position_unit must be m")
        expected_shapes = {
            "valid_trial_count": (samples,),
            "mean_positions_m": (samples, 3),
            "rms_radius_m": (samples,),
            "principal_sigma_m": (samples, 3),
            "principal_axes": (samples, 3, 3),
            "quiet_mask": (samples,),
        }
        for name, shape in expected_shapes.items():
            require(np.asarray(getattr(self, name)).shape == shape, f"invalid {name}")
        object.__setattr__(
            self, "sample_times_s", immutable_array(self.sample_times_s, float)
        )
        object.__setattr__(
            self,
            "valid_trial_count",
            immutable_array(self.valid_trial_count, int),
        )
        for name in (
            "mean_positions_m",
            "rms_radius_m",
            "principal_sigma_m",
            "principal_axes",
        ):
            object.__setattr__(self, name, immutable_array(getattr(self, name), float))
        object.__setattr__(self, "quiet_mask", immutable_array(self.quiet_mask, bool))

    @property
    def n_quiet_samples(self) -> int:
        """Return the number of samples satisfying the declared criteria."""
        return int(np.count_nonzero(self.quiet_mask))


def build_geometric_variability(
    dispersion: PositionDispersion,
    point_id: str,
    criteria: LowVariabilityCriteria,
) -> GeometricVariabilityData:
    """Prepare one point without recomputing simulation or presentation physics."""
    point_index = dispersion.point_index(point_id)
    intervals = find_low_variability_intervals(dispersion, criteria)
    selected = tuple(item for item in intervals if item.point_id == point_id)
    quiet = np.zeros(dispersion.sample_times_s.size, dtype=bool)
    for interval in selected:
        quiet[interval.start_index : interval.end_index + 1] = True
    eigenvalues = dispersion.eigenvalues_m2[:, point_index]
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(np.maximum(eigenvalues, 0.0))
    return GeometricVariabilityData(
        point_id=point_id,
        coordinate_frame=dispersion.coordinate_frame,
        alignment_basis=COMMON_TIME_ALIGNMENT,
        position_unit="m",
        sample_times_s=dispersion.sample_times_s,
        valid_trial_count=dispersion.count[:, point_index],
        mean_positions_m=dispersion.mean_positions_m[:, point_index],
        rms_radius_m=dispersion.rms_radius_m[:, point_index],
        principal_sigma_m=sigma,
        principal_axes=dispersion.principal_axes[:, point_index],
        quiet_mask=quiet,
        quiet_intervals=selected,
        criteria=criteria,
    )


@dataclass(frozen=True)
class DispersionMetricVariabilityData:
    """Plot-ready selected dispersion authority and ranked quiet intervals."""

    point_id: str
    coordinate_frame: str
    alignment_basis: str
    metric: str
    authority_unit: str
    display_unit: str
    confidence_level: float | None
    interpretation: str
    sample_times_s: np.ndarray = field(repr=False)
    valid_trial_count: np.ndarray = field(repr=False)
    mean_positions_m: np.ndarray = field(repr=False)
    principal_sigma_m: np.ndarray = field(repr=False)
    principal_axes: np.ndarray = field(repr=False)
    metric_values: np.ndarray = field(repr=False)
    display_values: np.ndarray = field(repr=False)
    adequacy: tuple[str, ...]
    adequacy_counts: Mapping[str, int]
    quiet_mask: np.ndarray = field(repr=False)
    quiet_intervals: tuple[RankedLowVariabilityInterval, ...]
    criteria: LowVariabilityMetricCriteria

    def __post_init__(self) -> None:
        samples = np.asarray(self.sample_times_s).size
        require(self.alignment_basis == COMMON_TIME_ALIGNMENT, "invalid alignment")
        require(self.authority_unit in _DISPLAY_SCALE, "invalid authority unit")
        require(
            self.display_unit == _DISPLAY_UNIT[self.authority_unit],
            "invalid display unit",
        )
        expected = {
            "valid_trial_count": (samples,),
            "mean_positions_m": (samples, 3),
            "principal_sigma_m": (samples, 3),
            "principal_axes": (samples, 3, 3),
            "metric_values": (samples,),
            "display_values": (samples,),
            "quiet_mask": (samples,),
        }
        for name, shape in expected.items():
            require(np.asarray(getattr(self, name)).shape == shape, f"invalid {name}")
        require(len(self.adequacy) == samples, "invalid adequacy")
        require(
            sum(self.adequacy_counts.values()) == samples,
            "invalid adequacy counts",
        )
        for name, dtype in (
            ("sample_times_s", float),
            ("valid_trial_count", int),
            ("mean_positions_m", float),
            ("principal_sigma_m", float),
            ("principal_axes", float),
            ("metric_values", float),
            ("display_values", float),
            ("quiet_mask", bool),
        ):
            object.__setattr__(self, name, immutable_array(getattr(self, name), dtype))
        object.__setattr__(
            self, "adequacy_counts", MappingProxyType(dict(self.adequacy_counts))
        )

    @property
    def n_quiet_samples(self) -> int:
        """Return the number of samples in ranked qualifying intervals."""
        return int(np.count_nonzero(self.quiet_mask))

    @property
    def unavailable_count(self) -> int:
        """Return samples unavailable for the selected metric."""
        usable = (
            (ESTIMABLE,)
            if self.metric == ELLIPSOID_VOLUME
            else (
                ESTIMABLE,
                RANK_DEFICIENT,
            )
        )
        return sum(
            count
            for state, count in self.adequacy_counts.items()
            if state not in usable
        )


def build_dispersion_metric_variability(
    dispersion: PositionDispersion,
    point_id: str,
    criteria: LowVariabilityMetricCriteria,
) -> DispersionMetricVariabilityData:
    """Prepare one selected shared metric without recreating statistical rules."""
    series = build_dispersion_metric_series(
        dispersion,
        point_id,
        criteria.metric,
        criteria.confidence_level,
    )
    point_index = dispersion.point_index(point_id)
    point_criteria = replace(criteria, point_ids=(point_id,))
    intervals = tuple(
        item
        for item in find_ranked_low_variability_intervals(dispersion, point_criteria)
        if item.point_id == point_id
    )
    quiet = np.zeros(dispersion.sample_times_s.size, dtype=bool)
    for interval in intervals:
        quiet[interval.start_index : interval.end_index + 1] = True
    eigenvalues = dispersion.eigenvalues_m2[:, point_index]
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(np.maximum(eigenvalues, 0.0))
    scale = _DISPLAY_SCALE[series.unit]
    counts = {state: series.adequacy.count(state) for state in _ADEQUACY_STATES}
    return DispersionMetricVariabilityData(
        point_id=point_id,
        coordinate_frame=dispersion.coordinate_frame,
        alignment_basis=COMMON_TIME_ALIGNMENT,
        metric=series.metric,
        authority_unit=series.unit,
        display_unit=_DISPLAY_UNIT[series.unit],
        confidence_level=series.confidence_level,
        interpretation=(
            GAUSSIAN_POSITION_CONTENT_REGION
            if series.metric == ELLIPSOID_VOLUME
            else "sample-position-dispersion"
        ),
        sample_times_s=series.sample_times_s,
        valid_trial_count=dispersion.count[:, point_index],
        mean_positions_m=dispersion.mean_positions_m[:, point_index],
        principal_sigma_m=sigma,
        principal_axes=dispersion.principal_axes[:, point_index],
        metric_values=series.values,
        display_values=series.values * scale,
        adequacy=series.adequacy,
        adequacy_counts=counts,
        quiet_mask=quiet,
        quiet_intervals=intervals,
        criteria=point_criteria,
    )


def build_dispersion_metric_variability_view(
    dispersion: PositionDispersion,
    traces: EnsemblePositionTraces,
    point_id: str,
    criteria: LowVariabilityMetricCriteria,
    trial_indices: np.ndarray | None = None,
    sample_count: int | None = None,
) -> DispersionMetricVariabilityData:
    """Build selected metric data over an optional trial/time view."""
    selected = (
        dispersion
        if trial_indices is None and sample_count is None
        else compute_position_dispersion_view(
            traces,
            trial_indices=trial_indices,
            sample_count=sample_count,
        )
    )
    return build_dispersion_metric_variability(selected, point_id, criteria)


__all__ = [
    "COMMON_TIME_ALIGNMENT",
    "DispersionMetricVariabilityData",
    "GeometricVariabilityData",
    "build_dispersion_metric_variability",
    "build_dispersion_metric_variability_view",
    "build_geometric_variability",
]
