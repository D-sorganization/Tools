"""Contracts and immutable value types for ensemble swing geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shared.python.contracts import require

from .engine import VariationDataset

CARTESIAN_DIMENSIONS = 3
MIN_TRIALS_FOR_COVARIANCE = 2
NO_IMPACT = -1


def immutable_array(value: np.ndarray, dtype: Any) -> np.ndarray:
    """Return an owned, read-only NumPy array with the requested dtype."""
    result: np.ndarray = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def require_coordinate_frame_id(coordinate_frame: str) -> None:
    """Require an unambiguous stable coordinate-frame identifier."""
    require(
        isinstance(coordinate_frame, str)
        and bool(coordinate_frame)
        and coordinate_frame == coordinate_frame.strip(),
        "coordinate_frame must be a non-empty, trimmed stable ID",
        coordinate_frame,
    )


def require_point_ids(point_ids: tuple[str, ...]) -> None:
    """Require a non-empty ordered set of stable modeled-point IDs."""
    require(len(point_ids) > 0, "point_ids must be non-empty", point_ids)
    valid_ids = all(
        isinstance(point_id, str) and bool(point_id) and point_id == point_id.strip()
        for point_id in point_ids
    )
    require(valid_ids, "point_ids must contain non-empty, trimmed strings", point_ids)
    require(
        len(set(point_ids)) == len(point_ids), "point_ids must be unique", point_ids
    )


def validated_sample_times(sample_times_s: np.ndarray) -> np.ndarray:
    """Validate and return a finite, strictly increasing 1-D sample grid."""
    times: np.ndarray = np.asarray(sample_times_s, dtype=float)
    require(
        times.ndim == 1 and times.size > 0,
        "sample times must be 1-D and non-empty",
        times.shape,
    )
    require(np.all(np.isfinite(times)), "sample times must be finite", times)
    require(
        np.all(np.diff(times) > 0.0),
        "sample times must be strictly increasing",
        times,
    )
    return times


@dataclass(frozen=True)
class _TraceArrays:
    """Normalized constructor arrays used by trace contract checks."""

    times: np.ndarray
    point_ids: tuple[str, ...]
    positions: np.ndarray
    valid: np.ndarray
    impacts: np.ndarray


def _trace_arrays(trace: EnsemblePositionTraces) -> _TraceArrays:
    """Normalize an ensemble's array-like constructor values."""
    return _TraceArrays(
        times=np.asarray(trace.sample_times_s, dtype=float),
        point_ids=tuple(trace.point_ids),
        positions=np.asarray(trace.positions_m, dtype=float),
        valid=np.asarray(trace.sample_valid, dtype=bool),
        impacts=np.asarray(trace.impact_sample_indices, dtype=int),
    )


def _validate_trace_grid(arrays: _TraceArrays) -> None:
    """Validate the shared sample coordinate and stable point identifiers."""
    validated_sample_times(arrays.times)
    require_point_ids(arrays.point_ids)


def _validate_trace_shapes(arrays: _TraceArrays, n_trials: int) -> None:
    """Validate all tensor axes against the variation trial count."""
    require(
        arrays.positions.ndim == 4
        and arrays.positions.shape[-1] == CARTESIAN_DIMENSIONS,
        "positions_m must have three Cartesian coordinates",
        arrays.positions.shape,
    )
    expected = (
        n_trials,
        arrays.times.size,
        len(arrays.point_ids),
        CARTESIAN_DIMENSIONS,
    )
    require(
        arrays.positions.shape == expected,
        "positions_m must be (n_trials, n_samples, n_points, 3)",
        arrays.positions.shape,
    )
    require(
        arrays.valid.shape == expected[:2],
        "sample_valid must be (n_trials, n_samples)",
        arrays.valid.shape,
    )
    require(
        arrays.impacts.shape == (n_trials,),
        "impact_sample_indices must be (n_trials,)",
        arrays.impacts.shape,
    )


def _validate_trace_values(arrays: _TraceArrays) -> None:
    """Validate missing-data and impact-marker invariants."""
    require(
        np.all(np.isfinite(arrays.positions[arrays.valid])),
        "valid samples must contain finite positions",
        None,
    )
    require(
        np.all(np.isnan(arrays.positions[~arrays.valid])),
        "invalid samples must contain only NaN positions",
        None,
    )
    impact_present = arrays.impacts >= 0
    legal = (arrays.impacts == NO_IMPACT) | (
        impact_present & (arrays.impacts < arrays.times.size)
    )
    require(
        np.all(legal),
        "impact sample must be -1 or a valid sample index",
        arrays.impacts,
    )
    rows = np.flatnonzero(impact_present)
    require(
        np.all(arrays.valid[rows, arrays.impacts[rows]]),
        "impact sample must refer to a valid trace sample",
        arrays.impacts,
    )


@dataclass(frozen=True)
class EnsemblePositionTraces:
    """Common-grid position traces aligned to a variation dataset.

    ``coordinate_frame`` and ``point_ids`` are stable identifiers.
    ``positions_m`` has shape ``(trial, sample, point, xyz)`` and aligns to
    the variation rows. Invalid samples contain only ``NaN``.
    ``impact_sample_indices == -1`` retains a no-impact trial as a full row.
    """

    variation: VariationDataset
    sample_times_s: np.ndarray = field(repr=False)
    coordinate_frame: str
    point_ids: tuple[str, ...]
    positions_m: np.ndarray = field(repr=False)
    sample_valid: np.ndarray = field(repr=False)
    impact_sample_indices: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        require(
            isinstance(self.variation, VariationDataset),
            "variation must be a VariationDataset",
            type(self.variation).__name__,
        )
        require_coordinate_frame_id(self.coordinate_frame)
        arrays = _trace_arrays(self)
        _validate_trace_grid(arrays)
        _validate_trace_shapes(arrays, self.variation.plan.n_runs)
        _validate_trace_values(arrays)
        object.__setattr__(self, "sample_times_s", immutable_array(arrays.times, float))
        object.__setattr__(self, "point_ids", arrays.point_ids)
        object.__setattr__(
            self, "positions_m", immutable_array(arrays.positions, float)
        )
        object.__setattr__(self, "sample_valid", immutable_array(arrays.valid, bool))
        object.__setattr__(
            self, "impact_sample_indices", immutable_array(arrays.impacts, int)
        )

    @property
    def n_trials(self) -> int:
        """Return the number of trial rows, including no-impact trials."""
        return int(self.variation.plan.n_runs)

    @property
    def impact_occurred(self) -> np.ndarray:
        """Boolean impact status for every trial without dropping misses."""
        occurred: np.ndarray = self.impact_sample_indices >= 0
        return occurred

    @property
    def n_no_impact(self) -> int:
        """Return the number of trials whose club never reached impact."""
        return int(np.count_nonzero(~self.impact_occurred))

    def point_index(self, point_id: str) -> int:
        """Return the stable column index for ``point_id``."""
        require(point_id in self.point_ids, "unknown point_id", point_id)
        return self.point_ids.index(point_id)


def _dispersion_shape(dispersion: PositionDispersion) -> tuple[int, int]:
    """Validate dispersion identifiers and return its sample/point shape."""
    times = validated_sample_times(dispersion.sample_times_s)
    require_coordinate_frame_id(dispersion.coordinate_frame)
    require_point_ids(tuple(dispersion.point_ids))
    return int(times.size), len(dispersion.point_ids)


@dataclass(frozen=True)
class PositionDispersion:
    """Per-sample, per-point position dispersion in one explicit frame.

    Array axes begin with ``(sample, point)``. Covariance uses the unbiased
    sample convention; ``rms_radius_m`` is the population RMS distance from
    the centroid. Eigenvalues are descending and principal-axis columns use
    the corresponding sign-canonicalized vectors. A repeated-eigenvalue
    eigenspace has no physically unique orientation.
    """

    sample_times_s: np.ndarray = field(repr=False)
    coordinate_frame: str
    point_ids: tuple[str, ...]
    count: np.ndarray = field(repr=False)
    mean_positions_m: np.ndarray = field(repr=False)
    covariance_m2: np.ndarray = field(repr=False)
    eigenvalues_m2: np.ndarray = field(repr=False)
    principal_axes: np.ndarray = field(repr=False)
    rms_radius_m: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        samples, points = _dispersion_shape(self)
        shape = (samples, points)
        arrays = {
            "count": (self.count, shape),
            "mean_positions_m": (self.mean_positions_m, shape + (3,)),
            "covariance_m2": (self.covariance_m2, shape + (3, 3)),
            "eigenvalues_m2": (self.eigenvalues_m2, shape + (3,)),
            "principal_axes": (self.principal_axes, shape + (3, 3)),
            "rms_radius_m": (self.rms_radius_m, shape),
        }
        for name, (value, expected) in arrays.items():
            require(
                np.asarray(value).shape == expected,
                f"{name} has invalid shape",
                np.asarray(value).shape,
            )
        require(
            np.all(np.asarray(self.count) >= 0),
            "count must be non-negative",
            self.count,
        )
        object.__setattr__(
            self, "sample_times_s", immutable_array(self.sample_times_s, float)
        )
        object.__setattr__(self, "point_ids", tuple(self.point_ids))
        object.__setattr__(self, "count", immutable_array(self.count, int))
        for name in arrays.keys() - {"count"}:
            object.__setattr__(self, name, immutable_array(getattr(self, name), float))

    def point_index(self, point_id: str) -> int:
        """Return the stable result-column index for ``point_id``."""
        require(point_id in self.point_ids, "unknown point_id", point_id)
        return self.point_ids.index(point_id)


@dataclass(frozen=True)
class LowVariabilityCriteria:
    """Explicit threshold and continuity rules for quiet-zone detection."""

    max_rms_radius_m: float
    min_duration_s: float = 0.0
    min_samples: int = 1
    point_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require(
            math.isfinite(self.max_rms_radius_m) and self.max_rms_radius_m > 0.0,
            "max_rms_radius_m must be finite and > 0",
            self.max_rms_radius_m,
        )
        require(
            math.isfinite(self.min_duration_s) and self.min_duration_s >= 0.0,
            "min_duration_s must be finite and >= 0",
            self.min_duration_s,
        )
        require(
            isinstance(self.min_samples, int) and self.min_samples >= 1,
            "min_samples must be an integer >= 1",
            self.min_samples,
        )
        points = tuple(self.point_ids)
        require(len(set(points)) == len(points), "point_ids must be unique", points)
        object.__setattr__(self, "point_ids", points)


@dataclass(frozen=True)
class LowVariabilityInterval:
    """Contiguous sample interval satisfying low-variability criteria."""

    point_id: str
    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    n_samples: int
    mean_rms_radius_m: float
    max_rms_radius_m: float


__all__ = [
    "CARTESIAN_DIMENSIONS",
    "MIN_TRIALS_FOR_COVARIANCE",
    "EnsemblePositionTraces",
    "LowVariabilityCriteria",
    "LowVariabilityInterval",
    "PositionDispersion",
    "immutable_array",
    "require_coordinate_frame_id",
    "require_point_ids",
    "validated_sample_times",
]
