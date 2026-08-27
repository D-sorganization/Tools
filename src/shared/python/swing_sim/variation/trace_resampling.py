"""Fail-closed common-time-grid alignment for ensemble position traces.

The authority uses coordinate-wise linear interpolation only between adjacent
valid source samples. It never extrapolates and never bridges an invalid
sample. Impact indices remain display markers rather than exact event times;
the returned error records their displacement from the retained source marker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.python.contracts import require

from .ensemble_types import (
    EnsemblePositionTraces,
    immutable_array,
    validated_sample_times,
)

TRACE_RESAMPLING_POLICY_ID = "swing-trace-time-linear-contiguous/v1"
_COORDINATE_KIND = "time"
_COORDINATE_UNIT = "s"
_POSITION_METHOD = "piecewise_linear_adjacent_valid_samples"
_OUTSIDE_DOMAIN = "reject"
_INVALID_GAP = "preserve_unavailable"
_IMPACT_MARKER_METHOD = "nearest_valid_target_lower_tie"


@dataclass(frozen=True)
class TraceResamplingResult:
    """Aligned traces plus inspectable impact-marker approximation error."""

    traces: EnsemblePositionTraces
    impact_alignment_error_s: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        require(
            isinstance(self.traces, EnsemblePositionTraces),
            "traces must be EnsemblePositionTraces",
        )
        errors = np.asarray(self.impact_alignment_error_s, dtype=float)
        require(
            errors.shape == (self.traces.n_trials,),
            "impact_alignment_error_s must contain one value per trial",
        )
        impacts = self.traces.impact_occurred
        require(
            bool(np.all(np.isfinite(errors[impacts]))),
            "impact alignment errors must be finite for retained markers",
        )
        require(
            bool(np.all(errors[impacts] >= 0.0)),
            "impact alignment errors must be non-negative",
        )
        require(
            bool(np.all(np.isnan(errors[~impacts]))),
            "no-impact rows must have unavailable alignment error",
        )
        object.__setattr__(
            self, "impact_alignment_error_s", immutable_array(errors, float)
        )

    @property
    def policy_id(self) -> str:
        """Return the stable interpolation-policy identifier."""
        return TRACE_RESAMPLING_POLICY_ID

    @property
    def coordinate_kind(self) -> str:
        """Return the aligned coordinate kind."""
        return _COORDINATE_KIND

    @property
    def coordinate_unit(self) -> str:
        """Return the aligned coordinate unit."""
        return _COORDINATE_UNIT

    @property
    def position_method(self) -> str:
        """Return the position interpolation rule."""
        return _POSITION_METHOD

    @property
    def outside_domain(self) -> str:
        """Return the out-of-source-domain behavior."""
        return _OUTSIDE_DOMAIN

    @property
    def invalid_gap(self) -> str:
        """Return the missing-interval behavior."""
        return _INVALID_GAP

    @property
    def impact_marker_method(self) -> str:
        """Return the approximate impact-marker mapping rule."""
        return _IMPACT_MARKER_METHOD


def resample_position_traces(
    source: EnsemblePositionTraces, target_times_s: np.ndarray
) -> TraceResamplingResult:
    """Align validated ensemble traces onto one target time grid.

    Preconditions:
        ``source`` satisfies :class:`EnsemblePositionTraces`; target times are
        finite, one-dimensional, strictly increasing, and lie within the
        source grid's closed domain.

    Postconditions:
        Frame, point order, trial order, and variation identity are preserved.
        Samples are available only on exact valid source coordinates or inside
        an interval whose adjacent endpoints are both valid. Impact display
        markers select the nearest valid target sample, with the lower index
        winning an exact tie; their absolute timing error is returned.
    """
    require(
        isinstance(source, EnsemblePositionTraces),
        "source must be EnsemblePositionTraces",
    )
    target = validated_sample_times(np.asarray(target_times_s, dtype=float))
    source_times = source.sample_times_s
    require(
        target[0] >= source_times[0] and target[-1] <= source_times[-1],
        "target times lie outside the source domain",
    )
    positions, valid = _resample_positions(source, target)
    impacts, errors = _map_impact_markers(source, target, valid)
    traces = EnsemblePositionTraces(
        variation=source.variation,
        sample_times_s=target,
        coordinate_frame=source.coordinate_frame,
        point_ids=source.point_ids,
        positions_m=positions,
        sample_valid=valid,
        impact_sample_indices=impacts,
    )
    return TraceResamplingResult(traces, errors)


def _resample_positions(
    source: EnsemblePositionTraces, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and validity without extrapolating or bridging gaps."""
    source_times = source.sample_times_s
    insertions = np.searchsorted(source_times, target, side="left")
    right = np.clip(insertions, 0, source_times.size - 1)
    exact = source_times[right] == target
    left = np.clip(insertions - 1, 0, source_times.size - 1)
    output_shape = (
        source.n_trials,
        target.size,
        len(source.point_ids),
        source.positions_m.shape[-1],
    )
    positions = np.full(output_shape, np.nan)
    valid = np.zeros(output_shape[:2], dtype=bool)
    _copy_exact_samples(source, positions, valid, right, exact)
    _interpolate_adjacent_samples(source, target, positions, valid, left, right, exact)
    return positions, valid


def _copy_exact_samples(
    source: EnsemblePositionTraces,
    positions: np.ndarray,
    valid: np.ndarray,
    right: np.ndarray,
    exact: np.ndarray,
) -> None:
    """Copy exact source coordinates while preserving invalid samples."""
    targets = np.flatnonzero(exact)
    if targets.size == 0:
        return
    source_indices = right[targets]
    available = source.sample_valid[:, source_indices]
    valid[:, targets] = available
    values = source.positions_m[:, source_indices]
    positions[:, targets] = np.where(
        available[:, :, np.newaxis, np.newaxis], values, np.nan
    )


def _interpolate_adjacent_samples(
    source: EnsemblePositionTraces,
    target: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    exact: np.ndarray,
) -> None:
    """Interpolate only when both immediate source endpoints are valid."""
    targets = np.flatnonzero(~exact)
    if targets.size == 0:
        return
    left_indices = left[targets]
    right_indices = right[targets]
    available = (
        source.sample_valid[:, left_indices] & source.sample_valid[:, right_indices]
    )
    denominator = (
        source.sample_times_s[right_indices] - source.sample_times_s[left_indices]
    )
    weights = (target[targets] - source.sample_times_s[left_indices]) / denominator
    left_values = source.positions_m[:, left_indices]
    right_values = source.positions_m[:, right_indices]
    values = left_values + weights[np.newaxis, :, np.newaxis, np.newaxis] * (
        right_values - left_values
    )
    valid[:, targets] = available
    positions[:, targets] = np.where(
        available[:, :, np.newaxis, np.newaxis], values, np.nan
    )


def _map_impact_markers(
    source: EnsemblePositionTraces, target: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map retained marker times to nearest valid targets without fabrication."""
    impacts = np.full(source.n_trials, -1, dtype=int)
    errors = np.full(source.n_trials, np.nan)
    for trial_index in np.flatnonzero(source.impact_occurred):
        valid_targets = np.flatnonzero(valid[trial_index])
        require(
            valid_targets.size > 0,
            "impact marker has no valid target sample",
            int(trial_index),
        )
        source_index = int(source.impact_sample_indices[trial_index])
        source_time = source.sample_times_s[source_index]
        distances = np.abs(target[valid_targets] - source_time)
        target_index = int(valid_targets[int(np.argmin(distances))])
        impacts[trial_index] = target_index
        errors[trial_index] = float(abs(target[target_index] - source_time))
    return impacts, errors


__all__ = [
    "TRACE_RESAMPLING_POLICY_ID",
    "TraceResamplingResult",
    "resample_position_traces",
]
