"""Pure-data geometric analysis for multi-trial swing ensembles.

No swing physics runs here. Any solver or importer can supply stable point
IDs and common-grid traces through :class:`EnsemblePositionTraces`.
"""

from __future__ import annotations

import numpy as np

from shared.python.contracts import require

from .ensemble_types import (
    CARTESIAN_DIMENSIONS,
    MIN_TRIALS_FOR_COVARIANCE,
    EnsemblePositionTraces,
    LowVariabilityCriteria,
    LowVariabilityInterval,
    PositionDispersion,
    immutable_array,
)

MAX_DISPERSION_ACCUMULATOR_BYTES = 256_000_000


def _principal_components(
    covariance: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return descending, sign-canonicalized covariance eigenpairs."""
    eigenvalues = np.full(counts.shape + (CARTESIAN_DIMENSIONS,), np.nan)
    principal_axes = np.full(
        counts.shape + (CARTESIAN_DIMENSIONS, CARTESIAN_DIMENSIONS), np.nan
    )
    covariance_available = counts >= MIN_TRIALS_FOR_COVARIANCE
    if np.any(covariance_available):
        values, vectors = np.linalg.eigh(covariance[covariance_available])
        largest_rows = np.argmax(np.abs(vectors), axis=1)
        selected = np.take_along_axis(
            vectors, largest_rows[:, np.newaxis, :], axis=1
        ).squeeze(axis=1)
        vectors = vectors * np.where(selected < 0.0, -1.0, 1.0)[:, np.newaxis, :]
        eigenvalues[covariance_available] = values[:, ::-1]
        principal_axes[covariance_available] = vectors[:, :, ::-1]
    return eigenvalues, principal_axes


def compute_position_dispersion(ensemble: EnsemblePositionTraces) -> PositionDispersion:
    """Compute per-sample/per-point covariance, eigenpairs, and RMS radius.

    ``variation.success`` is intentionally not a filter: a no-impact or
    invalid shot outcome may still have a valid swing trace. Only the explicit
    trace-level ``sample_valid`` mask controls geometric inclusion.
    """
    return compute_position_dispersion_view(ensemble)


class PositionDispersionAccumulator:
    """Bounded online position moments over arbitrary contiguous trial chunks."""

    def __init__(self, sample_count: int, point_count: int) -> None:
        require(
            type(sample_count) is int and sample_count >= 1,
            "sample_count must be a positive integer",
        )
        require(
            type(point_count) is int and point_count >= 1,
            "point_count must be a positive integer",
        )
        estimated_bytes = sample_count * 8 + sample_count * point_count * 12 * 8
        require(
            estimated_bytes <= MAX_DISPERSION_ACCUMULATOR_BYTES,
            "dispersion accumulator memory budget exceeded",
            estimated_bytes,
        )
        self._shape = (sample_count, point_count)
        self._count: np.ndarray = np.zeros(sample_count, dtype=np.int64)
        self._mean = np.zeros(self._shape + (CARTESIAN_DIMENSIONS,), dtype=float)
        self._centered_sum = np.zeros(
            self._shape + (CARTESIAN_DIMENSIONS, CARTESIAN_DIMENSIONS), dtype=float
        )

    def accept(self, positions_m: np.ndarray, sample_valid: np.ndarray) -> None:
        """Merge one nonempty trial chunk without retaining any trial rows."""
        positions = np.asarray(positions_m, dtype=float)
        valid = np.asarray(sample_valid)
        require(positions.ndim == 4, "positions_m must be four-dimensional")
        trials, samples, points, dimensions = positions.shape
        require(trials >= 1, "positions_m must contain at least one trial")
        require(
            (samples, points) == self._shape and dimensions == CARTESIAN_DIMENSIONS,
            "positions_m does not match accumulator shape",
        )
        require(valid.dtype == np.bool_, "sample_valid must be Boolean")
        require(valid.shape == (trials, samples), "sample_valid shape is invalid")
        require(
            bool(np.all(np.isfinite(positions[valid]))),
            "valid positions must be finite",
        )
        for trial_positions, trial_valid in zip(positions, valid, strict=True):
            self._accept_trial(trial_positions, trial_valid)

    def _accept_trial(self, positions: np.ndarray, valid: np.ndarray) -> None:
        safe = np.where(valid[:, np.newaxis, np.newaxis], positions, self._mean)
        delta = safe - self._mean
        new_count = self._count + valid
        scale: np.ndarray = np.zeros(self._count.shape, dtype=float)
        np.divide(valid, new_count, out=scale, where=new_count > 0)
        self._mean += delta * scale[:, np.newaxis, np.newaxis]
        delta_after = safe - self._mean
        self._centered_sum += np.einsum("spc,spd->spcd", delta, delta_after)
        self._count = new_count

    def freeze(
        self,
        sample_times_s: np.ndarray,
        coordinate_frame: str,
        point_ids: tuple[str, ...],
    ) -> PositionDispersion:
        """Return immutable covariance geometry for the accumulated prefix."""
        times = np.asarray(sample_times_s, dtype=float)
        require(times.shape == (self._shape[0],), "sample_times_s shape is invalid")
        require(len(point_ids) == self._shape[1], "point_ids shape is invalid")
        counts = np.broadcast_to(self._count[:, np.newaxis], self._shape).copy()
        mean = self._mean.copy()
        mean[counts == 0] = np.nan
        centered_sum = 0.5 * (self._centered_sum + self._centered_sum.swapaxes(-1, -2))
        divisor = counts[:, :, np.newaxis, np.newaxis] - 1
        covariance = np.full(centered_sum.shape, np.nan)
        np.divide(centered_sum, divisor, out=covariance, where=divisor > 0)
        radius_sum = np.trace(centered_sum, axis1=-2, axis2=-1)
        rms_radius = np.full(self._shape, np.nan)
        np.divide(radius_sum, counts, out=rms_radius, where=counts > 0)
        np.sqrt(np.maximum(rms_radius, 0.0), out=rms_radius)
        eigenvalues, principal_axes = _principal_components(covariance, counts)
        return PositionDispersion(
            immutable_array(times, float),
            coordinate_frame,
            tuple(point_ids),
            immutable_array(counts, int),
            immutable_array(mean, float),
            immutable_array(covariance, float),
            immutable_array(eigenvalues, float),
            immutable_array(principal_axes, float),
            immutable_array(rms_radius, float),
        )


def compute_position_dispersion_view(
    ensemble: EnsemblePositionTraces,
    trial_indices: np.ndarray | None = None,
    sample_count: int | None = None,
) -> PositionDispersion:
    """Compute dispersion for a validated trial subset and leading time window."""
    indices = (
        np.arange(ensemble.positions_m.shape[0], dtype=int)
        if trial_indices is None
        else np.asarray(trial_indices, dtype=int)
    )
    require(indices.ndim == 1, "trial_indices must be one-dimensional")
    require(indices.size >= 1, "trial_indices must select at least one trial")
    require(np.all(indices >= 0), "trial_indices cannot be negative")
    require(
        np.all(indices < ensemble.positions_m.shape[0]),
        "trial_indices exceed the ensemble",
    )
    require(np.unique(indices).size == indices.size, "trial_indices must be unique")
    count = ensemble.sample_times_s.size if sample_count is None else sample_count
    require(1 <= count <= ensemble.sample_times_s.size, "invalid sample_count")
    positions = ensemble.positions_m[indices, :count]
    sample_valid = ensemble.sample_valid[indices, :count]
    accumulator = PositionDispersionAccumulator(count, positions.shape[2])
    accumulator.accept(positions, sample_valid)
    return accumulator.freeze(
        ensemble.sample_times_s[:count], ensemble.coordinate_frame, ensemble.point_ids
    )


def _true_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return inclusive index bounds of contiguous true regions."""
    padded = np.pad(mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts: np.ndarray = np.flatnonzero(transitions == 1)
    ends: np.ndarray = np.flatnonzero(transitions == -1) - 1
    return tuple(zip(starts.tolist(), ends.tolist(), strict=True))


def find_low_variability_intervals(
    dispersion: PositionDispersion,
    criteria: LowVariabilityCriteria,
) -> tuple[LowVariabilityInterval, ...]:
    """Find contiguous low-RMS-radius intervals in point then time order."""
    requested_points = criteria.point_ids or dispersion.point_ids
    for point_id in requested_points:
        require(point_id in dispersion.point_ids, "unknown point_id", point_id)

    intervals: list[LowVariabilityInterval] = []
    for point_id in requested_points:
        point_index = dispersion.point_index(point_id)
        radii = dispersion.rms_radius_m[:, point_index]
        enough_trials = dispersion.count[:, point_index] >= MIN_TRIALS_FOR_COVARIANCE
        qualifying = (
            enough_trials & np.isfinite(radii) & (radii <= criteria.max_rms_radius_m)
        )
        for start, end in _true_runs(qualifying):
            n_samples = end - start + 1
            duration_s = float(
                dispersion.sample_times_s[end] - dispersion.sample_times_s[start]
            )
            if n_samples < criteria.min_samples or duration_s < criteria.min_duration_s:
                continue
            interval_radii = radii[start : end + 1]
            intervals.append(
                LowVariabilityInterval(
                    point_id=point_id,
                    start_index=start,
                    end_index=end,
                    start_time_s=float(dispersion.sample_times_s[start]),
                    end_time_s=float(dispersion.sample_times_s[end]),
                    n_samples=n_samples,
                    mean_rms_radius_m=float(np.mean(interval_radii)),
                    max_rms_radius_m=float(np.max(interval_radii)),
                )
            )
    return tuple(intervals)


__all__ = [
    "EnsemblePositionTraces",
    "LowVariabilityCriteria",
    "LowVariabilityInterval",
    "MAX_DISPERSION_ACCUMULATOR_BYTES",
    "PositionDispersion",
    "PositionDispersionAccumulator",
    "compute_position_dispersion",
    "compute_position_dispersion_view",
    "find_low_variability_intervals",
]
