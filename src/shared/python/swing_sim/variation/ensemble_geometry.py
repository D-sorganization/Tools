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
    positions = ensemble.positions_m
    valid = ensemble.sample_valid[:, :, np.newaxis, np.newaxis]
    counts_by_sample = np.count_nonzero(ensemble.sample_valid, axis=0)
    counts = np.broadcast_to(
        counts_by_sample[:, np.newaxis], (positions.shape[1], positions.shape[2])
    ).copy()
    position_sum = np.sum(np.where(valid, positions, 0.0), axis=0)
    divisor = counts_by_sample[:, np.newaxis, np.newaxis]
    mean = np.full(position_sum.shape, np.nan)
    np.divide(position_sum, divisor, out=mean, where=divisor > 0)

    centered = np.where(valid, positions - mean[np.newaxis, ...], 0.0)
    covariance_sum = np.einsum("tspc,tspd->spcd", centered, centered)
    covariance_divisor = counts[:, :, np.newaxis, np.newaxis] - 1
    covariance = np.full(covariance_sum.shape, np.nan)
    np.divide(
        covariance_sum, covariance_divisor, out=covariance, where=covariance_divisor > 0
    )

    squared_radius = np.einsum("tspc,tspc->tsp", centered, centered)
    radius_sum = np.sum(squared_radius, axis=0)
    rms_radius = np.full(counts.shape, np.nan)
    np.divide(radius_sum, counts, out=rms_radius, where=counts > 0)
    np.sqrt(rms_radius, out=rms_radius)
    eigenvalues, principal_axes = _principal_components(covariance, counts)

    return PositionDispersion(
        sample_times_s=immutable_array(ensemble.sample_times_s, float),
        coordinate_frame=ensemble.coordinate_frame,
        point_ids=ensemble.point_ids,
        count=immutable_array(counts, int),
        mean_positions_m=immutable_array(mean, float),
        covariance_m2=immutable_array(covariance, float),
        eigenvalues_m2=immutable_array(eigenvalues, float),
        principal_axes=immutable_array(principal_axes, float),
        rms_radius_m=immutable_array(rms_radius, float),
    )


def _true_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return inclusive index bounds of contiguous true regions."""
    padded = np.pad(mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
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
    "PositionDispersion",
    "compute_position_dispersion",
    "find_low_variability_intervals",
]
