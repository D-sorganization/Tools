"""Denominator-matched absolute scatter and standardized input-noise response."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from shared.python.contracts import require

from .ensemble_geometry import compute_position_dispersion
from .ensemble_types import (
    EnsemblePositionTraces,
    immutable_array,
    require_coordinate_frame_id,
    require_point_ids,
    validated_sample_times,
)
from .spec import NoiseSpec

POSITION_NOISE_RESPONSE_METHOD = "empirical-centered-declared-distribution-sd-rms/v1"
_DISTRIBUTION_SD_DIVISOR = {
    "normal": 1.0,
    "uniform": math.sqrt(3.0),
    "triangular": math.sqrt(6.0),
}


def _declared_standard_deviation(spec: NoiseSpec) -> float:
    """Return the untruncated standard deviation declared by one noise spec."""
    divisor = _DISTRIBUTION_SD_DIVISOR[spec.distribution]
    return spec.scale / divisor


def _validate_nonnegative_finite_or_nan(name: str, values: np.ndarray) -> None:
    """Require every available magnitude to be finite and non-negative."""
    available = np.isfinite(values)
    require(
        bool(np.all(np.isnan(values) | available)),
        f"{name} must contain only finite values or NaN",
    )
    require(
        bool(np.all(values[available] >= 0.0)),
        f"{name} available values must be non-negative",
    )


@dataclass(frozen=True)
class PositionNoiseResponse:
    """Per-sample, per-point scatter and aggregate standardized-input gain.

    ``response_gain_m`` is absolute RMS positional scatter divided by the
    empirical RMS of the same valid rows' centered inputs after scaling each
    input by its declared distribution standard deviation. It is a
    model-conditional aggregate amplification, not causal attribution.
    """

    sample_times_s: np.ndarray = field(repr=False)
    coordinate_frame: str
    point_ids: tuple[str, ...]
    input_spec_ids: tuple[str, ...]
    count: np.ndarray = field(repr=False)
    absolute_rms_radius_m: np.ndarray = field(repr=False)
    standardized_input_rms: np.ndarray = field(repr=False)
    response_gain_m: np.ndarray = field(repr=False)
    normalization_method: str = POSITION_NOISE_RESPONSE_METHOD

    def __post_init__(self) -> None:
        times = validated_sample_times(self.sample_times_s)
        require_coordinate_frame_id(self.coordinate_frame)
        points = tuple(self.point_ids)
        require_point_ids(points)
        specs = tuple(self.input_spec_ids)
        require_point_ids(specs)
        shape = (times.size, len(points))
        arrays = {
            "count": np.asarray(self.count),
            "absolute_rms_radius_m": np.asarray(self.absolute_rms_radius_m),
            "standardized_input_rms": np.asarray(self.standardized_input_rms),
            "response_gain_m": np.asarray(self.response_gain_m),
        }
        for name, values in arrays.items():
            require(values.shape == shape, f"{name} has invalid shape", values.shape)
        require(
            bool(np.all(arrays["count"] >= 0)),
            "count must be non-negative",
            arrays["count"],
        )
        require(
            bool(np.all(arrays["count"] == np.floor(arrays["count"]))),
            "count must contain integer values",
            arrays["count"],
        )
        for name in arrays.keys() - {"count"}:
            _validate_nonnegative_finite_or_nan(name, arrays[name])
        require(
            self.normalization_method == POSITION_NOISE_RESPONSE_METHOD,
            "normalization_method is unsupported",
            self.normalization_method,
        )
        object.__setattr__(self, "sample_times_s", immutable_array(times, float))
        object.__setattr__(self, "point_ids", points)
        object.__setattr__(self, "input_spec_ids", specs)
        object.__setattr__(self, "count", immutable_array(arrays["count"], int))
        for name in arrays.keys() - {"count"}:
            object.__setattr__(self, name, immutable_array(arrays[name], float))


def _standardized_input_rms(traces: EnsemblePositionTraces) -> np.ndarray:
    """Compute denominator-matched aggregate input RMS without a 3-D tensor."""
    variation = traces.variation
    specs = variation.plan.noise
    expected_names = tuple(spec.variable_key for spec in specs)
    require(
        variation.input_names == expected_names,
        "variation input_names must match plan noise order",
        variation.input_names,
    )
    scales = np.asarray(
        [_declared_standard_deviation(spec) for spec in specs], dtype=float
    )
    resolved_base = variation.plan.resolved_base()
    bases = np.asarray([resolved_base[spec.variable_key] for spec in specs])
    standardized = (np.asarray(variation.inputs, dtype=float) - bases) / scales
    valid = np.asarray(traces.sample_valid, dtype=float)
    counts = np.sum(valid, axis=0)
    sums = valid.T @ standardized
    means = np.zeros_like(sums)
    np.divide(sums, counts[:, None], out=means, where=counts[:, None] > 0.0)
    sum_squares = valid.T @ np.square(standardized)
    centered_sum_squares = sum_squares - counts[:, None] * np.square(means)
    total_centered = np.sum(np.maximum(centered_sum_squares, 0.0), axis=1)
    result = np.full(counts.shape, np.nan, dtype=float)
    eligible = counts >= 2.0
    np.divide(total_centered, counts, out=result, where=eligible)
    np.sqrt(result, out=result, where=eligible)
    return result


def compute_position_noise_response(
    traces: EnsemblePositionTraces,
) -> PositionNoiseResponse:
    """Return absolute scatter and same-row standardized input-noise gain."""
    require(
        isinstance(traces, EnsemblePositionTraces),
        "traces must be EnsemblePositionTraces",
        type(traces).__name__,
    )
    dispersion = compute_position_dispersion(traces)
    sample_input_rms = _standardized_input_rms(traces)
    shape = dispersion.rms_radius_m.shape
    input_rms = np.broadcast_to(sample_input_rms[:, None], shape).copy()
    gain = np.full(shape, np.nan, dtype=float)
    eligible = (dispersion.count >= 2) & (input_rms > 0.0)
    np.divide(
        dispersion.rms_radius_m,
        input_rms,
        out=gain,
        where=eligible,
    )
    return PositionNoiseResponse(
        sample_times_s=dispersion.sample_times_s,
        coordinate_frame=dispersion.coordinate_frame,
        point_ids=dispersion.point_ids,
        input_spec_ids=tuple(str(spec.spec_id) for spec in traces.variation.plan.noise),
        count=dispersion.count,
        absolute_rms_radius_m=dispersion.rms_radius_m,
        standardized_input_rms=input_rms,
        response_gain_m=gain,
    )


__all__ = [
    "POSITION_NOISE_RESPONSE_METHOD",
    "PositionNoiseResponse",
    "compute_position_noise_response",
]
