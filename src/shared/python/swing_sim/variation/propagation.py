"""Honest paired-intervention propagation over common-grid position traces.

This module measures geometric displacement induced by a planted intervention.
It does not infer causality from association, rank correlation, or observational
multi-input trials. Global association and correlation analysis remain separate
future methods and must not be presented as paired intervention results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.python.contracts import require

from .ensemble_types import (
    CARTESIAN_DIMENSIONS,
    EnsemblePositionTraces,
    immutable_array,
    require_coordinate_frame_id,
    require_point_ids,
    validated_sample_times,
)
from .spec import NoiseSpec


def _validate_grid(
    sample_times_s: np.ndarray, coordinate_frame: str, point_ids: tuple[str, ...]
) -> np.ndarray:
    """Validate and return a common sample grid."""
    times = validated_sample_times(sample_times_s)
    require_coordinate_frame_id(coordinate_frame)
    require_point_ids(point_ids)
    return times


@dataclass(frozen=True)
class CommonReferenceTrace:
    """One immutable reference trace intentionally shared by all trial rows."""

    sample_times_s: np.ndarray = field(repr=False)
    coordinate_frame: str
    point_ids: tuple[str, ...]
    positions_m: np.ndarray = field(repr=False)
    sample_valid: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        points = tuple(self.point_ids)
        times = _validate_grid(self.sample_times_s, self.coordinate_frame, points)
        positions = np.asarray(self.positions_m, dtype=float)
        valid = np.asarray(self.sample_valid, dtype=bool)
        expected = (times.size, len(points), CARTESIAN_DIMENSIONS)
        require(
            positions.shape == expected,
            "reference positions have invalid shape",
            positions.shape,
        )
        require(
            valid.shape == (times.size,),
            "reference sample_valid has invalid shape",
            valid.shape,
        )
        require(
            np.all(np.isfinite(positions[valid])),
            "valid reference samples must be finite",
            None,
        )
        require(
            np.all(np.isnan(positions[~valid])),
            "invalid reference samples must contain only NaN",
            None,
        )
        object.__setattr__(self, "sample_times_s", immutable_array(times, float))
        object.__setattr__(self, "point_ids", points)
        object.__setattr__(self, "positions_m", immutable_array(positions, float))
        object.__setattr__(self, "sample_valid", immutable_array(valid, bool))


BaselineTrace = EnsemblePositionTraces | CommonReferenceTrace


def _validate_layout(
    baseline: BaselineTrace, perturbed: EnsemblePositionTraces
) -> None:
    """Require identical physical layout and, for ensembles, paired trial rows."""
    require(
        baseline.coordinate_frame == perturbed.coordinate_frame,
        "baseline and perturbed frame must match",
        (baseline.coordinate_frame, perturbed.coordinate_frame),
    )
    require(
        np.array_equal(baseline.sample_times_s, perturbed.sample_times_s),
        "baseline and perturbed time grids must match",
        None,
    )
    require(
        baseline.point_ids == perturbed.point_ids,
        "baseline and perturbed point layouts must match",
        (baseline.point_ids, perturbed.point_ids),
    )
    if isinstance(baseline, EnsemblePositionTraces):
        require(
            baseline.n_trials == perturbed.n_trials,
            "paired baseline and perturbed trial counts must match",
            (baseline.n_trials, perturbed.n_trials),
        )


@dataclass(frozen=True)
class PairedIntervention:
    """A stable perturbation specification associated with paired geometry.

    The provider contract is that baseline and perturbed data differ only by
    this specification. Trace arrays cannot independently prove that OAT
    intervention design, so callers must preserve its provenance.
    """

    spec: NoiseSpec
    baseline: BaselineTrace
    perturbed: EnsemblePositionTraces

    def __post_init__(self) -> None:
        require(isinstance(self.spec, NoiseSpec), "spec must be a NoiseSpec", self.spec)
        require(
            isinstance(self.baseline, (EnsemblePositionTraces, CommonReferenceTrace)),
            "baseline must be an ensemble or common reference trace",
            type(self.baseline).__name__,
        )
        require(
            isinstance(self.perturbed, EnsemblePositionTraces),
            "perturbed must be EnsemblePositionTraces",
            type(self.perturbed).__name__,
        )
        _validate_layout(self.baseline, self.perturbed)


@dataclass(frozen=True)
class PropagationResult:
    """Per-sample/per-point paired displacement induced by one source spec."""

    spec_id: str
    variable_key: str
    source_time_window_s: tuple[float, float] | None
    source_point_ids: tuple[str, ...]
    reference_kind: str
    sample_times_s: np.ndarray = field(repr=False)
    coordinate_frame: str
    point_ids: tuple[str, ...]
    paired_count: np.ndarray = field(repr=False)
    mean_displacement_m: np.ndarray = field(repr=False)
    rms_induced_displacement_m: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        require(
            isinstance(self.spec_id, str)
            and bool(self.spec_id)
            and self.spec_id == self.spec_id.strip(),
            "spec_id must be a non-empty, trimmed stable ID",
            self.spec_id,
        )
        require(
            isinstance(self.variable_key, str)
            and bool(self.variable_key)
            and self.variable_key == self.variable_key.strip(),
            "variable_key must be a non-empty, trimmed registry key",
            self.variable_key,
        )
        points = tuple(self.point_ids)
        times = _validate_grid(self.sample_times_s, self.coordinate_frame, points)
        shape = (times.size, len(points))
        count = np.asarray(self.paired_count, dtype=int)
        mean = np.asarray(self.mean_displacement_m, dtype=float)
        rms = np.asarray(self.rms_induced_displacement_m, dtype=float)
        require(
            self.reference_kind in ("paired", "common"),
            "invalid reference_kind",
            self.reference_kind,
        )
        require(
            count.shape == shape and np.all(count >= 0),
            "paired_count has invalid shape or values",
            count,
        )
        require(
            mean.shape == shape + (3,),
            "mean_displacement_m has invalid shape",
            mean.shape,
        )
        require(
            rms.shape == shape,
            "rms_induced_displacement_m has invalid shape",
            rms.shape,
        )
        observed = count > 0
        require(
            np.all(np.isfinite(mean[observed]))
            and np.all(np.isfinite(rms[observed]))
            and np.all(rms[observed] >= 0.0),
            "observed propagation values must be finite and RMS non-negative",
            None,
        )
        require(
            np.all(np.isnan(mean[~observed])) and np.all(np.isnan(rms[~observed])),
            "unobserved propagation values must contain only NaN",
            None,
        )
        source_points = tuple(self.source_point_ids)
        if source_points:
            require_point_ids(source_points)
        object.__setattr__(self, "sample_times_s", immutable_array(times, float))
        object.__setattr__(self, "point_ids", points)
        object.__setattr__(self, "source_point_ids", source_points)
        object.__setattr__(self, "paired_count", immutable_array(count, int))
        object.__setattr__(self, "mean_displacement_m", immutable_array(mean, float))
        object.__setattr__(
            self, "rms_induced_displacement_m", immutable_array(rms, float)
        )


def _baseline_arrays(
    intervention: PairedIntervention,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return baseline arrays broadcast to perturbed trial rows."""
    baseline = intervention.baseline
    perturbed = intervention.perturbed
    if isinstance(baseline, EnsemblePositionTraces):
        return baseline.positions_m, baseline.sample_valid, "paired"
    positions = np.broadcast_to(baseline.positions_m, perturbed.positions_m.shape)
    valid = np.broadcast_to(baseline.sample_valid, perturbed.sample_valid.shape)
    return positions, valid, "common"


def analyze_paired_intervention(intervention: PairedIntervention) -> PropagationResult:
    """Measure paired displacement without interpreting association as causation."""
    baseline_positions, baseline_valid, reference_kind = _baseline_arrays(intervention)
    perturbed = intervention.perturbed
    paired_valid = baseline_valid & perturbed.sample_valid
    valid = paired_valid[:, :, np.newaxis, np.newaxis]
    displacement = np.where(valid, perturbed.positions_m - baseline_positions, 0.0)
    count_by_sample = np.count_nonzero(paired_valid, axis=0)
    counts = np.broadcast_to(
        count_by_sample[:, np.newaxis],
        (perturbed.positions_m.shape[1], perturbed.positions_m.shape[2]),
    ).copy()
    displacement_sum = np.sum(displacement, axis=0)
    divisor = count_by_sample[:, np.newaxis, np.newaxis]
    mean = np.full(displacement_sum.shape, np.nan)
    np.divide(displacement_sum, divisor, out=mean, where=divisor > 0)
    squared_magnitude = np.einsum("tspc,tspc->tsp", displacement, displacement)
    rms_squared = np.full(counts.shape, np.nan)
    np.divide(
        np.sum(squared_magnitude, axis=0), counts, out=rms_squared, where=counts > 0
    )
    rms = np.sqrt(rms_squared)
    spec = intervention.spec
    assert spec.spec_id is not None
    return PropagationResult(
        spec_id=spec.spec_id,
        variable_key=spec.variable_key,
        source_time_window_s=spec.time_window_s,
        source_point_ids=spec.point_ids,
        reference_kind=reference_kind,
        sample_times_s=perturbed.sample_times_s,
        coordinate_frame=perturbed.coordinate_frame,
        point_ids=perturbed.point_ids,
        paired_count=counts,
        mean_displacement_m=mean,
        rms_induced_displacement_m=rms,
    )


__all__ = [
    "CommonReferenceTrace",
    "PairedIntervention",
    "PropagationResult",
    "analyze_paired_intervention",
]
