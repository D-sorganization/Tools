"""Plot-ready geometric variability and quiet-zone data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.python.contracts import require
from shared.python.swing_sim.variation import (
    LowVariabilityCriteria,
    LowVariabilityInterval,
    PositionDispersion,
    find_low_variability_intervals,
)
from shared.python.swing_sim.variation.ensemble_types import immutable_array

COMMON_TIME_ALIGNMENT = "common_simulation_time_s"


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


__all__ = [
    "COMMON_TIME_ALIGNMENT",
    "GeometricVariabilityData",
    "build_geometric_variability",
]
