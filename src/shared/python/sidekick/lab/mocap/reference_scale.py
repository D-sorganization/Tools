"""Metric-scale evidence for an established camera layout, never pose initialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from .extrinsics import CameraLayout
from .reference_placements import PlacementObservation, ReferenceCamera, ReferenceTarget


@dataclass(frozen=True, slots=True)
class ReferenceScaleProblem:
    """Snapshot a layout, its lens revisions, and identified stationary rulers.

    Preconditions: layout coordinates use metres; relative poses already exist.
    This operation cannot discover camera orientation or detect unreported movement.
    """

    layout: CameraLayout
    cameras: Mapping[str, ReferenceCamera]
    targets: Mapping[str, ReferenceTarget]
    observations: tuple[PlacementObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layout, CameraLayout):
            raise TypeError("layout must be a CameraLayout")
        snapshot = replace(self.layout)
        object.__setattr__(
            snapshot, "camera_poses", MappingProxyType(dict(snapshot.camera_poses))
        )
        object.__setattr__(self, "layout", snapshot)
        object.__setattr__(self, "cameras", MappingProxyType(dict(self.cameras)))
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))
        object.__setattr__(self, "observations", tuple(self.observations))


@dataclass(frozen=True, slots=True)
class ReferenceScaleOptions:
    """Review thresholds in degrees, pixels, and dimensionless relative length.

    The fixed world anchor remains unchanged when camera translations are scaled.
    Thresholds are software rejection limits, not physical accuracy guarantees.
    """

    anchor_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    min_parallax_degrees: float = 1.0
    max_reprojection_error_px: float = 2.0
    max_relative_error: float = 0.05

    def __post_init__(self) -> None:
        anchor = np.asarray(self.anchor_m, dtype=float)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError("anchor must contain three finite metre coordinates")
        limits = (
            self.min_parallax_degrees,
            self.max_reprojection_error_px,
            self.max_relative_error,
        )
        if not np.isfinite(limits).all() or any(value <= 0 for value in limits):
            raise ValueError("scale review thresholds must be finite and positive")
        if self.min_parallax_degrees >= 90 or self.max_relative_error >= 1:
            raise ValueError(
                "parallax must be below 90 degrees and relative error below one"
            )
        object.__setattr__(self, "anchor_m", tuple(float(value) for value in anchor))


@dataclass(frozen=True, slots=True)
class ScaleResidual:
    """Reconstructed and corrected ruler length with independent holdout status."""

    placement_id: str
    known_length_m: float
    reconstructed_length_m: float
    scaled_length_m: float
    relative_error: float
    held_out: bool
    min_parallax_degrees: float
    max_reprojection_error_px: float


@dataclass(frozen=True, slots=True)
class ReferenceScaleResult:
    """Scale candidate; repeatability excludes lens and reference uncertainty."""

    source_layout_id: str
    scaled_layout: CameraLayout
    scale_factor: float
    fit_placement_ids: tuple[str, ...]
    held_out_placement_ids: tuple[str, ...]
    residuals: tuple[ScaleResidual, ...]
    scale_repeatability: float | None
    camera_profiles: Mapping[str, str]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "camera_profiles", MappingProxyType(dict(self.camera_profiles))
        )


def estimate_reference_scale(
    problem: ReferenceScaleProblem, options: ReferenceScaleOptions | None = None
) -> ReferenceScaleResult:
    """Fit a positive global scale without changing rotations or source evidence.

    Args:
        problem: Existing geometry and stationary, identified ruler observations.
        options: Explicit world anchor and geometric rejection thresholds.

    Returns:
        A new layout and per-placement evidence requiring consumer review.

    Raises:
        TypeError: Inputs are not the canonical scale contracts.
        ValueError: Geometry, profile associations, or length evidence is unusable.
    """
    from .reference_scale_fit import fit_scale
    from .reference_scale_triangulation import measure_placements

    if not isinstance(problem, ReferenceScaleProblem):
        raise TypeError("problem must be a ReferenceScaleProblem")
    if options is not None and not isinstance(options, ReferenceScaleOptions):
        raise TypeError("options must be ReferenceScaleOptions")
    options = options or ReferenceScaleOptions()
    measurements = measure_placements(problem, options)
    return fit_scale(problem, measurements, options)


__all__ = [
    "ReferenceScaleOptions",
    "ReferenceScaleProblem",
    "ReferenceScaleResult",
    "ScaleResidual",
    "estimate_reference_scale",
]
