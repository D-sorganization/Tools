"""Canonical 2-D and 3-D markerless-mocap observation records."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import (
    require_finite,
    require_nonnegative_integer,
    require_semver,
    require_text,
    require_unique_text,
)
from .enums import Availability


def _finite_tuple(
    values: tuple[float, ...], length: int, field_name: str
) -> tuple[float, ...]:
    normalized = tuple(require_finite(value, field_name) for value in values)
    if len(normalized) != length:
        raise ValueError(f"{field_name} must contain {length} values")
    return normalized


@dataclass(frozen=True, slots=True)
class SkeletonDefinition:
    """Versioned semantic keypoint set independent of a pose backend."""

    skeleton_id: str
    version: str
    keypoint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "skeleton_id", require_text(self.skeleton_id, "skeleton_id")
        )
        object.__setattr__(self, "version", require_semver(self.version, "version"))
        keypoints = require_unique_text(self.keypoint_ids, "keypoint_ids")
        if not keypoints:
            raise ValueError("keypoint_ids must be non-empty")
        object.__setattr__(self, "keypoint_ids", keypoints)


@dataclass(frozen=True, slots=True)
class PixelObservation:
    """One backend-labelled 2-D keypoint observation in pixel coordinates."""

    observation_id: str
    camera_id: str
    frame_sequence: int
    timestamp_ns: int
    skeleton_id: str
    keypoint_id: str
    uv_px: tuple[float, float]
    confidence: float
    covariance_px2: tuple[float, float, float, float]
    availability: Availability

    def __post_init__(self) -> None:
        for name in ("observation_id", "camera_id", "skeleton_id", "keypoint_id"):
            object.__setattr__(self, name, require_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "frame_sequence",
            require_nonnegative_integer(self.frame_sequence, "frame_sequence"),
        )
        object.__setattr__(
            self,
            "timestamp_ns",
            require_nonnegative_integer(self.timestamp_ns, "timestamp_ns"),
        )
        object.__setattr__(self, "uv_px", _finite_tuple(self.uv_px, 2, "uv_px"))
        confidence = require_finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        covariance = _finite_tuple(self.covariance_px2, 4, "covariance_px2")
        if covariance[0] < 0.0 or covariance[3] < 0.0:
            raise ValueError("covariance_px2 diagonal must be non-negative")
        object.__setattr__(self, "covariance_px2", covariance)
        if self.availability is not Availability.OBSERVED:
            raise ValueError("pixel observations must have observed availability")


@dataclass(frozen=True, slots=True)
class Landmark3D:
    """A 3-D landmark with provenance, covariance, and evidence classification."""

    landmark_id: str
    world_frame_id: str
    skeleton_id: str
    keypoint_id: str
    timestamp_ns: int
    xyz_m: tuple[float, float, float]
    covariance_m2: tuple[float, float, float, float, float, float, float, float, float]
    contributing_camera_ids: tuple[str, ...]
    rejected_camera_ids: tuple[str, ...]
    method_id: str
    availability: Availability

    def __post_init__(self) -> None:
        for name in (
            "landmark_id",
            "world_frame_id",
            "skeleton_id",
            "keypoint_id",
            "method_id",
        ):
            object.__setattr__(self, name, require_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "timestamp_ns",
            require_nonnegative_integer(self.timestamp_ns, "timestamp_ns"),
        )
        object.__setattr__(self, "xyz_m", _finite_tuple(self.xyz_m, 3, "xyz_m"))
        covariance = _finite_tuple(self.covariance_m2, 9, "covariance_m2")
        if any(covariance[index] < 0.0 for index in (0, 4, 8)):
            raise ValueError("covariance_m2 diagonal must be non-negative")
        object.__setattr__(self, "covariance_m2", covariance)
        contributing = require_unique_text(
            self.contributing_camera_ids, "contributing_camera_ids"
        )
        rejected = require_unique_text(self.rejected_camera_ids, "rejected_camera_ids")
        if set(contributing) & set(rejected):
            raise ValueError("contributing and rejected cameras must be disjoint")
        if self.availability is Availability.DERIVED and len(contributing) < 2:
            raise ValueError(
                "triangulated derived landmarks require two unique cameras"
            )
        object.__setattr__(self, "contributing_camera_ids", contributing)
        object.__setattr__(self, "rejected_camera_ids", rejected)


__all__: list[str] = []
