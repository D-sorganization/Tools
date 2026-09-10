"""Identified common references and multi-view placements in explicit SI frames.

These additive contracts describe physical point identities, not image-local
top-left corners. Intrinsics are fixed inputs; no zoom or field accuracy is inferred.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from ._validation import require_finite, require_nonnegative_integer, require_text
from .calibration import CalibrationObservation, FisheyeIntrinsics, PinholeIntrinsics
from .extrinsics import CameraLayout
from .geometry import CoordinateFrame, RigidTransform

__all__ = [
    "ReferenceTarget",
    "ReferenceCamera",
    "PlacementObservation",
    "PlacementResidual",
    "ReferenceLayoutResult",
    "ReferenceSolveCancelled",
    "common_reference",
    "estimate_reference_layout",
]


def _points(
    values: Sequence[Sequence[float]], dimensions: int
) -> tuple[tuple[float, ...], ...]:
    result = tuple(
        tuple(require_finite(x, "point coordinate") for x in p) for p in values
    )
    if not result or any(len(p) != dimensions for p in result):
        raise ValueError(f"points must each have {dimensions} finite coordinates")
    return result


@dataclass(frozen=True, slots=True)
class ReferenceTarget:
    """Ordered physical points in one rigid reference's local metre frame.

    Rectangle origin is the marked corner, +x follows its long edge/arrow, +z
    follows its width, and +y is up from the marked face. Keep these identities
    across cameras; they are not independently reordered by image position.
    """

    reference_id: str
    point_ids: tuple[str, ...]
    object_points_m: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference_id", require_text(self.reference_id, "reference_id")
        )
        identifiers = tuple(require_text(key, "point ID") for key in self.point_ids)
        points = _points(self.object_points_m, 3)
        if not 2 <= len(points) <= 10000 or len(identifiers) != len(points):
            raise ValueError("reference needs 2–10000 identified points")
        if len(set(identifiers)) != len(points) or len(set(points)) != len(points):
            raise ValueError("reference point IDs and coordinates must be unique")
        object.__setattr__(self, "point_ids", identifiers)
        object.__setattr__(self, "object_points_m", points)

    @classmethod
    def rectangle(
        cls, reference_id: str, width_m: float, length_m: float
    ) -> ReferenceTarget:
        """Describe four marked corners; positive measured dimensions are in m."""
        width = require_finite(width_m, "width_m")
        length = require_finite(length_m, "length_m")
        if min(width, length) <= 0:
            raise ValueError("reference dimensions must be positive")
        return cls(
            reference_id,
            ("origin", "along-arrow", "opposite", "across-width"),
            (
                (0.0, 0.0, 0.0),
                (length, 0.0, 0.0),
                (length, 0.0, width),
                (0.0, 0.0, width),
            ),
        )

    @classmethod
    def line(cls, reference_id: str, length_m: float) -> ReferenceTarget:
        """Describe labelled endpoints for scale; a line cannot initialize PnP."""
        length = require_finite(length_m, "length_m")
        if length <= 0:
            raise ValueError("reference length must be positive")
        return cls(
            reference_id, ("origin", "endpoint"), ((0.0, 0.0, 0.0), (length, 0.0, 0.0))
        )


def common_reference(reference_id: str) -> ReferenceTarget:
    """Return nominal US Letter, A4, yardstick or metre-stick dimensions.

    Consumers must let users replace nominal sizes with measured custom sizes.
    Paper must lie flat; ruler measurement endpoints are not necessarily its ends.
    """
    rectangles = {"us-letter": (0.2159, 0.2794), "a4": (0.210, 0.297)}
    lines = {"yardstick": 0.9144, "meter-stick": 1.0}
    if reference_id in rectangles:
        return ReferenceTarget.rectangle(reference_id, *rectangles[reference_id])
    if reference_id in lines:
        return ReferenceTarget.line(reference_id, lines[reference_id])
    raise ValueError("unknown common reference; supply measured custom dimensions")


@dataclass(frozen=True, slots=True)
class ReferenceCamera:
    """Fixed canonical intrinsics with an opaque, immutable profile revision ID."""

    camera_key: str
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics
    profile_id: str

    def __post_init__(self) -> None:
        for name in ("camera_key", "profile_id"):
            object.__setattr__(self, name, require_text(getattr(self, name), name))
        if not isinstance(self.intrinsics, (PinholeIntrinsics, FisheyeIntrinsics)):
            raise TypeError("intrinsics must use the canonical camera model")


@dataclass(frozen=True, slots=True)
class PlacementObservation:
    """One camera/frame observing one stationary, identified reference placement.

    A new physical placement needs a new ID shared by its observing cameras.
    Held-out observations are evaluated after fitting and never seed the solve.
    """

    placement_id: str
    reference_id: str
    camera_key: str
    profile_id: str
    point_ids: tuple[str, ...]
    pixels_px: tuple[tuple[float, float], ...]
    frame_sequence: int
    timestamp_ns: int
    held_out: bool = False
    detection_confidence: float = 0.0

    def __post_init__(self) -> None:
        for name in ("placement_id", "reference_id", "camera_key", "profile_id"):
            object.__setattr__(self, name, require_text(getattr(self, name), name))
        ids = tuple(require_text(key, "point ID") for key in self.point_ids)
        pixels = _points(self.pixels_px, 2)
        if len(ids) != len(pixels) or len(set(ids)) != len(ids):
            raise ValueError("each pixel needs one unique physical point ID")
        if not isinstance(self.held_out, bool):
            raise TypeError("held_out must be boolean")
        confidence = require_finite(self.detection_confidence, "detection_confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("detection_confidence must be in [0, 1]")
        object.__setattr__(self, "detection_confidence", confidence)
        for name in ("frame_sequence", "timestamp_ns"):
            object.__setattr__(
                self, name, require_nonnegative_integer(getattr(self, name), name)
            )
        object.__setattr__(self, "point_ids", ids)
        object.__setattr__(self, "pixels_px", pixels)

    def calibration_observation(
        self, target: ReferenceTarget
    ) -> CalibrationObservation:
        """Adapt identified points to the existing canonical observation contract."""
        if target.reference_id != self.reference_id:
            raise ValueError("observation reference does not match target")
        lookup = dict(zip(target.point_ids, target.object_points_m, strict=True))
        if any(key not in lookup for key in self.point_ids):
            raise ValueError("unknown physical point ID")
        return CalibrationObservation(
            self.camera_key,
            self.frame_sequence,
            self.timestamp_ns,
            tuple(lookup[key] for key in self.point_ids),
            self.pixels_px,
            self.detection_confidence,
        )


@dataclass(frozen=True, slots=True)
class PlacementResidual:
    """Pixel error evidence for one view; no physical accuracy certification."""

    placement_id: str
    camera_key: str
    held_out: bool
    mean_error_px: float
    max_error_px: float
    point_count: int

    def __post_init__(self) -> None:
        for name in ("placement_id", "camera_key"):
            object.__setattr__(self, name, require_text(getattr(self, name), name))
        mean = require_finite(self.mean_error_px, "mean_error_px")
        maximum = require_finite(self.max_error_px, "max_error_px")
        count = require_nonnegative_integer(self.point_count, "point_count")
        if not 0 <= mean <= maximum or count == 0:
            raise ValueError("residuals need 0 <= mean <= maximum and observed points")
        if not isinstance(self.held_out, bool):
            raise TypeError("held_out must be boolean")


@dataclass(frozen=True, slots=True)
class ReferenceLayoutResult:
    """Fitted cameras and target poses plus unmodified validation evidence."""

    layout: CameraLayout
    anchor_placement_id: str
    placement_transforms: Mapping[str, RigidTransform]
    residuals: tuple[PlacementResidual, ...]
    limitations: tuple[str, ...]
    observations: tuple[PlacementObservation, ...]
    camera_profiles: Mapping[str, str]

    def __post_init__(self) -> None:
        layout = CameraLayout(
            self.layout.layout_id, self.layout.world_frame, self.layout.camera_poses
        )
        object.__setattr__(
            layout, "camera_poses", MappingProxyType(dict(layout.camera_poses))
        )
        object.__setattr__(self, "layout", layout)
        object.__setattr__(
            self,
            "placement_transforms",
            MappingProxyType(dict(self.placement_transforms)),
        )
        object.__setattr__(self, "residuals", tuple(self.residuals))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(
            self, "camera_profiles", MappingProxyType(dict(self.camera_profiles))
        )


class ReferenceSolveCancelled(RuntimeError):
    """The caller cancelled a reference solve before it produced a result."""


def estimate_reference_layout(
    *,
    layout_id: str,
    world_frame: CoordinateFrame,
    targets: Mapping[str, ReferenceTarget],
    cameras: Mapping[str, ReferenceCamera],
    observations: Sequence[PlacementObservation],
    anchor_placement_id: str,
    anchor: RigidTransform,
    cancel_requested: Callable[[], bool] | None = None,
) -> ReferenceLayoutResult:
    """Jointly fit camera/target poses with fixed intrinsics and an explicit anchor.

    The anchor maps ``placement:<anchor_placement_id>`` into the named SI world.
    It fixes the gauge; callers must establish its physical origin/orientation.
    Returns numerical evidence, never a claim that paper alone calibrated a lens.
    At least two cameras and two connected, non-collinear reference placements
    are required. Each fitted view needs at least four identified points.
    """
    from .placement_solver import solve

    return solve(
        layout_id,
        world_frame,
        targets,
        cameras,
        tuple(observations),
        anchor_placement_id,
        anchor,
        cancel_requested,
    )
