"""Camera intrinsic calibration representations, distortion, and qualification."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ._validation import (
    require_finite,
    require_nonnegative_integer,
    require_text,
)


class DistortionModel(StrEnum):
    """Supported vendor-neutral lens distortion formulations."""

    NONE = "none"
    BROWN_CONRADY = "brown-conrady"
    RATIONAL = "rational"
    KANNALA_BRANDT = "kannala-brandt"


class CalibrationPatternKind(StrEnum):
    """Recognized calibration board or feature target patterns."""

    CHESSBOARD = "chessboard"
    CHARUCO = "charuco"
    CIRCLES_GRID = "circles-grid"
    CUSTOM = "custom"


class CalibrationDegeneracyKind(StrEnum):
    """Specific geometric or statistical degeneracies in calibration data."""

    NONE = "none"
    INSUFFICIENT_VIEWS = "insufficient-views"
    PLANAR_DEGENERACY = "planar-degeneracy"
    NARROW_VIEWING_ANGLE = "narrow-viewing-angle"
    POOR_SENSOR_COVERAGE = "poor-sensor-coverage"
    HIGH_REPROJECTION_ERROR = "high-reprojection-error"


class CalibrationQuality(StrEnum):
    """Categorical qualification floor for camera intrinsics."""

    QUALIFIED = "qualified"
    DEGRADED = "degraded"
    UNQUALIFIED = "unqualified"


@dataclass(frozen=True, slots=True)
class DistortionCoefficients:
    """Lens distortion parameters with explicit mathematical model type."""

    model: DistortionModel
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, DistortionModel):
            raise TypeError("model must be a DistortionModel")
        coeffs = tuple(
            require_finite(c, "distortion coefficient") for c in self.coefficients
        )
        object.__setattr__(self, "coefficients", coeffs)

        if self.model is DistortionModel.NONE and len(coeffs) != 0:
            raise ValueError("DistortionModel.NONE requires empty coefficients")
        elif self.model is DistortionModel.BROWN_CONRADY and len(coeffs) not in {
            4,
            5,
            8,
        }:
            raise ValueError(
                "Brown-Conrady distortion requires 4, 5, or 8 coefficients"
            )
        elif self.model is DistortionModel.RATIONAL and len(coeffs) not in {8, 12, 14}:
            raise ValueError("Rational distortion requires 8, 12, or 14 coefficients")
        elif self.model is DistortionModel.KANNALA_BRANDT and len(coeffs) != 4:
            raise ValueError(
                "Kannala-Brandt fisheye distortion requires 4 coefficients"
            )


@dataclass(frozen=True, slots=True)
class PinholeIntrinsics:
    """Standard pinhole camera intrinsic parameters with distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    resolution_px: tuple[int, int]
    distortion: DistortionCoefficients
    skew: float = 0.0

    def __post_init__(self) -> None:
        fx = require_finite(self.fx, "fx")
        fy = require_finite(self.fy, "fy")
        cx = require_finite(self.cx, "cx")
        cy = require_finite(self.cy, "cy")
        skew = require_finite(self.skew, "skew")

        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("focal length must be positive")

        if len(self.resolution_px) != 2:
            raise ValueError("resolution_px must contain two values")
        w, h = self.resolution_px
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            raise ValueError("resolution_px dimensions must be positive integers")

        if not (0.0 <= cx <= float(w) and 0.0 <= cy <= float(h)):
            raise ValueError("principal point must be within image bounds")

        if not isinstance(self.distortion, DistortionCoefficients):
            raise TypeError("distortion must be DistortionCoefficients")

        object.__setattr__(self, "fx", fx)
        object.__setattr__(self, "fy", fy)
        object.__setattr__(self, "cx", cx)
        object.__setattr__(self, "cy", cy)
        object.__setattr__(self, "skew", skew)

    def project_point(self, xyz: tuple[float, float, float]) -> tuple[float, float]:
        """Project a 3-D camera-frame point (X, Y, Z) to pixel coordinates (u, v)."""
        x, y, z = (require_finite(val, "point coordinate") for val in xyz)
        if z <= 0.0:
            raise ValueError("cannot project point with non-positive depth")

        # Normalized coordinates
        xn = x / z
        yn = y / z

        if self.distortion.model is DistortionModel.NONE:
            u = self.fx * xn + self.skew * yn + self.cx
            v = self.fy * yn + self.cy
            return (u, v)

        if self.distortion.model is DistortionModel.BROWN_CONRADY:
            coeffs = self.distortion.coefficients
            k1, k2 = coeffs[0], coeffs[1]
            p1, p2 = (coeffs[2], coeffs[3]) if len(coeffs) >= 4 else (0.0, 0.0)
            k3 = coeffs[4] if len(coeffs) >= 5 else 0.0

            r2 = xn * xn + yn * yn
            radial = 1.0 + k1 * r2 + k2 * (r2 * r2) + k3 * (r2 * r2 * r2)
            dx_tangential = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
            dy_tangential = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn

            xd = xn * radial + dx_tangential
            yd = yn * radial + dy_tangential

            u = self.fx * xd + self.skew * yd + self.cx
            v = self.fy * yd + self.cy
            return (u, v)

        # Fallback to undistorted
        u = self.fx * xn + self.skew * yn + self.cx
        v = self.fy * yn + self.cy
        return (u, v)

    def unproject_point(self, uv: tuple[float, float]) -> tuple[float, float, float]:
        """Unproject pixel (u, v) to a normalized unit ray (x, y, z) in camera frame."""
        u, v = (require_finite(val, "pixel coordinate") for val in uv)
        # Undistort approx (for NONE, exact)
        yn = (v - self.cy) / self.fy
        xn = (u - self.cx - self.skew * yn) / self.fx
        norm = math.sqrt(xn * xn + yn * yn + 1.0)
        return (xn / norm, yn / norm, 1.0 / norm)


@dataclass(frozen=True, slots=True)
class FisheyeIntrinsics:
    """Equidistant / Kannala-Brandt fisheye camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    resolution_px: tuple[int, int]
    distortion: DistortionCoefficients

    def __post_init__(self) -> None:
        fx = require_finite(self.fx, "fx")
        fy = require_finite(self.fy, "fy")
        cx = require_finite(self.cx, "cx")
        cy = require_finite(self.cy, "cy")

        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("focal length must be positive")

        if len(self.resolution_px) != 2:
            raise ValueError("resolution_px must contain two values")
        w, h = self.resolution_px
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            raise ValueError("resolution_px dimensions must be positive integers")

        if not (0.0 <= cx <= float(w) and 0.0 <= cy <= float(h)):
            raise ValueError("principal point must be within image bounds")

        if not isinstance(self.distortion, DistortionCoefficients):
            raise TypeError("distortion must be DistortionCoefficients")

        object.__setattr__(self, "fx", fx)
        object.__setattr__(self, "fy", fy)
        object.__setattr__(self, "cx", cx)
        object.__setattr__(self, "cy", cy)

    def project_point(self, xyz: tuple[float, float, float]) -> tuple[float, float]:
        """Project a 3-D camera-frame point (X, Y, Z) to fisheye pixel coordinates."""
        x, y, z = (require_finite(val, "point coordinate") for val in xyz)
        if z <= 0.0:
            raise ValueError("cannot project point with non-positive depth")

        r = math.sqrt(x * x + y * y)
        if r < 1e-12:
            return (self.cx, self.cy)

        theta = math.atan2(r, z)
        if (
            self.distortion.model is DistortionModel.KANNALA_BRANDT
            and self.distortion.coefficients
        ):
            k1, k2, k3, k4 = self.distortion.coefficients
            theta2 = theta * theta
            theta_d = theta * (
                1.0
                + k1 * theta2
                + k2 * (theta2 * theta2)
                + k3 * (theta2 * theta2 * theta2)
                + k4 * (theta2 * theta2 * theta2 * theta2)
            )
        else:
            theta_d = theta

        scale = theta_d / r
        xd = x * scale
        yd = y * scale

        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy
        return (u, v)

    def unproject_point(self, uv: tuple[float, float]) -> tuple[float, float, float]:
        """Unproject pixel (u, v) to a normalized unit ray (x, y, z) in camera frame."""
        u, v = (require_finite(val, "pixel coordinate") for val in uv)
        xd = (u - self.cx) / self.fx
        yd = (v - self.cy) / self.fy
        theta_d = math.sqrt(xd * xd + yd * yd)
        if theta_d < 1e-12:
            return (0.0, 0.0, 1.0)

        # For zero or tiny distortion, theta = theta_d
        theta = theta_d
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        scale = sin_theta / theta_d
        x = xd * scale
        y = yd * scale
        z = cos_theta
        norm = math.sqrt(x * x + y * y + z * z)
        return (x / norm, y / norm, z / norm)


@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    """Physical target specifications (e.g. checkerboard grid size and pitch)."""

    pattern_kind: CalibrationPatternKind
    grid_size: tuple[int, int]
    feature_spacing_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.pattern_kind, CalibrationPatternKind):
            raise TypeError("pattern_kind must be a CalibrationPatternKind")
        if len(self.grid_size) != 2 or self.grid_size[0] < 2 or self.grid_size[1] < 2:
            raise ValueError("grid_size dimensions must be at least 2")
        spacing = require_finite(self.feature_spacing_m, "feature_spacing_m")
        if spacing <= 0.0:
            raise ValueError("feature_spacing_m must be positive")
        object.__setattr__(self, "feature_spacing_m", spacing)

    @property
    def total_points(self) -> int:
        """Total number of internal keypoints on this calibration target."""
        return self.grid_size[0] * self.grid_size[1]


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    """Detected feature points and nominal 3-D object target points for one frame."""

    camera_key: str
    frame_sequence: int
    timestamp_ns: int
    target_object_points: tuple[tuple[float, float, float], ...]
    detected_pixel_points: tuple[tuple[float, float], ...]
    detection_confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "camera_key", require_text(self.camera_key, "camera_key")
        )
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
        confidence = require_finite(self.detection_confidence, "detection_confidence")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("detection_confidence must be in [0, 1]")
        object.__setattr__(self, "detection_confidence", confidence)

        if len(self.target_object_points) != len(self.detected_pixel_points):
            raise ValueError(
                "target_object_points and detected_pixel_points count must match"
            )


@dataclass(frozen=True, slots=True)
class ReprojectionResidual:
    """Individual 2-D residual vector and norm for one observed calibration point."""

    observation_index: int
    point_index: int
    residual_uv_px: tuple[float, float]
    norm_px: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_index",
            require_nonnegative_integer(self.observation_index, "observation_index"),
        )
        object.__setattr__(
            self,
            "point_index",
            require_nonnegative_integer(self.point_index, "point_index"),
        )
        u_res = require_finite(self.residual_uv_px[0], "residual_u")
        v_res = require_finite(self.residual_uv_px[1], "residual_v")
        norm = require_finite(self.norm_px, "norm_px")
        if norm < 0.0:
            raise ValueError("norm_px must be non-negative")
        expected_norm = math.hypot(u_res, v_res)
        if not math.isclose(norm, expected_norm, rel_tol=1e-4, abs_tol=1e-5):
            raise ValueError("norm_px does not match residual_uv_px Euclidean norm")
        object.__setattr__(self, "residual_uv_px", (u_res, v_res))
        object.__setattr__(self, "norm_px", norm)


@dataclass(frozen=True, slots=True)
class IntrinsicCalibrationResult:
    """Result of intrinsic calibration with covariance and qualification floor."""

    camera_key: str
    calibrated_at_utc: str
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics
    mean_reprojection_residual_px: float
    max_reprojection_residual_px: float
    parameter_covariance: tuple[tuple[float, ...], ...]
    quality: CalibrationQuality
    degeneracies: tuple[CalibrationDegeneracyKind, ...] = ()
    reprojection_residuals: tuple[ReprojectionResidual, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "camera_key", require_text(self.camera_key, "camera_key")
        )
        object.__setattr__(
            self,
            "calibrated_at_utc",
            require_text(self.calibrated_at_utc, "calibrated_at_utc"),
        )
        if not isinstance(self.intrinsics, (PinholeIntrinsics, FisheyeIntrinsics)):
            raise TypeError("intrinsics must be PinholeIntrinsics or FisheyeIntrinsics")
        mean_res = require_finite(
            self.mean_reprojection_residual_px, "mean_reprojection_residual_px"
        )
        max_res = require_finite(
            self.max_reprojection_residual_px, "max_reprojection_residual_px"
        )
        if mean_res < 0.0 or max_res < 0.0:
            raise ValueError("residuals must be non-negative")
        if mean_res > max_res:
            raise ValueError("mean residual must not exceed max residual")
        if not isinstance(self.quality, CalibrationQuality):
            raise TypeError("quality must be a CalibrationQuality")
        object.__setattr__(self, "mean_reprojection_residual_px", mean_res)
        object.__setattr__(self, "max_reprojection_residual_px", max_res)

    @property
    def is_qualified(self) -> bool:
        """Return True if calibration meets qualification requirements."""
        return self.quality is CalibrationQuality.QUALIFIED


def evaluate_intrinsic_quality(
    mean_residual_px: float,
    views_count: int,
    sensor_coverage_ratio: float,
    floor_threshold_px: float = 1.0,
) -> tuple[CalibrationQuality, tuple[CalibrationDegeneracyKind, ...]]:
    """Evaluate qualification floor and identify potential geometric degeneracies."""
    mean_res = require_finite(mean_residual_px, "mean_residual_px")
    views = require_nonnegative_integer(views_count, "views_count")
    coverage = require_finite(sensor_coverage_ratio, "sensor_coverage_ratio")
    threshold = require_finite(floor_threshold_px, "floor_threshold_px")

    degeneracies: list[CalibrationDegeneracyKind] = []

    if views < 5:
        degeneracies.append(CalibrationDegeneracyKind.INSUFFICIENT_VIEWS)
    if coverage < 0.3:
        degeneracies.append(CalibrationDegeneracyKind.POOR_SENSOR_COVERAGE)
    if mean_res > 2.0 * threshold:
        degeneracies.append(CalibrationDegeneracyKind.HIGH_REPROJECTION_ERROR)

    if mean_res > 2.0 * threshold or views < 3 or coverage < 0.2:
        return (CalibrationQuality.UNQUALIFIED, tuple(degeneracies))

    if mean_res > threshold or views < 5 or coverage < 0.4:
        return (CalibrationQuality.DEGRADED, tuple(degeneracies))

    return (CalibrationQuality.QUALIFIED, tuple(degeneracies))


def check_coverage_and_degeneracy(
    points: Sequence[tuple[float, float]],
    resolution_px: tuple[int, int],
    min_views: int = 5,
    views_count: int = 1,
) -> tuple[float, tuple[CalibrationDegeneracyKind, ...]]:
    """Compute 2-D convex hull / bounding coverage ratio across sensor area."""
    if not points:
        return (
            0.0,
            (
                CalibrationDegeneracyKind.INSUFFICIENT_VIEWS,
                CalibrationDegeneracyKind.POOR_SENSOR_COVERAGE,
            ),
        )

    w, h = resolution_px
    total_area = float(w * h)
    min_u = min(p[0] for p in points)
    max_u = max(p[0] for p in points)
    min_v = min(p[1] for p in points)
    max_v = max(p[1] for p in points)

    bbox_area = max(0.0, max_u - min_u) * max(0.0, max_v - min_v)
    coverage = min(1.0, bbox_area / total_area)

    degeneracies: list[CalibrationDegeneracyKind] = []
    if views_count < min_views:
        degeneracies.append(CalibrationDegeneracyKind.INSUFFICIENT_VIEWS)
    if coverage < 0.3:
        degeneracies.append(CalibrationDegeneracyKind.POOR_SENSOR_COVERAGE)

    return (coverage, tuple(degeneracies))


__all__: list[str] = []
