"""Ground-plane geometry for the camera putting monitor.

Everything a putt needs geometrically, with no camera or GUI in sight:

* :class:`GroundPlane` — the homography between image pixels and the ground
  in millimetres, calibrated from the hitting mat's four corners (a rectangle
  of known size) or from any four-or-more point correspondences.
* :class:`WorldFrame` convention — ``x`` across the green, positive to the
  player's RIGHT; ``y`` along the green, positive TOWARD the target. Corners
  are labelled by their physical position from the player's point of view,
  so a camera facing the player (a mirrored view) needs no flip flag: the
  convention absorbs it.
* :func:`fit_launch` — launch speed and direction from a short window of
  world-space positions right after the ball starts moving: a straight-line
  fit over many frames rather than two gate crossings, so one bad detection
  cannot decide the shot.
* :func:`hla_degrees` — GSPro's horizontal launch angle: signed degrees
  between the launch direction and the target line, positive to the right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from shared.python.contracts import require

FloatArray: TypeAlias = npt.NDArray[np.float64]

MM_PER_M = 1000.0
MPH_PER_MPS = 2.2369362920544
GOLF_BALL_DIAMETER_MM = 42.67  # USGA minimum


@dataclass(frozen=True)
class GroundPlane:
    """Image (px) <-> ground (mm) homography.

    Invariant: ``h`` is a 3x3 non-singular matrix mapping homogeneous image
    coordinates to homogeneous ground coordinates in millimetres.
    """

    h: FloatArray

    def __post_init__(self) -> None:
        require(self.h.shape == (3, 3), "homography must be 3x3", self.h.shape)
        require(bool(np.isfinite(self.h).all()), "homography must be finite")
        require(bool(abs(np.linalg.det(self.h)) > 1e-12), "homography is singular")

    @classmethod
    def from_correspondences(
        cls, image_px: FloatArray, world_mm: FloatArray
    ) -> GroundPlane:
        """Least-squares homography from >= 4 matched points (DLT).

        Precondition: at least four correspondences, not all collinear.
        Postcondition: the returned plane reprojects the inputs; callers
        should check :meth:`reprojection_error_px` against their tolerance.
        """
        image_px = np.asarray(image_px, dtype=np.float64)
        world_mm = np.asarray(world_mm, dtype=np.float64)
        require(image_px.shape == world_mm.shape, "point sets must match")
        require(image_px.ndim == 2 and image_px.shape[1] == 2, "points are Nx2")
        require(len(image_px) >= 4, "need at least 4 correspondences", len(image_px))
        h = _dlt(image_px, world_mm)
        return cls(h)

    @classmethod
    def from_rectangle(
        cls,
        corners_px: FloatArray,
        width_mm: float,
        length_mm: float,
    ) -> GroundPlane:
        """From a rectangle of known size seen in the image.

        ``corners_px`` are the four corners in this order, judged from where
        the PLAYER stands looking toward the target:
        near-left, near-right, far-right, far-left. The world frame puts the
        origin at near-left, ``x`` toward near-right (player's right) and
        ``y`` toward far-left (toward the target).
        Precondition: positive dimensions; four distinct corners.
        """
        require(
            width_mm > 0 and length_mm > 0,
            "rectangle dimensions",
            (width_mm, length_mm),
        )
        corners_px = np.asarray(corners_px, dtype=np.float64)
        require(corners_px.shape == (4, 2), "four corners", corners_px.shape)
        world = np.array(
            [[0.0, 0.0], [width_mm, 0.0], [width_mm, length_mm], [0.0, length_mm]]
        )
        return cls.from_correspondences(corners_px, world)

    def to_world(self, points_px: FloatArray) -> FloatArray:
        """Ground coordinates (mm) of image points (Nx2 or 2)."""
        return _apply(self.h, np.asarray(points_px, dtype=np.float64))

    def to_image(self, points_mm: FloatArray) -> FloatArray:
        inverse = np.asarray(np.linalg.inv(self.h), dtype=np.float64)
        return _apply(inverse, np.asarray(points_mm, dtype=np.float64))

    def reprojection_error_px(
        self, image_px: FloatArray, world_mm: FloatArray
    ) -> float:
        """RMS pixel error mapping ``world_mm`` back into the image."""
        back = self.to_image(np.asarray(world_mm, dtype=np.float64))
        diff = back - np.asarray(image_px, dtype=np.float64)
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

    def mm_per_px_at(self, point_px: FloatArray) -> float:
        """Local ground scale at an image point (for radius sanity checks)."""
        p = np.asarray(point_px, dtype=np.float64)
        a = self.to_world(p)
        b = self.to_world(p + np.array([1.0, 0.0]))
        c = self.to_world(p + np.array([0.0, 1.0]))
        return float((np.linalg.norm(b - a) + np.linalg.norm(c - a)) / 2.0)


def _apply(h: FloatArray, pts: FloatArray) -> FloatArray:
    single = pts.ndim == 1
    p = np.atleast_2d(pts)
    hom = np.hstack([p, np.ones((len(p), 1))])
    out = hom @ h.T
    w = out[:, 2:3]
    require(bool((np.abs(w) > 1e-12).all()), "point maps to infinity")
    res = out[:, :2] / w
    return res[0] if single else res


def _dlt(src: FloatArray, dst: FloatArray) -> FloatArray:
    """Normalised direct linear transform (Hartley) for a 2-D homography."""

    def normalise(p: FloatArray) -> tuple[FloatArray, FloatArray]:
        mean = p.mean(axis=0)
        d = np.sqrt(((p - mean) ** 2).sum(axis=1)).mean()
        require(bool(d > 1e-9), "degenerate (coincident) points")
        s = np.sqrt(2.0) / d
        t = np.array([[s, 0, -s * mean[0]], [0, s, -s * mean[1]], [0, 0, 1.0]])
        q = (np.hstack([p, np.ones((len(p), 1))]) @ t.T)[:, :2]
        return q, t

    s_n, t_s = normalise(src)
    d_n, t_d = normalise(dst)
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(s_n, d_n, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    a = np.asarray(rows, dtype=np.float64)
    _, sv, vt = np.linalg.svd(a)
    require(bool(sv[-2] > 1e-9 * sv[0]), "correspondences are degenerate (collinear?)")
    h_n = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_d) @ h_n @ t_s
    return np.asarray(h / h[2, 2], dtype=np.float64)


# -- launch fit -----------------------------------------------------------------------


@dataclass(frozen=True)
class Launch:
    """The putt as GSPro needs it, plus how much to trust it.

    ``speed_mps`` along ``direction`` (unit vector in the world frame),
    fitted over ``points`` observations spanning ``span_mm`` of travel;
    ``r2`` is the coefficient of determination of distance-vs-time.
    """

    speed_mps: float
    direction: tuple[float, float]
    points: int
    span_mm: float
    r2: float

    @property
    def speed_mph(self) -> float:
        return self.speed_mps * MPH_PER_MPS


def fit_launch(
    times_s: FloatArray,
    positions_mm: FloatArray,
    *,
    window_mm: float = 300.0,
    min_points: int = 4,
) -> Launch:
    """Launch speed and direction from the first ``window_mm`` of travel.

    The direction is the principal axis of the positions, oriented from the
    first point toward the last; speed is the slope of a least-squares line
    through (time, distance-along-direction). Using the opening window keeps
    this the *launch* speed — the ball is already slowing on the mat further
    out, and GSPro simulates the roll itself.

    Preconditions: at least ``min_points`` samples, strictly increasing
    times, positive window. Postconditions: ``speed_mps >= 0``,
    ``direction`` is unit length, ``0 <= r2 <= 1``.
    """
    t = np.asarray(times_s, dtype=np.float64)
    p = np.asarray(positions_mm, dtype=np.float64)
    require(window_mm > 0, "window must be positive", window_mm)
    require(min_points >= 3, "need at least three points to fit", min_points)
    require(t.ndim == 1 and p.shape == (len(t), 2), "times N, positions Nx2")
    require(len(t) >= min_points, "too few samples", len(t))
    require(bool((np.diff(t) > 0).all()), "times must strictly increase")
    travelled = np.linalg.norm(p - p[0], axis=1)
    keep = travelled <= window_mm
    keep[: min(min_points, len(t))] = True  # never below the minimum
    t, p = t[keep], p[keep]
    centred = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    if np.dot(p[-1] - p[0], axis) < 0:
        axis = -axis
    along = centred @ axis
    tt = t - t.mean()
    denom = float(np.dot(tt, tt))
    require(denom > 0, "times have no spread")
    slope = float(np.dot(tt, along) / denom)
    pred = slope * tt
    ss_res = float(np.sum((along - along.mean() - pred) ** 2))
    ss_tot = float(np.sum((along - along.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    speed = max(0.0, slope) / MM_PER_M
    return Launch(
        speed_mps=speed,
        direction=(float(axis[0]), float(axis[1])),
        points=int(len(t)),
        span_mm=float(np.linalg.norm(p[-1] - p[0])),
        r2=float(min(1.0, max(0.0, r2))),
    )


def hla_degrees(
    direction: tuple[float, float], target: tuple[float, float] = (0.0, 1.0)
) -> float:
    """GSPro horizontal launch angle: signed degrees, positive to the RIGHT.

    ``direction`` and ``target`` are vectors in the world frame (x to the
    player's right, y toward the target). Precondition: both non-zero.
    Postcondition: result in (-180, 180].
    """
    d = np.asarray(direction, dtype=np.float64)
    g = np.asarray(target, dtype=np.float64)
    require(bool(np.linalg.norm(d) > 0 and np.linalg.norm(g) > 0), "zero-length vector")
    # angle from target to direction; in an (x right, y forward) frame a
    # clockwise turn (toward +x) must read positive, hence the sign.
    cross = g[0] * d[1] - g[1] * d[0]
    dot = float(np.dot(g, d))
    return float(np.degrees(np.arctan2(-cross, dot)))
