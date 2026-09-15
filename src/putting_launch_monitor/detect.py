"""Finding the ball in a frame.

One protocol, :class:`BallDetector`, so the tracker never knows how a ball
was found. :class:`HsvBallDetector` is the workhorse: threshold in HSV,
clean up, take the most circular blob of plausible size inside the region
of interest. Colour profiles are data, so a ball or a lighting change is a
setting, not code. Everything is pure on ``ndarray`` frames; OpenCV is
imported lazily so the module loads without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from shared.python.contracts import require

Frame: TypeAlias = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class BallObservation:
    """A ball seen at ``(cx, cy)`` pixels with ``radius_px`` and ``area_px``.

    Invariants: positive radius and area; ``circularity`` in [0, 1].
    """

    cx: float
    cy: float
    radius_px: float
    area_px: float
    circularity: float

    def __post_init__(self) -> None:
        require(self.radius_px > 0, "radius must be positive", self.radius_px)
        require(self.area_px > 0, "area must be positive", self.area_px)
        require(0.0 <= self.circularity <= 1.0, "circularity", self.circularity)

    @property
    def centre(self) -> tuple[float, float]:
        return (self.cx, self.cy)


@dataclass(frozen=True)
class HsvRange:
    """Inclusive HSV bounds in OpenCV's ranges (H 0..179, S and V 0..255)."""

    h_min: int = 0
    s_min: int = 0
    v_min: int = 180
    h_max: int = 179
    s_max: int = 60
    v_max: int = 255

    def __post_init__(self) -> None:
        require(
            0 <= self.h_min <= self.h_max <= 179, "hue bounds", (self.h_min, self.h_max)
        )
        require(
            0 <= self.s_min <= self.s_max <= 255, "sat bounds", (self.s_min, self.s_max)
        )
        require(
            0 <= self.v_min <= self.v_max <= 255, "val bounds", (self.v_min, self.v_max)
        )

    def lower(self) -> npt.NDArray[np.uint8]:
        return np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)

    def upper(self) -> npt.NDArray[np.uint8]:
        return np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)


# A white ball on a green mat: low saturation, high value, any hue.
WHITE_BALL = HsvRange(0, 0, 170, 179, 70, 255)
# The community's recommended orange ball for dim rooms.
ORANGE_BALL = HsvRange(3, 150, 120, 25, 255, 255)
YELLOW_BALL = HsvRange(20, 100, 120, 40, 255, 255)
COLOUR_PROFILES: dict[str, HsvRange] = {
    "white": WHITE_BALL,
    "orange": ORANGE_BALL,
    "yellow": YELLOW_BALL,
}


@dataclass(frozen=True)
class RegionOfInterest:
    """A pixel rectangle ``[x0, x1) x [y0, y1)`` where the ball may appear."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        require(self.x0 >= 0 and self.y0 >= 0, "roi origin", (self.x0, self.y0))
        require(self.x1 > self.x0 and self.y1 > self.y0, "roi extent", self)

    def clip(self, frame: Frame) -> tuple[Frame, int, int]:
        """The sub-image and its offset; the rectangle is clipped to the frame."""
        h, w = frame.shape[:2]
        x0, y0 = min(self.x0, w - 1), min(self.y0, h - 1)
        x1, y1 = min(self.x1, w), min(self.y1, h)
        return frame[y0:y1, x0:x1], x0, y0


class BallDetector(Protocol):
    """Anything that can say where balls are in a BGR frame.

    ``detect_all`` returns every plausible ball, best first, so a tracker can
    follow *its* ball when several rest on the mat; ``detect`` is the best one.
    """

    def detect_all(self, frame_bgr: Frame) -> list[BallObservation]: ...

    def detect(self, frame_bgr: Frame) -> BallObservation | None: ...


@dataclass(frozen=True)
class HsvBallDetector:
    """Threshold in HSV, then the most circular blob of plausible size.

    ``min_radius_px``/``max_radius_px`` bound the ball's apparent size (the
    calibration's local scale gives good values); ``min_circularity``
    rejects streaks and mat markings. Preconditions: positive radii with
    ``min < max``; circularity in [0, 1].
    """

    colour: HsvRange = WHITE_BALL
    min_radius_px: float = 4.0
    max_radius_px: float = 60.0
    min_circularity: float = 0.6
    roi: RegionOfInterest | None = None
    blur_px: int = 5

    def __post_init__(self) -> None:
        require(0 < self.min_radius_px < self.max_radius_px, "radius bounds", self)
        require(0.0 <= self.min_circularity <= 1.0, "circularity", self.min_circularity)
        require(
            self.blur_px >= 0 and self.blur_px % 2 == 1 or self.blur_px == 0,
            "blur odd",
            self.blur_px,
        )

    def mask(self, frame_bgr: Frame) -> npt.NDArray[np.uint8]:
        """The binary mask the blob search runs on (for tuning displays)."""
        cv2 = _cv2()
        image = frame_bgr
        if self.blur_px:
            image = cv2.GaussianBlur(image, (self.blur_px, self.blur_px), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.colour.lower(), self.colour.upper())
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return np.asarray(
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), dtype=np.uint8
        )

    def detect_all(self, frame_bgr: Frame) -> list[BallObservation]:
        """Every plausible ball in the search region, most circular first."""
        require(frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3, "BGR frame expected")
        cv2 = _cv2()
        sub, ox, oy = self.roi.clip(frame_bgr) if self.roi else (frame_bgr, 0, 0)
        contours, _ = cv2.findContours(
            self.mask(sub), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        found = [self._score(cv2, c, ox, oy) for c in contours]
        kept = [obs for obs in found if obs is not None]
        kept.sort(key=lambda obs: -obs.circularity)
        return kept

    def detect(self, frame_bgr: Frame) -> BallObservation | None:
        """The best ball candidate, or ``None`` when nothing plausible is seen."""
        found = self.detect_all(frame_bgr)
        return found[0] if found else None

    def _score(
        self, cv2: Any, contour: Any, ox: int, oy: int
    ) -> BallObservation | None:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            return None
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not (self.min_radius_px <= radius <= self.max_radius_px):
            return None
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            return None
        circularity = min(1.0, 4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < self.min_circularity:
            return None
        return BallObservation(
            cx=float(cx) + ox,
            cy=float(cy) + oy,
            radius_px=float(radius),
            area_px=area,
            circularity=float(circularity),
        )


def _cv2() -> Any:
    import cv2

    return cv2
