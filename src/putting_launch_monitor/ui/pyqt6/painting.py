"""Theme-sourced pens and image conversion for the live view and the wizard.

Every colour comes from :mod:`shared.python.theme` so the overlays follow
the active theme and no literal ever lands in GUI code (issue #3992). The
frame conversion shares the ndarray's memory where Qt allows it, since a
60 fps preview cannot afford copies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen

from putting_launch_monitor.detect import BallObservation, Frame, RegionOfInterest
from shared.python.contracts import require
from shared.python.theme import Colors, get_qcolor
from shared.python.theme.zoom import ZoomTokenSet

_QIMAGE_BGR = QImage.Format.Format_BGR888
_QIMAGE_GRAY = QImage.Format.Format_Grayscale8
_SOLID = Qt.PenStyle.SolidLine
_DASHED = Qt.PenStyle.DashLine
_NO_BRUSH = Qt.BrushStyle.NoBrush

#: Layout tokens at 100 % zoom (padding 8 px, spacing 6 px), from the theme.
TOKENS = ZoomTokenSet.from_percent(100)


@dataclass(frozen=True)
class OverlayPalette:
    """The colours the overlays use, resolved once from the active theme."""

    search_region: QColor
    candidate: QColor
    tracked: QColor
    corner: QColor
    text: QColor

    @classmethod
    def from_theme(cls) -> OverlayPalette:
        return cls(
            search_region=get_qcolor(Colors.INFO),
            candidate=get_qcolor(Colors.WARNING),
            tracked=get_qcolor(Colors.SUCCESS),
            corner=get_qcolor(Colors.ERROR),
            text=get_qcolor(Colors.TEXT_PRIMARY),
        )


def frame_to_qimage(frame: Frame) -> QImage:
    """A BGR ``(h, w, 3)`` frame as a QImage that owns a copy of the pixels.

    Precondition: three-channel uint8. The copy makes the image safe to keep
    after the worker thread has moved on to the next frame.
    """
    require(frame.ndim == 3 and frame.shape[2] == 3, "BGR frame", frame.shape)
    h, w = frame.shape[:2]
    contiguous = np.ascontiguousarray(frame)
    image = QImage(contiguous.data, w, h, 3 * w, _QIMAGE_BGR)
    return image.copy()


def mask_to_qimage(mask: np.ndarray) -> QImage:
    """A single-channel uint8 mask as a greyscale QImage (copied)."""
    require(mask.ndim == 2, "2-D mask", mask.shape)
    h, w = mask.shape
    contiguous = np.ascontiguousarray(mask)
    return QImage(contiguous.data, w, h, w, _QIMAGE_GRAY).copy()


def fit_scale(image_w: int, image_h: int, box_w: int, box_h: int) -> float:
    """The uniform scale that fits an image inside a box (never upscales past 1:1)."""
    require(image_w > 0 and image_h > 0, "image size", (image_w, image_h))
    if box_w <= 0 or box_h <= 0:
        return 1.0
    return min(box_w / image_w, box_h / image_h, 1.0)


def draw_region(painter: QPainter, region: RegionOfInterest, colour: QColor) -> None:
    pen = QPen(colour)
    pen.setStyle(_DASHED)
    pen.setWidth(2)
    painter.setPen(pen)
    painter.setBrush(_NO_BRUSH)
    painter.drawRect(
        QRectF(region.x0, region.y0, region.x1 - region.x0, region.y1 - region.y0)
    )


def draw_candidates(
    painter: QPainter,
    candidates: Sequence[BallObservation],
    tracked: BallObservation | None,
    palette: OverlayPalette,
) -> None:
    """Circle every candidate; the tracked one heavier and in its own colour."""
    painter.setBrush(_NO_BRUSH)
    for obs in candidates:
        is_tracked = tracked is not None and obs == tracked
        pen = QPen(palette.tracked if is_tracked else palette.candidate)
        pen.setStyle(_SOLID)
        pen.setWidth(3 if is_tracked else 1)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(obs.cx, obs.cy), obs.radius_px, obs.radius_px)


def draw_corners(
    painter: QPainter,
    corners: Sequence[tuple[float, float]],
    palette: OverlayPalette,
    radius: float = 6.0,
) -> None:
    """Numbered corner marks joined in order (the last closes the polygon)."""
    pen = QPen(palette.corner)
    pen.setWidth(2)
    painter.setPen(pen)
    painter.setBrush(_NO_BRUSH)
    points = [QPointF(x, y) for x, y in corners]
    for i, p in enumerate(points):
        painter.drawEllipse(p, radius, radius)
        painter.drawText(QPointF(p.x() + radius + 2, p.y() - 2), str(i + 1))
    if len(points) >= 2:
        closed = points + ([points[0]] if len(points) == 4 else [])
        for a, b in zip(closed, closed[1:], strict=False):
            painter.drawLine(a, b)
