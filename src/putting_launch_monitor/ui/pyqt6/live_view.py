"""The camera frame with the search region and detections drawn over it.

:class:`LiveView` keeps the latest frame and overlay state and repaints on
demand; it is fed by :meth:`show_event` from the GUI thread. The same
canvas serves the calibration wizard as a clickable image
(:class:`CornerCanvas`), mapping widget clicks back to frame pixels
through the fitted scale.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPainter, QPaintEvent
from PyQt6.QtWidgets import QSizePolicy, QWidget

from putting_launch_monitor.detect import BallObservation, Frame, RegionOfInterest
from putting_launch_monitor.monitor import FrameEvent
from shared.python.contracts import require

from .painting import (
    OverlayPalette,
    draw_candidates,
    draw_corners,
    draw_region,
    fit_scale,
    frame_to_qimage,
)

_EXPANDING = QSizePolicy.Policy.Expanding
_LEFT_BUTTON = Qt.MouseButton.LeftButton
_SMOOTH = QPainter.RenderHint.SmoothPixmapTransform
_ANTIALIAS = QPainter.RenderHint.Antialiasing
_MIN_CANVAS_PX = 240


class ImageCanvas(QWidget):
    """A QImage scaled to fit, with an overlay hook in image coordinates.

    ``clicked`` reports left clicks in *image* pixels; clicks outside the
    image, or before one is set, are ignored.
    """

    clicked = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._overlay: Callable[[QPainter], None] | None = None
        self.setSizePolicy(_EXPANDING, _EXPANDING)
        self.setMinimumSize(_MIN_CANVAS_PX, _MIN_CANVAS_PX * 5 // 8)

    # -- state ----------------------------------------------------------------------
    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_frame(self, frame: Frame | None) -> None:
        self.set_image(frame_to_qimage(frame) if frame is not None else None)

    def set_overlay(self, overlay: Callable[[QPainter], None] | None) -> None:
        """A callable drawing in image pixel coordinates after the image."""
        self._overlay = overlay
        self.update()

    @property
    def image_size(self) -> tuple[int, int] | None:
        if self._image is None:
            return None
        return self._image.width(), self._image.height()

    def scale(self) -> float | None:
        size = self.image_size
        if size is None:
            return None
        return fit_scale(size[0], size[1], self.width(), self.height())

    def to_image(self, x: float, y: float) -> tuple[float, float] | None:
        """Widget coordinates -> image pixels, or ``None`` outside the image."""
        s, size = self.scale(), self.image_size
        if s is None or size is None:
            return None
        ix, iy = x / s, y / s
        if 0 <= ix < size[0] and 0 <= iy < size[1]:
            return ix, iy
        return None

    # -- Qt -------------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(_SMOOTH)
        painter.setRenderHint(_ANTIALIAS)
        s = self.scale()
        if self._image is None or s is None:
            painter.end()
            return
        painter.scale(s, s)
        painter.drawImage(QPointF(0.0, 0.0), self._image)
        if self._overlay is not None:
            self._overlay(painter)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is None or event.button() != _LEFT_BUTTON:
            return
        pos = event.position()
        hit = self.to_image(pos.x(), pos.y())
        if hit is not None:
            self.clicked.emit(*hit)

    def click_at_image(self, x: float, y: float) -> None:
        """Deliver a click in image pixels (for tests and scripted use)."""
        require(x >= 0 and y >= 0, "image point", (x, y))
        self.clicked.emit(float(x), float(y))


class LiveView(ImageCanvas):
    """The running monitor's frame, search region and candidates."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.palette_ = OverlayPalette.from_theme()
        self.region: RegionOfInterest | None = None
        self.candidates: tuple[BallObservation, ...] = ()
        self.tracked: BallObservation | None = None
        self.set_overlay(self._draw)
        self.setToolTip(
            "Live camera frame at the decode width. Dashed box: the search "
            "region; thin circles: ball candidates; heavy circle: the ball "
            "the monitor is following."
        )

    def set_region(self, region: RegionOfInterest | None) -> None:
        self.region = region
        self.update()

    def show_event(self, event: FrameEvent) -> None:
        """Adopt one :class:`FrameEvent` (frame, candidates, tracked ball)."""
        self.candidates = event.candidates
        self.tracked = event.seen
        if event.frame is not None:
            self.set_frame(event.frame)
        else:
            self.update()

    def _draw(self, painter: QPainter) -> None:
        if self.region is not None:
            draw_region(painter, self.region, self.palette_.search_region)
        draw_candidates(painter, self.candidates, self.tracked, self.palette_)


class CornerCanvas(ImageCanvas):
    """A snapshot the operator clicks four mat corners on, in order."""

    corners_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.palette_ = OverlayPalette.from_theme()
        self.corners: list[tuple[float, float]] = []
        self.region: RegionOfInterest | None = None
        self.set_overlay(self._draw)
        self.clicked.connect(self.add_corner)
        self.setToolTip(
            "Click the four mat corners in order: 1 near-left, 2 near-right, "
            "3 far-right, 4 far-left, as the player sees them."
        )

    def add_corner(self, x: float, y: float) -> None:
        """Record a corner; a fifth click starts over. Postcondition: <= 4 corners."""
        if len(self.corners) >= 4:
            self.corners.clear()
        self.corners.append((float(x), float(y)))
        self.corners_changed.emit()
        self.update()

    def clear_corners(self) -> None:
        self.corners.clear()
        self.region = None
        self.corners_changed.emit()
        self.update()

    def set_region(self, region: RegionOfInterest | None) -> None:
        self.region = region
        self.update()

    def _draw(self, painter: QPainter) -> None:
        if self.region is not None:
            draw_region(painter, self.region, self.palette_.search_region)
        draw_corners(painter, self.corners, self.palette_)
