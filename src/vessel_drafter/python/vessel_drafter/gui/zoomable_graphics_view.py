"""Zoom-aware graphics view helpers for drafting previews."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import QGraphicsView, QWidget


class ZoomableGraphicsView(QGraphicsView):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        default_zoom_out_factor: float = 0.9,
    ) -> None:
        super().__init__(parent)
        self._default_zoom_out_factor = default_zoom_out_factor
        self._user_zoom_factor = 1.0
        self._minimum_zoom_factor = 0.4
        self._maximum_zoom_factor = 6.0

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHints(
            self.renderHints()
            | QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )

    @property
    def zoom_factor(self) -> float:
        return self._user_zoom_factor

    def zoom_in(self) -> None:
        self._set_zoom_factor(self._user_zoom_factor * 1.2)

    def zoom_out(self) -> None:
        self._set_zoom_factor(self._user_zoom_factor / 1.2)

    def reset_zoom(self) -> None:
        self._user_zoom_factor = 1.0
        self.sync_to_scene()

    def sync_to_scene(self) -> None:
        scene = self.scene()
        viewport = self.viewport()
        if (
            scene is None
            or scene.sceneRect().isNull()
            or viewport is None
            or viewport.rect().isEmpty()
        ):
            return

        self.resetTransform()
        self.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.scale_factor(self._effective_zoom_factor)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        if event.angleDelta().y() == 0:
            super().wheelEvent(event)
            return

        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.sync_to_scene()

    @property
    def _effective_zoom_factor(self) -> float:
        return self._default_zoom_out_factor * self._user_zoom_factor

    def _set_zoom_factor(self, zoom_factor: float) -> None:
        self._user_zoom_factor = min(
            max(zoom_factor, self._minimum_zoom_factor),
            self._maximum_zoom_factor,
        )
        self.sync_to_scene()

    def scale_factor(self, factor: float) -> None:
        self.scale(factor, factor)
