"""Lifecycle-safe Matplotlib canvas for the Rate desktop UI."""

from __future__ import annotations

from typing import overload

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import sip
from PyQt6.QtCore import QRect, QTimer
from PyQt6.QtGui import QCloseEvent, QRegion

__all__ = ["LifecycleSafeFigureCanvas"]


class LifecycleSafeFigureCanvas(FigureCanvasQTAgg):
    """Own idle redraw work and tolerate destruction at the Qt update boundary."""

    def __init__(self, figure: Figure) -> None:
        super().__init__(figure)
        self._idle_draw_timer = QTimer(self)
        self._idle_draw_timer.setSingleShot(True)
        self._idle_draw_timer.timeout.connect(self._draw_idle)

    def draw_idle(self) -> None:
        """Coalesce redraw requests on a timer owned by this canvas."""
        if sip.isdeleted(self) or self._draw_pending or self._is_drawing:
            return
        self._draw_pending = True
        self._idle_draw_timer.start(0)

    def cancel_pending_draw(self) -> None:
        """Cancel queued idle work; safe to call repeatedly during teardown."""
        if not sip.isdeleted(self._idle_draw_timer):
            self._idle_draw_timer.stop()
        self._draw_pending = False

    def has_pending_draw(self) -> bool:
        """Return whether this canvas owns a queued idle redraw."""
        return self._draw_pending

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Cancel queued drawing before Qt starts destroying the canvas."""
        self.cancel_pending_draw()
        super().closeEvent(event)

    @overload
    def update(self) -> None: ...

    @overload
    def update(self, rect: QRect) -> None: ...

    @overload
    def update(self, region: QRegion) -> None: ...

    @overload
    def update(self, x: int, y: int, width: int, height: int) -> None: ...

    def update(self, *args: object, **kwargs: object) -> None:
        """Queue repaint unless a draw callback already destroyed the widget."""
        if sip.isdeleted(self):
            return
        super().update(*args, **kwargs)
