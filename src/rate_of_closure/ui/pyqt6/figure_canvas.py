"""Lifecycle-safe Matplotlib canvas for the Rate desktop UI."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import overload

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import sip
from PyQt6.QtCore import QRect, QTimer
from PyQt6.QtGui import QCloseEvent, QPaintEvent, QRegion

__all__ = ["LifecycleSafeFigureCanvas"]


class LifecycleSafeFigureCanvas(FigureCanvasQTAgg):
    """Own idle redraw work and tolerate destruction at the Qt update boundary."""

    def __init__(self, figure: Figure) -> None:
        super().__init__(figure)
        self._draw_pending: bool = False
        self._idle_draw_timer = QTimer(self)
        self._idle_draw_timer.setSingleShot(True)
        self._idle_draw_timer.timeout.connect(self._draw_idle)
        self._idle_draw_suppression_depth = 0
        self._idle_draws_paused = False

    def draw_idle(self) -> None:
        """Coalesce redraw requests on a timer owned by this canvas."""
        if (
            sip.isdeleted(self)
            or self._idle_draws_paused
            or self._idle_draw_suppression_depth
            or self._draw_pending
            or self._is_drawing
        ):
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

    @contextmanager
    def suppress_idle_draws(self) -> Iterator[None]:
        """Prevent backend callbacks from leaking work outside an atomic draw."""
        self._idle_draw_suppression_depth += 1
        self.cancel_pending_draw()
        try:
            yield
        finally:
            self.cancel_pending_draw()
            self._idle_draw_suppression_depth -= 1

    def pause_idle_draws(self) -> None:
        """Hold deferred redraws after an un-restorable publication failure."""
        self._idle_draws_paused = True
        self.cancel_pending_draw()

    def resume_idle_draws(self) -> None:
        """Allow deferred redraws after a successful synchronous retry."""
        self.cancel_pending_draw()
        self._idle_draws_paused = False

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Paint with the owning window pinned so it cannot be destroyed mid-paint.

        Matplotlib's paintEvent flushes a pending draw before it creates its
        QPainter. If cyclic garbage collection runs during that draw, it can
        free an unreferenced top-level window and delete this canvas inside its
        own paint, which segfaults Qt (#5905). Holding the window's wrapper
        keeps it reachable until the paint returns.
        """
        if sip.isdeleted(self):
            return
        owner = self.window()
        try:
            super().paintEvent(event)
        finally:
            del owner

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
