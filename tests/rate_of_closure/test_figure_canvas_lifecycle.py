"""Regression tests for Rate's Qt/Matplotlib canvas ownership."""

from __future__ import annotations

import pytest
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from PyQt6.QtCore import QCoreApplication, QEvent

from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class _FailingArtist(Artist):
    def draw(self, renderer) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError("intentional renderer failure")


def test_deleted_canvas_during_idle_draw_has_no_qt_traceback(qapp, capsys) -> None:  # type: ignore[no-untyped-def]
    canvas = LifecycleSafeFigureCanvas(Figure())

    def delete_during_draw(_event) -> None:  # type: ignore[no-untyped-def]
        canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    canvas.mpl_connect("draw_event", delete_during_draw)
    canvas.draw_idle()
    QCoreApplication.processEvents()

    assert "wrapped C/C++ object" not in capsys.readouterr().err


def test_cancel_pending_draw_is_idempotent(qtbot) -> None:  # type: ignore[no-untyped-def]
    canvas = LifecycleSafeFigureCanvas(Figure())
    qtbot.addWidget(canvas)
    canvas.draw_idle()

    canvas.cancel_pending_draw()
    canvas.cancel_pending_draw()

    assert not canvas.has_pending_draw()


def test_unrelated_renderer_runtime_error_propagates(qtbot) -> None:  # type: ignore[no-untyped-def]
    figure = Figure()
    figure.add_artist(_FailingArtist())
    canvas = LifecycleSafeFigureCanvas(figure)
    qtbot.addWidget(canvas)

    with pytest.raises(RuntimeError, match="intentional renderer failure"):
        canvas.draw()
