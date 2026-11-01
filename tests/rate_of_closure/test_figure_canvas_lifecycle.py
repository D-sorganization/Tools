"""Regression tests for Rate's Qt/Matplotlib canvas ownership."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from PyQt6.QtCore import QCoreApplication, QEvent

from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"


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


# Runs in a child interpreter: on regression Qt aborts the process, which would
# take down the whole xdist worker instead of failing one test (#5905).
_GC_DURING_PAINT_SCRIPT = textwrap.dedent(
    """
    import gc
    from matplotlib.figure import Figure
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget
    from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas

    app = QApplication([])

    def build_unreferenced_window():
        window = QWidget()
        canvas = LifecycleSafeFigureCanvas(Figure())
        QVBoxLayout(window).addWidget(canvas)
        window.cycle = window
        window.resize(200, 150)
        window.show()
        app.processEvents()
        canvas.draw()
        canvas.mpl_connect("draw_event", lambda _event: gc.collect())
        canvas._draw_pending = True
        return canvas

    gc.disable()
    build_unreferenced_window().repaint()
    """
)


def test_gc_during_paint_flush_does_not_delete_canvas_mid_paint() -> None:
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": os.pathsep.join(
            [str(_SRC_DIR), os.environ.get("PYTHONPATH", "")]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", _GC_DURING_PAINT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )

    assert "wrapped C/C++ object" not in result.stderr
    assert result.returncode == 0, result.stderr
