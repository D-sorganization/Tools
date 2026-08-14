"""Plotting compatibility helpers for the P1AM desktop UI."""

from __future__ import annotations

from typing import Any

try:
    import pyqtgraph as pg
    import pyqtgraph.exporters as _pg_exporters
except ImportError:
    from PyQt6.QtWidgets import QWidget

    _pg_exporters = None

    class _FallbackCurve:
        def setData(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: N802
            return None

    class _FallbackPlotWidget(QWidget):
        """Minimal QWidget-backed plot used when pyqtgraph is not installed."""

        plotItem = None

        def setBackground(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: N802
            return None

        def showGrid(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: N802
            return None

        def setLabel(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: N802
            return None

        def addLegend(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: N802
            return None

        def setTitle(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: N802
            return None

        def autoRange(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: N802
            return None

        def setXRange(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: N802
            return None

        def plot(self, *_args: Any, **_kwargs: Any) -> _FallbackCurve:
            return _FallbackCurve()

    class _FallbackPyQtGraph:
        PlotWidget = _FallbackPlotWidget

        @staticmethod
        def mkPen(
            *args: Any, **kwargs: Any
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:  # noqa: N802
            return args, kwargs

    pg = _FallbackPyQtGraph()


def build_svg_exporter(plot_widget: Any) -> Any | None:
    """Return a pyqtgraph SVG exporter when the optional dependency exists."""
    if _pg_exporters is None:
        return None
    return _pg_exporters.SVGExporter(plot_widget.plotItem)


__all__ = ["build_svg_exporter", "pg"]
