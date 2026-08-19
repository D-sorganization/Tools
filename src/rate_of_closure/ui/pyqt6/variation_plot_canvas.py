"""Small lifecycle-safe Matplotlib canvas for variation result views."""

from __future__ import annotations

from matplotlib.figure import Figure

from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas


class VariationPlotCanvas(LifecycleSafeFigureCanvas):
    """Canvas exposing one themed axes to tests and visualization widgets."""

    def __init__(self, *, projection: str | None = None) -> None:
        figure = Figure(figsize=(6.0, 4.5), layout="constrained")
        super().__init__(figure)
        self.axes = figure.add_subplot(111, projection=projection)

    def apply_theme(self) -> None:
        """Apply the current Qt palette to figure, axes, and labels."""
        window = self.palette().window().color().name()
        text = self.palette().text().color().name()
        self.figure.set_facecolor(window)
        self.axes.set_facecolor(self.palette().window().color().lighter(105).name())
        self.axes.tick_params(colors=text, labelsize=8)
        axes = [self.axes.xaxis, self.axes.yaxis]
        if hasattr(self.axes, "zaxis"):
            axes.append(self.axes.zaxis)
        for axis in axes:
            axis.label.set_color(text)
        self.axes.title.set_color(text)


__all__ = ["VariationPlotCanvas"]
