"""Independent Matplotlib viewport for one managed plot."""

from __future__ import annotations

from collections.abc import Callable
from textwrap import fill

from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.plotting import PlotData, render_plot
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)

_ZOOM_STEP = 1.25
_TITLE_WRAP_CHARS = 42


class PlotCanvasPane(QFrame):
    """One plot with an independent canvas, transform, and legend policy."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAccessibleName(f"{label} Plot Viewport")
        self._data: PlotData | None = None
        self._custom_renderer: Callable[[Figure], None] | None = None
        self._zoom = 1.0
        self._figure = Figure(figsize=(5.2, 3.5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumSize(380, 260)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        controls = QHBoxLayout()
        self._title = QLabel(label)
        self._title.setStyleSheet("font-weight: 600")
        controls.addWidget(self._title, stretch=1)
        self._zoom_out = self._button(
            "Zoom Out", "Show a wider range around the fitted data.", self.zoom_out
        )
        self._zoom_in = self._button(
            "Zoom In", "Magnify around the fitted data center.", self.zoom_in
        )
        self._auto_fit = self._button(
            "Auto Fit",
            "Recompute readable limits from all visible data.",
            self.auto_fit,
        )
        controls.addWidget(self._zoom_out)
        controls.addWidget(self._zoom_in)
        controls.addWidget(self._auto_fit)
        self._zoom_label = QLabel("100%")
        self._zoom_label.setMinimumWidth(38)
        controls.addWidget(self._zoom_label)
        self._legend = QComboBox()
        self._legend.setAccessibleName("Legend Position")
        self._legend.setToolTip("Move the legend outside, place it inside, or hide it.")
        self._legend.addItem("Legend: Outside Right", "outside_right")
        self._legend.addItem("Legend: Inside Upper Right", "inside_upper_right")
        self._legend.addItem("Legend: Inside Lower Right", "inside_lower_right")
        self._legend.addItem("Legend: Inside Lower Left", "inside_lower_left")
        self._legend.addItem("Legend: Hidden", "hidden")
        self._legend.currentIndexChanged.connect(self._apply_legend)
        controls.addWidget(self._legend)
        layout.addLayout(controls)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

    @staticmethod
    def _button(text: str, tooltip: str, handler: Callable[[], None]) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.clicked.connect(handler)
        return button

    def render_data(self, data: PlotData) -> None:
        """Render new data and reset the viewport to a readable fit."""
        self._data = data
        self._custom_renderer = None
        self._render_and_fit()

    def render_custom(self, renderer: Callable[[Figure], None]) -> None:
        """Render caller-owned plot data with the same managed controls."""
        if not callable(renderer):
            raise TypeError("renderer must be callable")
        self._data = None
        self._custom_renderer = renderer
        self._render_and_fit()

    def clear(self) -> None:
        """Remove current data so Auto Fit cannot resurrect a stale result."""
        self._data = None
        self._custom_renderer = None
        self._figure.clear()
        self._toolbar.update()
        self._zoom = 1.0
        self._zoom_label.setText("100%")
        self._canvas.draw_idle()

    def _render_and_fit(self) -> None:
        if self._data is not None:
            render_plot(self._data, self._figure)
        elif self._custom_renderer is not None:
            self._custom_renderer(self._figure)
        else:
            return
        self._toolbar.update()
        self._wrap_canvas_titles()
        self._zoom = 1.0
        self._zoom_label.setText("100%")
        self._apply_legend()
        self._canvas.draw_idle()

    def _wrap_canvas_titles(self) -> None:
        """Keep long scientific titles inside a compact application pane."""
        for axes in self._figure.axes:
            title = axes.get_title()
            if len(title) > _TITLE_WRAP_CHARS:
                axes.set_title(fill(title, width=_TITLE_WRAP_CHARS))

    def canvas(self) -> FigureCanvas:
        """Return this pane's distinct canvas."""
        return self._canvas

    def figure(self) -> Figure:
        """Return this pane's distinct figure."""
        return self._figure

    def toolbar(self) -> NavigationToolbar2QT:
        """Return the native pan/zoom/save toolbar."""
        return self._toolbar

    def zoom_percent(self) -> int:
        """Return the explicit zoom as a percentage of fitted bounds."""
        return round(self._zoom * 100)

    def legend_placement(self) -> str:
        """Return the stable legend-placement key."""
        return str(self._legend.currentData())

    def set_legend_placement(self, placement: str) -> None:
        """Select a supported legend placement."""
        index = self._legend.findData(placement)
        if index < 0:
            raise ValueError(f"unsupported legend placement: {placement}")
        self._legend.setCurrentIndex(index)

    def zoom_in(self) -> None:
        """Magnify both axes by one deterministic step."""
        self._scale_limits(_ZOOM_STEP)

    def zoom_out(self) -> None:
        """Widen both axes by one deterministic step."""
        self._scale_limits(1.0 / _ZOOM_STEP)

    def auto_fit(self) -> None:
        """Restore data-derived axis limits and a 5% margin."""
        self._render_and_fit()

    def _scale_limits(self, factor: float) -> None:
        if not self._figure.axes:
            return
        for axes in self._figure.axes:
            x0, x1 = axes.get_xlim()
            y0, y1 = axes.get_ylim()
            x_mid, y_mid = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            x_half = (x1 - x0) / (2.0 * factor)
            y_half = (y1 - y0) / (2.0 * factor)
            axes.set_xlim(x_mid - x_half, x_mid + x_half)
            axes.set_ylim(y_mid - y_half, y_mid + y_half)
        self._zoom = min(20.0, max(0.2, self._zoom * factor))
        self._zoom_label.setText(f"{self.zoom_percent()}%")
        self._canvas.draw_idle()

    def _apply_legend(self, _index: int = 0) -> None:
        placement = self.legend_placement()
        for axes in self._figure.axes:
            legend = axes.get_legend()
            if legend is None:
                continue
            if placement == "hidden":
                legend.set_visible(False)
                continue
            legend.set_visible(True)
            if placement == "outside_right":
                legend.set_loc("upper left")
                legend.set_bbox_to_anchor((1.02, 1.0))
            else:
                legend.set_bbox_to_anchor(None)
                legend.set_loc(
                    {
                        "inside_upper_right": "upper right",
                        "inside_lower_right": "lower right",
                        "inside_lower_left": "lower left",
                    }[placement]
                )
        self._canvas.draw_idle()

    def _on_scroll(self, event: MouseEvent) -> None:
        if event.button == "up":
            self.zoom_in()
        elif event.button == "down":
            self.zoom_out()


__all__ = ["PlotCanvasPane"]
