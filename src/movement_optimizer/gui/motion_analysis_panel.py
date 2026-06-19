# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Matplotlib analysis panel for the swingset and chain tabs.

Mirrors the barbell exercise tabs' Figure + canvas + toolbar pattern, exposing a
named grid of axes that the tab populates with ``plot_renderer`` functions.

Design Principles:
    DBC -- the constructor validates its grid/axis arguments (ValueError).
    LoD -- the panel owns only its figure; the tab supplies the plot content.
    DRY -- reuses rendering.restyle_figure for theme-consistent styling.
"""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import (  # type: ignore[attr-defined]  # matplotlib stubs omit NavigationToolbar2QT
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..rendering import Palette, restyle_figure


class MotionAnalysisPanel(QWidget):
    """A themed grid of named matplotlib axes with a navigation toolbar."""

    _LEGEND_HEIGHT_RATIO = 0.32
    _DATA_LEGEND_HSPACE = 0.50
    _MIN_AXIS_WIDTH_PX = 260
    _MIN_DATA_HEIGHT_PX = 380
    _MIN_LEGEND_HEIGHT_PX = 76
    _MIN_TOOLBAR_HEIGHT_PX = 40

    def __init__(self, axis_names: Sequence[str], *, rows: int, cols: int) -> None:
        """Build the panel.

        Preconditions:
            ``axis_names`` is non-empty and ``rows * cols >= len(axis_names)``;
            ``rows`` and ``cols`` are positive.
        """
        super().__init__()
        if not axis_names:
            raise ValueError("axis_names must be non-empty")
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be positive")
        if rows * cols < len(axis_names):
            raise ValueError("rows * cols must be at least len(axis_names)")

        self._axis_names: tuple[str, ...] = tuple(axis_names)
        self._rows = rows
        self._cols = cols

        self.figure = Figure(figsize=(8.0, 5.0), facecolor=Palette.BG)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(self._minimum_canvas_width(), self._minimum_canvas_height())
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.axes: dict[str, Axes] = {}
        self.legend_axes: dict[str, Axes] = {}
        self._legends_visible = True
        self.setMinimumSize(
            self._minimum_canvas_width(),
            self._minimum_canvas_height() + self._MIN_TOOLBAR_HEIGHT_PX,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self._build_axes()

    def _build_axes(self) -> None:
        self.figure.clear()
        height_ratios = tuple(
            ratio for _ in range(self._rows) for ratio in (1.0, self._LEGEND_HEIGHT_RATIO)
        )
        grid = self.figure.add_gridspec(
            self._rows * 2,
            self._cols,
            height_ratios=height_ratios,
            hspace=self._DATA_LEGEND_HSPACE,
            wspace=0.34,
        )
        self.axes = {}
        self.legend_axes = {}
        for index, name in enumerate(self._axis_names):
            row, col = divmod(index, self._cols)
            data_row = row * 2
            legend_row = data_row + 1
            self.axes[name] = self.figure.add_subplot(grid[data_row, col])
            legend_axis = self.figure.add_subplot(grid[legend_row, col])
            legend_axis.set_axis_off()
            self.axes[name].set_zorder(2)
            legend_axis.set_zorder(1)
            self.legend_axes[name] = legend_axis
        restyle_figure(self.figure)

    def _minimum_canvas_width(self) -> int:
        """Return the minimum canvas width that keeps plot legends readable."""
        return self._cols * self._MIN_AXIS_WIDTH_PX

    def _minimum_canvas_height(self) -> int:
        """Return the minimum canvas height that keeps plot legends readable."""
        return self._rows * (self._MIN_DATA_HEIGHT_PX + self._MIN_LEGEND_HEIGHT_PX)

    def clear(self) -> None:
        """Reset every axis to a blank, themed state."""
        self._build_axes()

    @staticmethod
    def _legend_columns(label_count: int) -> int:
        """Return a compact column count for a dedicated legend strip."""
        if label_count < 1:
            return 1
        return min(3, label_count)

    def _dock_legends(self) -> None:
        """Move per-plot legends into reserved strips outside data axes."""
        for name, axes in self.axes.items():
            handles, labels = axes.get_legend_handles_labels()
            existing = axes.get_legend()
            if existing is not None:
                existing.remove()

            legend_axis = self.legend_axes[name]
            legend_axis.clear()
            legend_axis.set_axis_off()
            if not handles or not labels or not self._legends_visible:
                continue

            legend = legend_axis.legend(
                handles,
                labels,
                loc="center",
                bbox_to_anchor=(0.5, 0.5),
                ncol=self._legend_columns(len(labels)),
                fontsize=6,
                facecolor=Palette.BG_PANEL,
                edgecolor=Palette.FG_DIM,
                labelcolor=Palette.FG,
                framealpha=0.9,
                borderaxespad=0.0,
                handlelength=1.2,
                handletextpad=0.35,
                columnspacing=0.7,
            )
            legend.set_clip_on(True)
            legend.set_clip_box(legend_axis.bbox)

    def set_legends_visible(self, visible: bool) -> None:
        """Show or hide the legend on every axis that has one.

        Encapsulates legend management so callers need not reach into the
        figure's axes (Law of Demeter). Does not repaint; the caller draws.
        Axes without a legend are skipped.
        """
        self._legends_visible = bool(visible)
        for axes in self.axes.values():
            legend = axes.get_legend()
            if legend is not None:
                legend.set_visible(self._legends_visible)
        for axes in self.legend_axes.values():
            legend = axes.get_legend()
            if legend is not None:
                legend.set_visible(self._legends_visible)

    def has_legends(self) -> bool:
        """Return True if any axis currently carries a legend."""
        return any(axes.get_legend() is not None for axes in self.axes.values()) or any(
            axes.get_legend() is not None for axes in self.legend_axes.values()
        )

    def draw(self) -> None:
        """Lay out and repaint the figure after the axes have been populated."""
        self._enforce_minimum_figure_size()
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06)
        self._dock_legends()
        self.canvas.draw()

    def _enforce_minimum_figure_size(self) -> None:
        """Keep rendered figures at the panel's legend-safe minimum size."""
        dpi = float(self.figure.dpi)
        width_in, height_in = self.figure.get_size_inches()
        min_width_in = self._minimum_canvas_width() / dpi
        min_height_in = self._minimum_canvas_height() / dpi
        next_width_in = max(float(width_in), min_width_in)
        next_height_in = max(float(height_in), min_height_in)
        if next_width_in == float(width_in) and next_height_in == float(height_in):
            return
        self.figure.set_size_inches(next_width_in, next_height_in, forward=True)
