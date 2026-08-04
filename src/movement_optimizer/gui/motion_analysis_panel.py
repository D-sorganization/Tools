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
from typing import Any

from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import (  # type: ignore[attr-defined]  # matplotlib stubs omit NavigationToolbar2QT
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from ..rendering import Palette, restyle_figure


class MotionAnalysisPanel(QWidget):
    """A themed grid of named matplotlib axes with a navigation toolbar."""

    _LEGEND_ROW_PX = 58
    _GRID_WSPACE = 0.28
    _MIN_AXIS_WIDTH_PX = 440
    _MIN_DATA_HEIGHT_PX = 340
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
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.axes: dict[str, Axes] = {}
        self.legend_axes: dict[str, Axes] = {}
        self._figure_legend: Legend | None = None
        self._legends_visible = True
        self.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
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
        row_ratios = [
            ratio
            for _row in range(self._rows)
            for ratio in (self._MIN_DATA_HEIGHT_PX, self._LEGEND_ROW_PX)
        ]
        grid = self.figure.add_gridspec(
            self._rows * 2,
            self._cols,
            height_ratios=row_ratios,
            hspace=0.55,
            wspace=self._GRID_WSPACE,
        )
        self.axes = {}
        self.legend_axes = {}
        self._figure_legend = None
        for index, name in enumerate(self._axis_names):
            row, col = divmod(index, self._cols)
            self.axes[name] = self.figure.add_subplot(grid[row * 2, col])
            legend_axis = self.figure.add_subplot(grid[row * 2 + 1, col])
            legend_axis.set_axis_off()
            self.legend_axes[name] = legend_axis
        restyle_figure(self.figure)

    def _minimum_canvas_width(self) -> int:
        """Return the minimum canvas width that keeps plot legends readable."""
        return self._cols * self._MIN_AXIS_WIDTH_PX

    def _minimum_canvas_height(self) -> int:
        """Return the minimum canvas height that keeps plot legends readable."""
        return self._rows * (self._MIN_DATA_HEIGHT_PX + self._LEGEND_ROW_PX)

    def clear(self) -> None:
        """Reset every axis to a blank, themed state."""
        self._build_axes()

    def _dock_legends(self) -> None:
        """Move plot legends into reserved non-data axes under each plot."""
        if self._figure_legend is not None:
            self._figure_legend.remove()
            self._figure_legend = None

        for legend_axis in self.legend_axes.values():
            legend_axis.clear()
            legend_axis.set_axis_off()

        for name, axes in self.axes.items():
            handles_by_label: dict[str, Any] = {}
            handles, labels = axes.get_legend_handles_labels()
            existing = axes.get_legend()
            if existing is not None:
                existing.remove()
            for handle, label in zip(handles, labels, strict=False):
                if label and not label.startswith("_"):
                    handles_by_label.setdefault(label, handle)

            legend_labels = tuple(handles_by_label)
            if not legend_labels or not self._legends_visible:
                continue

            self.legend_axes[name].legend(
                [handles_by_label[label] for label in legend_labels],
                legend_labels,
                loc="center",
                ncol=self._legend_columns(legend_labels),
                fontsize=6,
                facecolor=Palette.BG_PANEL,
                edgecolor=Palette.FG_DIM,
                labelcolor=Palette.FG,
                framealpha=0.9,
                borderaxespad=0.0,
                borderpad=0.3,
                handlelength=1.15,
                handletextpad=0.32,
                columnspacing=0.62,
            )

    @staticmethod
    def _legend_columns(labels: Sequence[str]) -> int:
        """Return a compact, readable column count for the reserved legend footer."""
        if not labels:
            return 1
        return min(6, max(1, len(labels)))

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
        if self._figure_legend is not None:
            self._figure_legend.set_visible(self._legends_visible)

    def has_legends(self) -> bool:
        """Return True if any axis currently carries a legend."""
        return (
            self._figure_legend is not None
            or any(axes.get_legend() is not None for axes in self.axes.values())
            or any(axes.get_legend() is not None for axes in self.legend_axes.values())
        )

    def draw(self) -> None:
        """Lay out and repaint the figure after the axes have been populated."""
        self._enforce_minimum_figure_size()
        self.figure.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.94,
            bottom=0.045,
        )
        self._dock_legends()
        self.canvas.draw()

    def _enforce_minimum_figure_size(self) -> None:
        """Keep rendered figures at the panel's legend-safe minimum size."""
        dpi = float(getattr(self.figure, "dpi", 100.0))
        width_in, height_in = self.figure.get_size_inches()
        min_width_in = self._minimum_canvas_width() / dpi
        min_height_in = self._minimum_canvas_height() / dpi
        next_width_in = max(float(width_in), min_width_in)
        next_height_in = max(float(height_in), min_height_in)
        if next_width_in == float(width_in) and next_height_in == float(height_in):
            return
        self.figure.set_size_inches(next_width_in, next_height_in, forward=True)
