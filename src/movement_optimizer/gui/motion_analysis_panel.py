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
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from ..rendering import Palette, restyle_figure


class _ExternalLegendProxy:
    """Compatibility shim for callers that inspect panel legend state."""

    def __init__(self, panel: MotionAnalysisPanel, name: str) -> None:
        self._panel = panel
        self._name = name

    def get_legend(self) -> tuple[str, ...] | None:
        if not self._panel._legends_visible:
            return None
        return self._panel._legend_entries.get(self._name)


class MotionAnalysisPanel(QWidget):
    """A themed grid of named matplotlib axes with a navigation toolbar."""

    _PLOT_HSPACE = 0.58
    _LEGEND_COLUMNS = 3
    _LEGEND_FONT_STYLE = 'font-family: "Segoe UI", Arial, sans-serif; font-size: 10px;'

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
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.axes: dict[str, Axes] = {}
        self.legend_axes: dict[str, _ExternalLegendProxy] = {}
        self._legend_entries: dict[str, tuple[str, ...]] = {}
        self._legends_visible = True
        self._legend_scroll = QScrollArea()
        self._legend_scroll.setWidgetResizable(True)
        self._legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._legend_scroll.setMaximumHeight(112)
        self._legend_scroll.setVisible(False)
        self._legend_widget = QWidget()
        self._legend_layout = QGridLayout(self._legend_widget)
        self._legend_layout.setContentsMargins(8, 4, 8, 4)
        self._legend_layout.setHorizontalSpacing(10)
        self._legend_layout.setVerticalSpacing(4)
        self._legend_scroll.setWidget(self._legend_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self._legend_scroll)

        self._build_axes()

    def _build_axes(self) -> None:
        self.figure.clear()
        grid = self.figure.add_gridspec(
            self._rows,
            self._cols,
            hspace=self._PLOT_HSPACE,
            wspace=0.34,
        )
        self.axes = {}
        self.legend_axes = {}
        for index, name in enumerate(self._axis_names):
            row, col = divmod(index, self._cols)
            self.axes[name] = self.figure.add_subplot(grid[row, col])
            self.legend_axes[name] = _ExternalLegendProxy(self, name)
        restyle_figure(self.figure)

    def clear(self) -> None:
        """Reset every axis to a blank, themed state."""
        self._build_axes()
        self._legend_entries = {}
        self._clear_legend_layout()
        self._legend_scroll.setVisible(False)

    @staticmethod
    def _handle_color(handle: object) -> str:
        color_getter = getattr(handle, "get_color", None)
        if color_getter is None:
            return str(Palette.FG)
        try:
            return to_hex(color_getter())
        except (TypeError, ValueError):
            return str(Palette.FG)

    def _clear_legend_layout(self) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _make_legend_group(
        self, title: str, handles: Sequence[object], labels: Sequence[str]
    ) -> QWidget:
        group = QWidget()
        group_layout = QGridLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setHorizontalSpacing(5)
        group_layout.setVerticalSpacing(2)

        title_label = QLabel(f"{title}:")
        title_label.setStyleSheet(
            f"color: {Palette.FG}; font-weight: 600; {self._LEGEND_FONT_STYLE}"
        )
        group_layout.addWidget(title_label, 0, 0, 1, 6)

        for index, (handle, text) in enumerate(zip(handles, labels, strict=True)):
            row = 1 + index // 3
            col = (index % 3) * 2
            swatch = QFrame()
            swatch.setFixedSize(14, 3)
            swatch.setStyleSheet(f"background-color: {self._handle_color(handle)};")
            label = QLabel(text)
            label.setStyleSheet(f"color: {Palette.FG}; {self._LEGEND_FONT_STYLE}")
            group_layout.addWidget(swatch, row, col)
            group_layout.addWidget(label, row, col + 1)
        return group

    @staticmethod
    def _legend_title(name: str) -> str:
        return name.replace("_", " ").title()

    def _sync_legend_bar(self) -> None:
        """Move legends into the Qt legend bar below the canvas."""
        self._clear_legend_layout()
        self._legend_entries = {}
        self._legend_scroll.setStyleSheet(
            f"QScrollArea {{ background: {Palette.BG_PANEL}; border: 0; }}"
        )
        self._legend_widget.setStyleSheet(f"background: {Palette.BG_PANEL};")
        legend_sources: list[tuple[str, list[object], list[str]]] = []
        visible_count = 0
        for name, axes in self.axes.items():
            handles, labels = axes.get_legend_handles_labels()
            existing = axes.get_legend()
            if existing is not None:
                existing.set_visible(False)
                existing.remove()
            if not handles or not labels:
                continue
            legend_sources.append((name, handles, labels))

        if not self._legends_visible:
            self._legend_scroll.setVisible(False)
            return

        for name, handles, labels in legend_sources:
            label = self._make_legend_group(self._legend_title(name), handles, labels)
            row, col = divmod(visible_count, self._LEGEND_COLUMNS)
            self._legend_layout.addWidget(label, row, col)
            self._legend_entries[name] = tuple(labels)
            visible_count += 1
        self._legend_scroll.setVisible(visible_count > 0)

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
        self._legend_scroll.setVisible(self._legends_visible and bool(self._legend_entries))

    def has_legends(self) -> bool:
        """Return True if any axis currently carries a legend."""
        return bool(self._legend_entries) or any(
            bool(axes.get_legend_handles_labels()[1]) for axes in self.axes.values()
        )

    @property
    def legend_entries(self) -> dict[str, tuple[str, ...]]:
        """Return the labels currently shown in the external legend bar."""
        return dict(self._legend_entries)

    def draw(self) -> None:
        """Lay out and repaint the figure after the axes have been populated."""
        self._sync_legend_bar()
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.14)
        self.canvas.draw()
