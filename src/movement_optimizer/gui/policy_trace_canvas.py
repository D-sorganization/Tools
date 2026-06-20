# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Trace plot canvas for swingset policy searches."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import numpy as np
from PyQt6.QtCore import QPointF, QRect
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPen, QResizeEvent
from PyQt6.QtWidgets import QWidget

from movement_optimizer.models.swingset import CyclicPolicyTraceSample
from movement_optimizer.rendering import Palette, get_chart_color


def _build_trace_colors() -> dict[str, QColor]:
    return {
        "ARM": QColor(get_chart_color(2)),
        "BODY": QColor(get_chart_color(0)),
        "CHAIN": QColor(Palette.FG_DIM),
        "GRID": QColor(Palette.BG_INPUT),
        "LEG": QColor(get_chart_color(1)),
        "SURFACE": QColor(Palette.BG),
        "TRACE_BEST": QColor(Palette.GREEN),
        "TRACE_PARAM": QColor(get_chart_color(0)),
        "TRACE_SCORE": QColor(get_chart_color(1)),
    }


_TRACE_COLORS = _build_trace_colors()
ARM = _TRACE_COLORS["ARM"]
BODY = _TRACE_COLORS["BODY"]
CHAIN = _TRACE_COLORS["CHAIN"]
GRID = _TRACE_COLORS["GRID"]
LEG = _TRACE_COLORS["LEG"]
SURFACE = _TRACE_COLORS["SURFACE"]
TRACE_BEST = _TRACE_COLORS["TRACE_BEST"]
TRACE_PARAM = _TRACE_COLORS["TRACE_PARAM"]
TRACE_SCORE = _TRACE_COLORS["TRACE_SCORE"]


def refresh_policy_trace_palette() -> None:
    """Rebind trace-canvas colours from the active theme palette."""
    global ARM, BODY, CHAIN, GRID, LEG, SURFACE, TRACE_BEST, TRACE_PARAM, TRACE_SCORE
    colors = _build_trace_colors()
    ARM = colors["ARM"]
    BODY = colors["BODY"]
    CHAIN = colors["CHAIN"]
    GRID = colors["GRID"]
    LEG = colors["LEG"]
    SURFACE = colors["SURFACE"]
    TRACE_BEST = colors["TRACE_BEST"]
    TRACE_PARAM = colors["TRACE_PARAM"]
    TRACE_SCORE = colors["TRACE_SCORE"]


class PolicyTraceCanvas(QWidget):
    """Compact plot of policy-search score and parameter traces."""

    _BASE_MIN_HEIGHT_PX = 160
    _LEGEND_BOTTOM_PADDING_PX = 6
    _LEGEND_GAP_PX = 12
    _LEGEND_LINE_PX = 12
    _LEGEND_TEXT_GAP_PX = 4
    _LEGEND_ROW_HEIGHT_PX = 16
    _MARGIN_PX = 8
    _MINIMUM_PLOT_HEIGHT_PX = 96
    _AXIS_LABEL_TOP_PADDING_PX = 4
    _AXIS_LABEL_BOTTOM_PADDING_PX = 6

    def __init__(self) -> None:
        super().__init__()
        self._samples: tuple[CyclicPolicyTraceSample, ...] = ()
        self._series: dict[str, np.ndarray] = {}
        self._legend_visible = True
        self._sync_minimum_height()

    def set_trace(self, samples: tuple[CyclicPolicyTraceSample, ...]) -> None:
        self._samples = samples
        self._series = self._build_series(samples)
        self.update()

    def set_legend_visible(self, visible: bool) -> None:
        """Show or hide the inline legend and repaint."""
        self._legend_visible = bool(visible)
        self.update()

    def legend_visible(self) -> bool:
        """Return whether the inline legend is currently drawn."""
        return self._legend_visible

    def _top_margin(self) -> float:
        """Top inset for the plotted series, reserving room for the legend."""
        return float(self._legend_band_height() if self._legend_visible else self._MARGIN_PX)

    @staticmethod
    def _legend_entries() -> tuple[tuple[str, QColor], ...]:
        return (
            ("best", TRACE_BEST),
            ("score", TRACE_SCORE),
            ("freq", TRACE_PARAM),
            ("hip", ARM),
            ("torso", BODY),
            ("knee", LEG),
        )

    def _legend_item_width(self, label: str) -> int:
        metrics = QFontMetrics(self.font())
        return (
            self._LEGEND_LINE_PX
            + self._LEGEND_TEXT_GAP_PX
            + metrics.horizontalAdvance(label)
            + self._LEGEND_GAP_PX
        )

    def _available_legend_width(self, width: int | None = None) -> int:
        """Return the usable legend width after horizontal margins."""
        widget_width = self.width() if width is None else width
        return max(1, widget_width - 2 * self._MARGIN_PX)

    def _legend_row_count(self) -> int:
        return self._legend_row_count_for_width(self.width())

    def _legend_row_count_for_width(self, width: int) -> int:
        """Return wrapped legend rows for a proposed widget width."""
        available_width = self._available_legend_width(width)
        rows = 1
        row_width = 0
        for label, _color in self._legend_entries():
            item_width = self._legend_item_width(label)
            if row_width > 0 and row_width + item_width > available_width:
                rows += 1
                row_width = 0
            row_width += item_width
        return rows

    def _legend_band_height(self) -> int:
        """Return the reserved legend band height for the current widget width."""
        return self._legend_band_height_for_width(self.width())

    def _legend_band_height_for_width(self, width: int) -> int:
        """Return reserved legend height for a proposed widget width."""
        return (
            self._MARGIN_PX
            + self._legend_row_count_for_width(width) * self._LEGEND_ROW_HEIGHT_PX
            + self._LEGEND_BOTTOM_PADDING_PX
        )

    def _axis_label_band_height(self) -> int:
        """Return reserved bottom height for the trace x-axis label."""
        metrics = QFontMetrics(self.font())
        return (
            self._AXIS_LABEL_TOP_PADDING_PX + metrics.height() + self._AXIS_LABEL_BOTTOM_PADDING_PX
        )

    def _minimum_height_for_width(self, width: int) -> int:
        """Return the height needed for legends plus readable trace data."""
        return max(
            self._BASE_MIN_HEIGHT_PX,
            self._legend_band_height_for_width(width)
            + self._MINIMUM_PLOT_HEIGHT_PX
            + self._axis_label_band_height(),
        )

    def _sync_minimum_height(self) -> None:
        """Keep Qt layout constraints aligned with the wrapped legend height."""
        width = max(1, self.width())
        minimum_height = self._minimum_height_for_width(width)
        if self.minimumHeight() == minimum_height:
            return
        self.setMinimumHeight(minimum_height)
        self.updateGeometry()

    def hasHeightForWidth(self) -> bool:
        """Tell Qt layouts that narrow trace canvases need more height."""
        return True

    def heightForWidth(self, width: int) -> int:
        """Return width-aware preferred height for wrapped legends."""
        return self._minimum_height_for_width(width)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Update the minimum height when a narrower width wraps the legend."""
        self._sync_minimum_height()
        super().resizeEvent(event)

    def sample_count(self) -> int:
        return len(self._samples)

    def has_parameter_series(self, name: str) -> bool:
        return name in self._series and self._series[name].size > 0

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), SURFACE)
        painter.setPen(QPen(GRID, 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if len(self._samples) < 2:
            return
        self._draw_normalized_series(painter, "best_score_m", TRACE_BEST, 3)
        self._draw_normalized_series(painter, "score_m", TRACE_SCORE, 2)
        self._draw_normalized_series(painter, "frequency_hz", TRACE_PARAM, 1)
        self._draw_normalized_series(painter, "hip_rate_amplitude_rad_s", ARM, 1)
        self._draw_normalized_series(painter, "torso_rate_amplitude_rad_s", BODY, 1)
        self._draw_normalized_series(painter, "knee_rate_ratio", LEG, 1)
        if self._legend_visible:
            self._draw_legend(painter)
        self._draw_axis_label(painter)

    def _draw_normalized_series(
        self,
        painter: QPainter,
        key: str,
        color: QColor,
        width: int,
    ) -> None:
        values = self._series.get(key)
        if values is None or values.size < 2:
            return
        lower = float(np.min(values))
        upper = float(np.max(values))
        if np.isclose(lower, upper):
            normalized = np.full(values.shape, 0.5, dtype=np.float64)
        else:
            normalized = (values - lower) / (upper - lower)
        top = self._top_margin()
        bottom = self._plot_bottom()
        span = max(bottom - top, 1.0)
        points = [
            QPointF(
                8.0 + index * (self.width() - 16.0) / (values.size - 1),
                bottom - value * span,
            )
            for index, value in enumerate(normalized)
        ]
        painter.setPen(QPen(color, width))
        for start, end in pairwise(points):
            painter.drawLine(start, end)

    def _plot_bottom(self) -> float:
        """Bottom edge of the data rectangle above the reserved axis label."""
        return float(self.height() - self._axis_label_band_height())

    def _draw_legend(self, painter: QPainter) -> None:
        available_width = self._available_legend_width()
        x = self._MARGIN_PX
        baseline = self._MARGIN_PX + QFontMetrics(painter.font()).ascent()
        y = baseline
        for label, color in self._legend_entries():
            item_width = self._legend_item_width(label)
            if x > self._MARGIN_PX and x + item_width > self._MARGIN_PX + available_width:
                x = self._MARGIN_PX
                y += self._LEGEND_ROW_HEIGHT_PX
            painter.setPen(QPen(color, 2))
            painter.drawLine(x, y - 4, x + 12, y - 4)
            painter.setPen(QPen(color, 1))
            painter.drawText(x + self._LEGEND_LINE_PX + self._LEGEND_TEXT_GAP_PX, y, label)
            x += item_width

    def _iteration_label_rect(self) -> QRect:
        """Return the reserved text rectangle for the trace x-axis label."""
        metrics = QFontMetrics(self.font())
        label = "iteration"
        text_width = metrics.horizontalAdvance(label)
        text_height = metrics.height()
        x = max(self._MARGIN_PX, self.width() - self._MARGIN_PX - text_width)
        y = self.height() - self._AXIS_LABEL_BOTTOM_PADDING_PX - text_height
        return QRect(x, y, text_width, text_height)

    def _draw_axis_label(self, painter: QPainter) -> None:
        """Draw trace axis text in its reserved band, outside plotted data."""
        painter.setPen(QPen(CHAIN, 1))
        painter.drawText(self._iteration_label_rect(), 0, "iteration")

    def _build_series(
        self,
        samples: tuple[CyclicPolicyTraceSample, ...],
    ) -> dict[str, np.ndarray]:
        return {
            "score_m": _trace_series(samples, lambda sample: sample.score_m),
            "best_score_m": _trace_series(samples, lambda sample: sample.best_score_m),
            "frequency_hz": _trace_series(samples, lambda sample: sample.parameters.frequency_hz),
            "hip_rate_amplitude_rad_s": _trace_series(
                samples, lambda sample: sample.parameters.hip_rate_amplitude_rad_s
            ),
            "torso_rate_amplitude_rad_s": _trace_series(
                samples, lambda sample: sample.parameters.torso_rate_amplitude_rad_s
            ),
            "knee_rate_ratio": _trace_series(
                samples, lambda sample: sample.parameters.knee_rate_ratio
            ),
            "phase_rad": _trace_series(samples, lambda sample: sample.parameters.phase_rad),
        }


def _trace_series(
    samples: tuple[CyclicPolicyTraceSample, ...],
    read_value: Callable[[CyclicPolicyTraceSample], float],
) -> np.ndarray:
    return np.asarray([read_value(sample) for sample in samples], dtype=np.float64)
