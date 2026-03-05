"""
Torque preview widget for polynomial joint torques.
"""

from typing import Iterable, List, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import QWidget


class TorquePreviewWidget(QWidget):
    """Plot polynomial torque profiles for one or more joints."""

    COLOR_BG = QColor(26, 26, 36)
    COLOR_GRID = QColor(50, 50, 65)
    COLOR_TEXT = QColor(180, 180, 200)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self._profiles: List[Tuple[str, List[float], QColor]] = []
        self._t_end = 2.0

    def set_duration(self, t_end: float) -> None:
        self._t_end = max(0.1, float(t_end))
        self.update()

    def set_profiles(self, profiles: Iterable[Tuple[str, List[float], QColor]]) -> None:
        self._profiles = [(name, list(coeffs), color) for name, coeffs, color in profiles]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.COLOR_BG)

        rect = self.rect().adjusted(10, 10, -10, -24)
        self._draw_grid(painter, rect)

        if not self._profiles:
            painter.setPen(self.COLOR_TEXT)
            painter.setFont(QFont("Sans", 9))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Torque preview")
            painter.end()
            return

        t = np.linspace(0.0, self._t_end, 120)
        series = []
        for name, coeffs, _ in self._profiles:
            if len(coeffs) == 0:
                series.append((name, np.zeros_like(t)))
                continue
            poly = np.array(coeffs[::-1])
            series.append((name, np.polyval(poly, t)))

        all_vals = np.concatenate([s[1] for s in series]) if series else np.array([0.0])
        v_min = float(np.min(all_vals))
        v_max = float(np.max(all_vals))
        if abs(v_max - v_min) < 1e-6:
            v_min -= 1.0
            v_max += 1.0

        for (name, values), (_, _, color) in zip(series, self._profiles):
            pen = QPen(color, 2)
            painter.setPen(pen)
            points = []
            for i, val in enumerate(values):
                x = rect.left() + (t[i] / self._t_end) * rect.width()
                y = rect.bottom() - (val - v_min) / (v_max - v_min) * rect.height()
                points.append((x, y))
            for i in range(1, len(points)):
                painter.drawLine(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1])

        self._draw_legend(painter, rect)
        painter.end()

    def _draw_grid(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(self.COLOR_GRID, 1, Qt.PenStyle.DotLine))
        for i in range(5):
            y = rect.top() + i * rect.height() / 4
            painter.drawLine(rect.left(), y, rect.right(), y)
        for i in range(5):
            x = rect.left() + i * rect.width() / 4
            painter.drawLine(x, rect.top(), x, rect.bottom())

    def _draw_legend(self, painter: QPainter, rect: QRectF) -> None:
        painter.setFont(QFont("Sans", 8))
        x = rect.left()
        y = rect.bottom() + 14
        for name, _, color in self._profiles:
            painter.setPen(color)
            painter.drawText(int(x), int(y), name)
            x += 70
