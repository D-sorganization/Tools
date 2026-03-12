"""
Widget that displays the 3x3 mass matrix with real-time values,
color-coded to distinguish diagonal (self-coupling) from off-diagonal
(cross-coupling) terms.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..simulation_triple import TripleSimulationResult


class TripleMatrixWidget(QWidget):
    """Displays the mass matrix and force decomposition in real time."""

    COLOR_DIAGONAL = QColor(70, 130, 230)
    COLOR_OFFDIAG = QColor(230, 160, 50)
    COLOR_BG = QColor(30, 30, 40)
    COLOR_CELL_BG_DIAG = QColor(40, 55, 80)
    COLOR_CELL_BG_OFF = QColor(80, 60, 30)
    COLOR_TEXT = QColor(220, 220, 235)
    COLOR_LABEL = QColor(160, 160, 180)
    COLOR_BRACKET = QColor(120, 120, 140)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 300)
        self._result: TripleSimulationResult | None = None
        self._current_idx: int = 0

    def set_simulation(self, result: TripleSimulationResult) -> None:
        self._result = result
        self._current_idx = 0
        self.update()

    def set_frame(self, idx: int) -> None:
        if self._result is None:
            return
        self._current_idx = max(0, min(idx, self._result.n_steps - 1))
        self.update()

    def clear(self) -> None:
        self._result = None
        self._current_idx = 0
        self.update()

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.COLOR_BG)

        if self._result is None:
            painter.setPen(self.COLOR_LABEL)
            painter.setFont(QFont("Sans", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No simulation loaded")
            painter.end()
            return

        mc = self._result.mass_matrix_at(self._current_idx)
        y_cursor = 10

        y_cursor = self._draw_section_title(painter, "Mass Matrix M(q)", y_cursor)
        y_cursor = self._draw_matrix(painter, mc, y_cursor)
        y_cursor += 10

        y_cursor = self._draw_section_title(painter, "Coupling Ratio", y_cursor)
        y_cursor = self._draw_coupling_ratio(painter, mc, y_cursor)
        y_cursor += 10

        y_cursor = self._draw_section_title(painter, "Force Balance", y_cursor)
        y_cursor = self._draw_force_balance(painter, y_cursor)
        y_cursor += 10

        y_cursor = self._draw_section_title(painter, "Energy", y_cursor)
        y_cursor = self._draw_energy(painter, y_cursor)

        painter.end()

    def _draw_section_title(self, painter: QPainter, title: str, y: int) -> int:
        painter.setPen(self.COLOR_TEXT)
        painter.setFont(QFont("Sans", 11, QFont.Weight.Bold))
        painter.drawText(15, y + 16, title)
        painter.setPen(QPen(self.COLOR_BRACKET, 1))
        painter.drawLine(15, y + 20, self.width() - 15, y + 20)
        return y + 28

    def _draw_matrix(self, painter: QPainter, mc: dict, y: int) -> int:
        cell_w = 100
        cell_h = 44
        gap = 8
        grid_w = 3 * cell_w + 2 * gap
        margin_x = (self.width() - grid_w) // 2
        bracket_w = 8

        entries = [
            (0, 0, "M11", mc["M11"], True),
            (0, 1, "M12", mc["M12"], False),
            (0, 2, "M13", mc["M13"], False),
            (1, 0, "M21", mc["M21"], False),
            (1, 1, "M22", mc["M22"], True),
            (1, 2, "M23", mc["M23"], False),
            (2, 0, "M31", mc["M31"], False),
            (2, 1, "M32", mc["M32"], False),
            (2, 2, "M33", mc["M33"], True),
        ]

        bx_left = margin_x - bracket_w - 4
        bx_right = margin_x + grid_w + bracket_w + 4
        by_top = y
        by_bot = y + 3 * cell_h + 2 * gap

        painter.setPen(QPen(self.COLOR_BRACKET, 2))
        painter.drawLine(bx_left + bracket_w, by_top, bx_left, by_top)
        painter.drawLine(bx_left, by_top, bx_left, by_bot)
        painter.drawLine(bx_left, by_bot, bx_left + bracket_w, by_bot)
        painter.drawLine(bx_right - bracket_w, by_top, bx_right, by_top)
        painter.drawLine(bx_right, by_top, bx_right, by_bot)
        painter.drawLine(bx_right, by_bot, bx_right - bracket_w, by_bot)

        for row, col, label, value, is_diag in entries:
            cx = margin_x + col * (cell_w + gap)
            cy = y + row * (cell_h + gap)

            bg = self.COLOR_CELL_BG_DIAG if is_diag else self.COLOR_CELL_BG_OFF
            border = self.COLOR_DIAGONAL if is_diag else self.COLOR_OFFDIAG

            rect = QRectF(cx, cy, cell_w, cell_h)
            painter.setPen(QPen(border, 2))
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(rect, 6, 6)

            painter.setPen(self.COLOR_LABEL)
            painter.setFont(QFont("Monospace", 8))
            painter.drawText(cx + 5, cy + 14, label)

            painter.setPen(self.COLOR_TEXT)
            painter.setFont(QFont("Monospace", 12, QFont.Weight.Bold))
            painter.drawText(cx + 5, cy + 34, f"{value:.3f}")

        ly = y + 3 * cell_h + 2 * gap + 16
        painter.setFont(QFont("Sans", 9))
        painter.setPen(self.COLOR_DIAGONAL)
        painter.drawText(margin_x, ly, "[ ] Diagonal (self-coupling)")
        painter.setPen(self.COLOR_OFFDIAG)
        painter.drawText(margin_x, ly + 16, "[ ] Off-diagonal (cross-coupling)")

        return ly + 32

    def _draw_coupling_ratio(self, painter: QPainter, mc: dict, y: int) -> int:
        diag = np.array([mc["M11"], mc["M22"], mc["M33"]])
        off = np.array(
            [
                mc["M12"],
                mc["M13"],
                mc["M21"],
                mc["M23"],
                mc["M31"],
                mc["M32"],
            ]
        )
        denom = np.mean(np.abs(diag)) if np.any(diag) else 1.0
        ratio = min(np.mean(np.abs(off)) / max(denom, 1e-12), 1.0)

        bar_x = 20
        bar_w = self.width() - 40
        bar_h = 22

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(50, 50, 60)))
        painter.drawRoundedRect(QRectF(bar_x, y, bar_w, bar_h), 4, 4)

        painter.setBrush(QBrush(self.COLOR_OFFDIAG))
        painter.drawRoundedRect(QRectF(bar_x, y, bar_w * ratio, bar_h), 4, 4)

        painter.setPen(self.COLOR_TEXT)
        painter.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(bar_x, y, bar_w, bar_h),
            Qt.AlignmentFlag.AlignCenter,
            f"avg |Moff|/|Mdiag| = {ratio:.1%}",
        )

        return y + bar_h + 6

    def _draw_force_balance(self, painter: QPainter, y: int) -> int:
        assert self._result is not None  # only called after None guard in paintEvent
        idx = self._current_idx
        tau = self._result.torques_at(idx)
        C = self._result.coriolis_at(idx)
        G = self._result.gravity_at(idx)

        painter.setFont(QFont("Monospace", 10))
        lines = [
            (f"tau1 (shoulder) = {tau[0]:+8.2f} N*m", self.COLOR_TEXT),
            (f"tau2 (elbow)    = {tau[1]:+8.2f} N*m", self.COLOR_TEXT),
            (f"tau3 (wrist)    = {tau[2]:+8.2f} N*m", self.COLOR_TEXT),
            (f"C1 (Coriolis)   = {C[0]:+8.2f} N*m", QColor(180, 130, 200)),
            (f"C2 (Coriolis)   = {C[1]:+8.2f} N*m", QColor(180, 130, 200)),
            (f"C3 (Coriolis)   = {C[2]:+8.2f} N*m", QColor(180, 130, 200)),
            (f"G1 (gravity)    = {G[0]:+8.2f} N*m", QColor(130, 200, 130)),
            (f"G2 (gravity)    = {G[1]:+8.2f} N*m", QColor(130, 200, 130)),
            (f"G3 (gravity)    = {G[2]:+8.2f} N*m", QColor(130, 200, 130)),
        ]
        for text, color in lines:
            painter.setPen(color)
            painter.drawText(20, y + 14, text)
            y += 18

        return y + 4

    def _draw_energy(self, painter: QPainter, y: int) -> int:
        assert self._result is not None  # only called after None guard in paintEvent
        e = self._result.energy_at(self._current_idx)
        painter.setFont(QFont("Monospace", 10))
        lines = [
            (f"Kinetic    = {e['kinetic']:+8.2f} J", QColor(230, 160, 80)),
            (f"Potential  = {e['potential']:+8.2f} J", QColor(80, 180, 230)),
            (f"Total      = {e['total']:+8.2f} J", self.COLOR_TEXT),
        ]
        for text, color in lines:
            painter.setPen(color)
            painter.drawText(20, y + 14, text)
            y += 18
        return y + 4
