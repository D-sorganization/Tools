"""
Widget for displaying the 8x8 mass matrix, force balance, energy,
constraint violation, ZTCF, and Delta matrix for the golfer model.

QPainter-based rendering with color-coded cells.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..simulation_golfer import GolferSimulationResult


class GolferMatrixWidget(QWidget):
    """Displays mass matrix, energy, forces, and constraints in real time."""

    COLOR_DIAGONAL = QColor(70, 130, 230)
    COLOR_OFFDIAG = QColor(230, 160, 50)
    COLOR_BG = QColor(30, 30, 40)
    COLOR_CELL_BG_DIAG = QColor(40, 55, 80)
    COLOR_CELL_BG_OFF = QColor(80, 60, 30)
    COLOR_TEXT = QColor(220, 220, 235)
    COLOR_LABEL = QColor(160, 160, 180)
    COLOR_BRACKET = QColor(120, 120, 140)
    COLOR_CONSTRAINT_OK = QColor(80, 200, 120)
    COLOR_CONSTRAINT_BAD = QColor(230, 80, 80)

    DOF_LABELS = ["Hub", "RS", "RE", "RH", "LS", "LE", "LH", "Club"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(280, 400)
        self._result: GolferSimulationResult | None = None
        self._current_idx: int = 0

    def set_simulation(self, result: GolferSimulationResult) -> None:
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
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No simulation loaded",
            )
            painter.end()
            return

        y = 10
        y = self._draw_title(painter, "Mass Matrix M(q)  [8x8]", y)
        y = self._draw_mass_matrix_compact(painter, y)
        y += 8

        y = self._draw_title(painter, "Constraint Violation", y)
        y = self._draw_constraint_violation(painter, y)
        y += 8

        y = self._draw_title(painter, "Force Balance", y)
        y = self._draw_force_balance(painter, y)
        y += 8

        y = self._draw_title(painter, "Energy", y)
        y = self._draw_energy(painter, y)

        painter.end()

    def _draw_title(self, painter: QPainter, title: str, y: int) -> int:
        painter.setPen(self.COLOR_TEXT)
        painter.setFont(QFont("Sans", 10, QFont.Weight.Bold))
        painter.drawText(12, y + 14, title)
        painter.setPen(QPen(self.COLOR_BRACKET, 1))
        painter.drawLine(12, y + 18, self.width() - 12, y + 18)
        return y + 24

    def _draw_mass_matrix_compact(self, painter: QPainter, y: int) -> int:
        """Draw the 8x8 mass matrix in compact heat-map style."""
        assert self._result is not None
        M = self._result.mass_matrix_at(self._current_idx)

        n = 8
        avail_w = self.width() - 60
        cell = max(16, min(32, avail_w // n))
        grid_w = n * cell
        margin_x = (self.width() - grid_w) // 2

        # Find max absolute value for color scaling
        max_val = max(np.abs(M).max(), 1e-12)

        painter.setFont(QFont("Monospace", 7))

        for row in range(n):
            for col in range(n):
                cx = margin_x + col * cell
                cy = y + row * cell
                val = M[row, col]
                is_diag = row == col

                # Color intensity based on magnitude
                intensity = min(abs(val) / max_val, 1.0)
                if is_diag:
                    bg = QColor(
                        int(40 + 40 * intensity),
                        int(55 + 40 * intensity),
                        int(80 + 50 * intensity),
                    )
                    border = self.COLOR_DIAGONAL
                else:
                    bg = QColor(
                        int(50 + 60 * intensity),
                        int(40 + 40 * intensity),
                        int(30 + 20 * intensity),
                    )
                    border = self.COLOR_OFFDIAG

                rect = QRectF(cx, cy, cell, cell)
                painter.setPen(QPen(border, 1))
                painter.setBrush(QBrush(bg))
                painter.drawRect(rect)

                # Value text (abbreviated)
                painter.setPen(self.COLOR_TEXT)
                if abs(val) > 0.01:
                    txt = f"{val:.1f}"
                else:
                    txt = f"{val:.0e}" if val != 0 else "0"
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)

        # Row/column labels
        painter.setPen(self.COLOR_LABEL)
        painter.setFont(QFont("Sans", 6))
        for i, label in enumerate(self.DOF_LABELS):
            # Column labels (top)
            lx = margin_x + i * cell
            painter.drawText(lx, y - 2, label)
            # Row labels (left)
            ly = y + i * cell + cell // 2 + 3
            painter.drawText(margin_x - 28, ly, label)

        return y + n * cell + 8

    def _draw_constraint_violation(self, painter: QPainter, y: int) -> int:
        assert self._result is not None
        v = self._result.constraint_violation_at(self._current_idx)

        bar_x = 20
        bar_w = self.width() - 40
        bar_h = 20

        # Background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(50, 50, 60)))
        painter.drawRoundedRect(QRectF(bar_x, y, bar_w, bar_h), 4, 4)

        # Fill bar (capped at 1.0 for display)
        ratio = min(v * 1000, 1.0)  # Scale: 0.001 = full bar
        color = self.COLOR_CONSTRAINT_OK if v < 1e-4 else self.COLOR_CONSTRAINT_BAD
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(QRectF(bar_x, y, bar_w * ratio, bar_h), 4, 4)

        # Text
        painter.setPen(self.COLOR_TEXT)
        painter.setFont(QFont("Monospace", 9, QFont.Weight.Bold))
        painter.drawText(
            QRectF(bar_x, y, bar_w, bar_h),
            Qt.AlignmentFlag.AlignCenter,
            f"||Phi|| = {v:.2e}",
        )

        return y + bar_h + 6

    def _draw_force_balance(self, painter: QPainter, y: int) -> int:
        assert self._result is not None
        idx = self._current_idx
        tau = self._result.torques_at(idx)
        G = self._result.gravity_at(idx)

        painter.setFont(QFont("Monospace", 9))
        joint_names = ["Hub", "RS", "RE", "RH", "LS", "LE", "LH"]

        for i, name in enumerate(joint_names):
            text = f"tau_{name:>3s} = {tau[i]:+7.2f}  G = {G[i]:+7.2f}"
            painter.setPen(self.COLOR_TEXT)
            painter.drawText(16, y + 13, text)
            y += 15

        return y + 4

    def _draw_energy(self, painter: QPainter, y: int) -> int:
        assert self._result is not None
        e = self._result.energy_at(self._current_idx)

        painter.setFont(QFont("Monospace", 10))
        lines = [
            (f"Kinetic    = {e['kinetic']:+8.2f} J", QColor(230, 160, 80)),
            (
                f"Potential  = {e['potential']:+8.2f} J",
                QColor(80, 180, 230),
            ),
            (f"Total      = {e['total']:+8.2f} J", self.COLOR_TEXT),
        ]
        for text, color in lines:
            painter.setPen(color)
            painter.drawText(16, y + 14, text)
            y += 18

        # Constraint forces
        try:
            lam = self._result.constraint_forces_at(self._current_idx)
            painter.setPen(QColor(200, 150, 220))
            painter.setFont(QFont("Monospace", 9))
            y += 4
            painter.drawText(16, y + 12, "Constraint Forces:")
            y += 14
            for i in range(len(lam)):
                painter.drawText(16, y + 12, f"  lambda_{i + 1} = {lam[i]:+8.3f}")
                y += 14
        except Exception:
            pass

        return y + 4
