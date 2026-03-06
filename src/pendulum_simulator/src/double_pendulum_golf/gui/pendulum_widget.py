"""
Custom QWidget that draws the double pendulum animation.

Renders the two segments, joint markers, a tip trail, and
optional ghosted past positions.  Coordinate mapping converts
physics-frame meters into pixel space with configurable scale.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..simulation import SimulationResult


class PendulumWidget(QWidget):
    """Animated visualization of the double pendulum.

    Draws shoulder (fixed pivot), arm segment, wrist joint, club segment,
    and club tip with a fading trail showing recent trajectory.
    """

    # Color palette
    COLOR_BG = QColor(20, 20, 30)
    COLOR_ARM = QColor(70, 130, 230)
    COLOR_CLUB = QColor(230, 120, 50)
    COLOR_SHOULDER = QColor(200, 200, 200)
    COLOR_WRIST = QColor(255, 220, 80)
    COLOR_WRIST2 = QColor(120, 220, 180)
    COLOR_TIP = QColor(255, 80, 80)
    COLOR_TRAIL = QColor(255, 80, 80, 120)
    COLOR_GRID = QColor(50, 50, 65)
    COLOR_TEXT = QColor(180, 180, 200)
    COLOR_GROUND = QColor(40, 100, 40, 80)

    TRAIL_LENGTH = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Responsive: no hard minimum — let the splitter govern size
        self.setMinimumSize(250, 300)

        self._result: SimulationResult | None = None
        self._current_idx: int = 0
        self._trail: deque = deque(maxlen=self.TRAIL_LENGTH)
        self._pixels_per_meter: float = 120.0  # recomputed each paint

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_simulation(self, result: SimulationResult) -> None:
        """Load a new simulation result and reset display."""
        assert result is not None
        self._result = result
        self._current_idx = 0
        self._trail.clear()
        self.update()

    def set_frame(self, idx: int) -> None:
        """Advance to frame idx and update the trail."""
        if self._result is None:
            return
        idx = max(0, min(idx, self._result.n_steps - 1))

        # Build trail: add tip position
        pos = self._result.positions_at(idx)
        self._trail.append(pos["tip"])
        self._current_idx = idx
        self.update()

    def clear(self) -> None:
        """Reset to blank state."""
        self._result = None
        self._current_idx = 0
        self._trail.clear()
        self.update()

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _compute_scale(self) -> float:
        """Compute pixels_per_meter so the full pendulum extent fits the widget.

        Uses actual widget width/height at call time — must be called at the
        start of each paintEvent so the scale is always current.
        """
        if self._result is not None:
            total_len = self._result.params.L1 + self._result.params.L2
        else:
            total_len = 2.0  # default for placeholder
        # Leave 20% margin horizontally, 60% of height available below pivot
        w_scale = self.width() * 0.40 / max(total_len, 1e-6)
        h_scale = self.height() * 0.60 / max(total_len, 1e-6)
        return max(30.0, min(w_scale, h_scale))

    def _world_to_pixel(self, x_world: float, y_world: float) -> QPointF:
        """Convert physics (x_right, y_up) to widget pixel coords (x_right, y_down).

        Origin (shoulder) is placed at top-center of the widget.
        """
        cx = self.width() / 2.0
        cy = self.height() * 0.20  # shoulder 20% down from top
        px = cx + x_world * self._pixels_per_meter
        py = cy - y_world * self._pixels_per_meter  # flip y
        return QPointF(px, py)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: object) -> None:
        # Recompute scale every paint so the pendulum fills the canvas at any size
        self._pixels_per_meter = self._compute_scale()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), self.COLOR_BG)

        self._draw_grid(painter)
        self._draw_ground_line(painter)

        if self._result is None:
            self._draw_placeholder(painter)
            painter.end()
            return

        self._draw_trail(painter)
        self._draw_pendulum(painter)
        self._draw_info(painter)
        painter.end()

    def _draw_grid(self, painter: QPainter) -> None:
        """Draw subtle reference grid lines every 0.5m."""
        pen = QPen(self.COLOR_GRID, 1, Qt.PenStyle.DotLine)
        painter.setPen(pen)

        max_range = 3.0  # meters
        step = 0.5
        r = max_range
        while r > -max_range:
            # Horizontal lines
            p1 = self._world_to_pixel(-max_range, r)
            p2 = self._world_to_pixel(max_range, r)
            painter.drawLine(p1, p2)
            # Vertical lines
            p1 = self._world_to_pixel(r, max_range)
            p2 = self._world_to_pixel(r, -max_range)
            painter.drawLine(p1, p2)
            r -= step

    def _draw_ground_line(self, painter: QPainter) -> None:
        """Draw a ground reference at y = -(L1+L2) to show full extension."""
        if self._result is None:
            return
        L_total = self._result.params.L1 + self._result.params.L2
        p1 = self._world_to_pixel(-3.0, -L_total)
        p2 = self._world_to_pixel(3.0, -L_total)
        pen = QPen(self.COLOR_GROUND, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

    def _draw_trail(self, painter: QPainter) -> None:
        """Draw fading trail of the club tip."""
        n = len(self._trail)
        if n < 2:
            return
        for i in range(1, n):
            alpha = int(40 + 160 * (i / n))
            color = QColor(255, 80, 80, alpha)
            width = 1.0 + 2.0 * (i / n)
            pen = QPen(color, width)
            painter.setPen(pen)
            x0, y0 = self._trail[i - 1]
            x1, y1 = self._trail[i]
            painter.drawLine(
                self._world_to_pixel(x0, y0),
                self._world_to_pixel(x1, y1),
            )

    def _draw_pendulum(self, painter: QPainter) -> None:
        """Draw the segments and joint markers."""
        assert self._result is not None  # only called after None guard in paintEvent
        pos = self._result.positions_at(self._current_idx)
        shoulder = self._world_to_pixel(*pos["shoulder"])
        tip = self._world_to_pixel(*pos["tip"])

        wrist2 = None
        if "wrist2" in pos:
            wrist1 = self._world_to_pixel(*pos["wrist1"])
            wrist2 = self._world_to_pixel(*pos["wrist2"])
        else:
            wrist1 = self._world_to_pixel(*pos["wrist"])

        # Arm segment
        pen = QPen(self.COLOR_ARM, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(shoulder, wrist1)

        if wrist2 is None:
            # Club segment
            pen = QPen(
                self.COLOR_CLUB, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap
            )
            painter.setPen(pen)
            painter.drawLine(wrist1, tip)
        else:
            # Segment 2
            pen = QPen(
                self.COLOR_CLUB, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap
            )
            painter.setPen(pen)
            painter.drawLine(wrist1, wrist2)

            # Segment 3
            pen = QPen(
                self.COLOR_TIP, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap
            )
            painter.setPen(pen)
            painter.drawLine(wrist2, tip)

        # Joint markers
        self._draw_joint(painter, shoulder, 8, self.COLOR_SHOULDER)
        self._draw_joint(painter, wrist1, 6, self.COLOR_WRIST)
        if wrist2 is not None:
            self._draw_joint(painter, wrist2, 5, self.COLOR_WRIST2)
        self._draw_joint(painter, tip, 5, self.COLOR_TIP)

        self._draw_force_vectors(painter, pos)

    def _draw_joint(
        self, painter: QPainter, pos: QPointF, radius: int, color: QColor
    ) -> None:
        """Draw a filled circle at a joint position."""
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(pos, radius, radius)

    def _draw_info(self, painter: QPainter) -> None:
        """Draw time and angle readout in the corner."""
        assert self._result is not None  # only called after None guard in paintEvent
        t = self._result.t[self._current_idx]
        s = self._result.states[self._current_idx]
        theta1_deg = np.degrees(s[0])
        phi_deg = np.degrees(s[1])

        font = QFont("Monospace", 10)
        painter.setFont(font)
        painter.setPen(self.COLOR_TEXT)

        lines = [f"t = {t:.3f} s", f"\u03b81 = {theta1_deg:+.1f}\u00b0"]

        if s.shape[0] >= 6:
            phi2_deg = np.degrees(s[2])
            lines += [
                f"\u03c61 = {phi_deg:+.1f}\u00b0",
                f"\u03c62 = {phi2_deg:+.1f}\u00b0",
                f"d\u03b81 = {s[3]:+.2f} rad/s",
                f"d\u03c61 = {s[4]:+.2f} rad/s",
                f"d\u03c62 = {s[5]:+.2f} rad/s",
            ]
        else:
            lines += [
                f"\u03c6  = {phi_deg:+.1f}\u00b0",
                f"d\u03b81 = {s[2]:+.2f} rad/s",
                f"d\u03c6  = {s[3]:+.2f} rad/s",
            ]

        y = 20
        for line in lines:
            painter.drawText(10, y, line)
            y += 18

    def _draw_placeholder(self, painter: QPainter) -> None:
        """Draw message when no simulation is loaded."""
        font = QFont("Sans", 14)
        painter.setFont(font)
        painter.setPen(self.COLOR_TEXT)
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "Configure parameters\nand click 'Run Simulation'",
        )

    def _draw_force_vectors(self, painter: QPainter, pos: dict) -> None:
        """Draw net force vectors at joints (proximal acting on distal)."""
        if self._result is None or not hasattr(self._result, "joint_forces_at"):
            return
        forces = self._result.joint_forces_at(self._current_idx)
        if not forces:
            return

        magnitudes = [np.hypot(f[0], f[1]) for f in forces.values()]
        max_mag = max(1.0, max(magnitudes))
        scale = 0.3 * self._pixels_per_meter / max_mag

        joint_map = {
            "shoulder": pos.get("shoulder"),
            "wrist": pos.get("wrist"),
            "wrist1": pos.get("wrist1"),
            "wrist2": pos.get("wrist2"),
        }

        painter.setPen(QPen(QColor(200, 240, 120), 2))
        for key, force in forces.items():
            joint_pos = joint_map.get(key)
            if joint_pos is None:
                continue
            fx, fy = force
            end = (
                joint_pos[0] + fx * scale / self._pixels_per_meter,
                joint_pos[1] + fy * scale / self._pixels_per_meter,
            )
            self._draw_arrow(painter, joint_pos, end)

    def _draw_arrow(self, painter: QPainter, origin: tuple, end: tuple) -> None:
        """Draw an arrow from origin to end in world coordinates."""
        p0 = self._world_to_pixel(origin[0], origin[1])
        p1 = self._world_to_pixel(end[0], end[1])
        painter.drawLine(p0, p1)

        dx = p1.x() - p0.x()
        dy = p1.y() - p0.y()
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        ux = dx / length
        uy = dy / length
        size = 8.0

        left = QPointF(
            p1.x() - size * (ux * 0.7 + uy * 0.7),
            p1.y() - size * (uy * 0.7 - ux * 0.7),
        )
        right = QPointF(
            p1.x() - size * (ux * 0.7 - uy * 0.7),
            p1.y() - size * (uy * 0.7 + ux * 0.7),
        )
        painter.drawLine(p1, left)
        painter.drawLine(p1, right)
