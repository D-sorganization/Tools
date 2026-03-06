"""
Custom QWidget that draws the double pendulum animation.

Renders the two segments, joint markers, a tip trail, force vectors,
and an interactive zoom/pan canvas.

New in UI/UX upgrade:
- Mouse-wheel zoom centered on cursor
- Click-drag pan
- Zoom toolbar overlay (zoom in / out / reset)
- show_forces flag gated by external toggle
- Gravity-off visual indicator
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..simulation import SimulationResult


class PendulumWidget(QWidget):
    """Animated zoomable/pannable visualization of the double (or triple) pendulum.

    Controls
    --------
    Scroll wheel   : zoom in / out centred on cursor
    Left drag      : pan the view
    Double-click   : reset zoom & pan
    """

    # Color palette
    COLOR_BG = QColor(16, 16, 28)
    COLOR_ARM = QColor(70, 140, 240)
    COLOR_CLUB = QColor(240, 130, 50)
    COLOR_SHOULDER = QColor(210, 210, 220)
    COLOR_WRIST = QColor(255, 225, 80)
    COLOR_WRIST2 = QColor(120, 225, 185)
    COLOR_TIP = QColor(255, 80, 80)
    COLOR_TRAIL = QColor(255, 80, 80)
    COLOR_GRID = QColor(40, 40, 58)
    COLOR_GRID_MAJOR = QColor(55, 55, 75)
    COLOR_TEXT = QColor(180, 180, 205)
    COLOR_GROUND = QColor(40, 110, 40, 70)
    COLOR_FORCE = QColor(200, 240, 120)
    COLOR_OVERLAY_BG = QColor(25, 25, 42, 200)
    COLOR_NO_GRAVITY = QColor(255, 180, 60)

    TRAIL_LENGTH = 300

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(250, 300)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._result: SimulationResult | None = None
        self._current_idx: int = 0
        self._trail: deque = deque(maxlen=self.TRAIL_LENGTH)
        self._pixels_per_meter: float = 120.0

        # Zoom & pan state
        self._zoom: float = 1.0
        self._pan_x: float = 0.0  # pixel offset
        self._pan_y: float = 0.0
        self._drag_start: QPoint | None = None
        self._drag_pan_start: tuple[float, float] = (0.0, 0.0)

        # Feature toggles
        self._show_forces: bool = False
        self._gravity_on: bool = True

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

    def set_show_forces(self, show: bool) -> None:
        """Toggle force vector overlay."""
        self._show_forces = show
        self.update()

    def set_gravity_on(self, on: bool) -> None:
        """Toggle gravity indicator (visual only — physics uses g from params)."""
        self._gravity_on = on
        self.update()

    # ------------------------------------------------------------------
    # Zoom / Pan — mouse events
    # ------------------------------------------------------------------

    def wheelEvent(self, event: object) -> None:
        from PyQt6.QtGui import QWheelEvent

        if not isinstance(event, QWheelEvent):
            return
        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else (1.0 / 1.15)

        # Zoom centred on cursor position
        cursor_x = event.position().x()
        cursor_y = event.position().y()
        self._pan_x = cursor_x - factor * (cursor_x - self._pan_x)
        self._pan_y = cursor_y - factor * (cursor_y - self._pan_y)
        self._zoom *= factor
        self._zoom = max(0.1, min(20.0, self._zoom))
        self.update()

    def mousePressEvent(self, event: object) -> None:
        if not isinstance(event, QMouseEvent):
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._handle_zoom_button_click(event.pos()):
                return
            self._drag_start = event.pos()
            self._drag_pan_start = (self._pan_x, self._pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: object) -> None:
        if not isinstance(event, QMouseEvent):
            return
        if self._drag_start is not None:
            delta = event.pos() - self._drag_start
            self._pan_x = self._drag_pan_start[0] + delta.x()
            self._pan_y = self._drag_pan_start[1] + delta.y()
            self.update()

    def mouseReleaseEvent(self, event: object) -> None:
        if not isinstance(event, QMouseEvent):
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseDoubleClickEvent(self, event: object) -> None:
        """Double-click resets zoom & pan to default."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _compute_base_scale(self) -> float:
        """Compute base pixels_per_meter ignoring user zoom."""
        if self._result is not None:
            total_len = self._result.params.L1 + self._result.params.L2
        else:
            total_len = 2.0
        w_scale = self.width() * 0.40 / max(total_len, 1e-6)
        h_scale = self.height() * 0.60 / max(total_len, 1e-6)
        return max(30.0, min(w_scale, h_scale))

    def _world_to_pixel(self, x_world: float, y_world: float) -> QPointF:
        """Convert physics coords to widget pixels, applying zoom and pan."""
        base_ppm = self._pixels_per_meter  # already = _compute_base_scale() * zoom
        cx = self.width() / 2.0 + self._pan_x
        cy = self.height() * 0.20 + self._pan_y
        px = cx + x_world * base_ppm
        py = cy - y_world * base_ppm
        return QPointF(px, py)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: object) -> None:
        base_scale = self._compute_base_scale()
        self._pixels_per_meter = base_scale * self._zoom

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.COLOR_BG)

        self._draw_grid(painter)
        self._draw_ground_line(painter)

        if self._result is None:
            self._draw_placeholder(painter)
            self._draw_zoom_controls(painter)
            painter.end()
            return

        self._draw_trail(painter)
        self._draw_pendulum(painter)

        if self._show_forces:
            pos = self._result.positions_at(self._current_idx)
            self._draw_force_vectors(painter, pos)

        self._draw_info(painter)
        self._draw_zoom_controls(painter)

        if not self._gravity_on:
            self._draw_no_gravity_badge(painter)

        painter.end()

    def _draw_grid(self, painter: QPainter) -> None:
        """Draw subtle reference grid."""
        max_range = 4.0
        step_minor = 0.5
        step_major = 1.0

        r = -max_range
        while r <= max_range:
            is_major = abs(r % step_major) < 1e-9
            color = self.COLOR_GRID_MAJOR if is_major else self.COLOR_GRID
            pen = QPen(color, 1 if is_major else 0.5, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            p1 = self._world_to_pixel(-max_range, r)
            p2 = self._world_to_pixel(max_range, r)
            painter.drawLine(p1, p2)
            p1 = self._world_to_pixel(r, max_range)
            p2 = self._world_to_pixel(r, -max_range)
            painter.drawLine(p1, p2)
            r += step_minor

    def _draw_ground_line(self, painter: QPainter) -> None:
        if self._result is None:
            return
        L_total = self._result.params.L1 + self._result.params.L2
        p1 = self._world_to_pixel(-3.5, -L_total)
        p2 = self._world_to_pixel(3.5, -L_total)
        pen = QPen(self.COLOR_GROUND, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

    def _draw_trail(self, painter: QPainter) -> None:
        n = len(self._trail)
        if n < 2:
            return
        for i in range(1, n):
            alpha = int(30 + 180 * (i / n))
            width = 1.0 + 2.5 * (i / n)
            color = QColor(self.COLOR_TRAIL)
            color.setAlpha(alpha)
            pen = QPen(color, width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            x0, y0 = self._trail[i - 1]
            x1, y1 = self._trail[i]
            painter.drawLine(
                self._world_to_pixel(x0, y0),
                self._world_to_pixel(x1, y1),
            )

    def _draw_pendulum(self, painter: QPainter) -> None:
        """Draw the segments and joint markers."""
        assert self._result is not None
        pos = self._result.positions_at(self._current_idx)
        shoulder = self._world_to_pixel(*pos["shoulder"])
        tip = self._world_to_pixel(*pos["tip"])

        wrist2 = None
        if "wrist2" in pos:
            wrist1 = self._world_to_pixel(*pos["wrist1"])
            wrist2 = self._world_to_pixel(*pos["wrist2"])
        else:
            wrist1 = self._world_to_pixel(*pos["wrist"])

        # Arm segment (shoulder → wrist)
        pen = QPen(self.COLOR_ARM, 5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(shoulder, wrist1)

        # Club segment 1 (wrist → wrist2 or tip)
        pen = QPen(self.COLOR_CLUB, 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(wrist1, wrist2 if wrist2 is not None else tip)

        # Club segment 2 (wrist2 → tip) if triple
        if wrist2 is not None:
            pen2 = QPen(self.COLOR_WRIST2, 3)
            pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen2)
            painter.drawLine(wrist2, tip)

        # Joints
        self._draw_joint(painter, shoulder, 8, self.COLOR_SHOULDER)
        self._draw_joint(painter, wrist1, 6, self.COLOR_WRIST)
        if wrist2 is not None:
            self._draw_joint(painter, wrist2, 5, self.COLOR_WRIST2)
        self._draw_joint(painter, tip, 5, self.COLOR_TIP)

    def _draw_joint(
        self, painter: QPainter, pos: QPointF, radius: float, color: QColor
    ) -> None:
        # Glow
        glow = QColor(color)
        glow.setAlpha(60)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(pos, radius * 2, radius * 2)
        # Core
        painter.setBrush(QBrush(color))
        painter.drawEllipse(pos, radius, radius)

    def _draw_info(self, painter: QPainter) -> None:
        assert self._result is not None
        t = self._result.t[self._current_idx]
        s = self._result.states[self._current_idx]
        theta1_deg = np.degrees(s[0])
        phi_deg = np.degrees(s[1])

        painter.setFont(QFont("Monospace", 9))
        painter.setPen(self.COLOR_TEXT)

        lines = [
            f"t = {t:.3f} s",
            f"θ1 = {theta1_deg:+.1f}°",
        ]
        if s.shape[0] >= 6:
            phi2_deg = np.degrees(s[2])
            lines.append(f"φ1 = {phi_deg:+.1f}°")
            lines.append(f"φ2 = {phi2_deg:+.1f}°")
        else:
            lines.append(f"φ = {phi_deg:+.1f}°")

        lines.append(f"zoom {self._zoom:.1f}×")

        y = 18
        for line in lines:
            painter.drawText(8, y, line)
            y += 15

    def _draw_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QColor(80, 80, 110))
        painter.setFont(QFont("Sans", 12))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "Configure parameters\nand click 'Run Simulation'\n\n"
            "Scroll to zoom · Drag to pan · Double-click to reset",
        )

    def _draw_force_vectors(self, painter: QPainter, pos: dict) -> None:
        """Draw net force vectors at joints."""
        if self._result is None or not hasattr(self._result, "joint_forces_at"):
            return
        forces = self._result.joint_forces_at(self._current_idx)
        if not forces:
            return

        magnitudes = [np.hypot(f[0], f[1]) for f in forces.values()]
        max_mag = max(1.0, max(magnitudes))
        scale = 0.4 * self._pixels_per_meter / max_mag

        joint_map = {
            "shoulder": pos.get("shoulder"),
            "wrist": pos.get("wrist"),
            "wrist1": pos.get("wrist1"),
            "wrist2": pos.get("wrist2"),
        }

        painter.setPen(QPen(self.COLOR_FORCE, 2))
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
        p0 = self._world_to_pixel(origin[0], origin[1])
        p1 = self._world_to_pixel(end[0], end[1])
        painter.drawLine(p0, p1)
        # Arrowhead
        dx = p1.x() - p0.x()
        dy = p1.y() - p0.y()
        length = max(1.0, np.hypot(dx, dy))
        ux, uy = dx / length, dy / length
        arrow_len = 8.0
        painter.drawLine(
            p1,
            QPointF(
                p1.x() - arrow_len * (ux + 0.4 * uy),
                p1.y() - arrow_len * (uy - 0.4 * ux),
            ),
        )
        painter.drawLine(
            p1,
            QPointF(
                p1.x() - arrow_len * (ux - 0.4 * uy),
                p1.y() - arrow_len * (uy + 0.4 * ux),
            ),
        )

    def _draw_zoom_controls(self, painter: QPainter) -> None:
        """Draw a small zoom toolbar in the top-right corner."""
        r = self.rect()
        btn_size = 24
        margin = 6
        x = r.right() - btn_size - margin
        y_start = margin

        buttons = [("⊕", "Zoom in"), ("⊖", "Zoom out"), ("⤢", "Reset view")]
        painter.setFont(QFont("Sans", 11))

        for i, (icon, _) in enumerate(buttons):
            bx = x
            by = y_start + i * (btn_size + 3)
            rect = QRect(bx, by, btn_size, btn_size)

            # Background
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.COLOR_OVERLAY_BG))
            painter.drawRoundedRect(rect, 4, 4)

            # Icon
            painter.setPen(QColor(180, 180, 210))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, icon)

        # Store rects for mouse click handling (stored as attribute)
        self._zoom_btn_rects = [
            QRect(x, y_start + i * (btn_size + 3), btn_size, btn_size)
            for i in range(len(buttons))
        ]

    def _draw_no_gravity_badge(self, painter: QPainter) -> None:
        badge_rect = QRect(8, self.height() - 28, 120, 22)
        painter.setPen(Qt.PenStyle.NoPen)
        bg = QColor(60, 40, 0, 180)
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(badge_rect, 4, 4)
        painter.setPen(self.COLOR_NO_GRAVITY)
        painter.setFont(QFont("Sans", 9, QFont.Weight.Bold))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "⚠ Gravity OFF")

    # Handle zoom button clicks
    def _handle_zoom_button_click(self, pos: QPoint) -> bool:
        if not hasattr(self, "_zoom_btn_rects"):
            return False
        for i, rect in enumerate(self._zoom_btn_rects):
            if rect.contains(pos):
                if i == 0:
                    self._zoom = min(20.0, self._zoom * 1.3)
                elif i == 1:
                    self._zoom = max(0.1, self._zoom / 1.3)
                else:
                    self._zoom = 1.0
                    self._pan_x = 0.0
                    self._pan_y = 0.0
                self.update()
                return True
        return False
