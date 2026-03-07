"""
Custom QWidget that draws the golfer upper-body model animation.

Renders the branching topology: hub → two arm chains → shared club,
with joint markers, club tip trail, force vectors, and zoom/pan canvas.

Separate from PendulumWidget to avoid bloating the original widget
and to keep the golfer's branching topology rendering clean.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..simulation_golfer import GolferSimulationResult


class GolferPendulumWidget(QWidget):
    """Animated visualization of the golfer upper-body model.

    Draws the branching arm topology and shared club segment.
    Supports zoom (scroll), pan (drag), and double-click reset.
    """

    # Color palette
    COLOR_BG = QColor(16, 16, 28)
    COLOR_HUB = QColor(180, 180, 200)
    COLOR_RIGHT_ARM = QColor(70, 140, 240)
    COLOR_LEFT_ARM = QColor(120, 200, 140)
    COLOR_CLUB_SHAFT = QColor(240, 180, 50)
    COLOR_CLUBHEAD = QColor(255, 80, 80)
    COLOR_JOINT = QColor(210, 210, 220)
    COLOR_GRIP = QColor(255, 225, 80)
    COLOR_TRAIL = QColor(255, 80, 80)
    COLOR_GRID = QColor(40, 40, 58)
    COLOR_GRID_MAJOR = QColor(55, 55, 75)
    COLOR_TEXT = QColor(180, 180, 205)
    COLOR_FORCE = QColor(200, 240, 120)
    COLOR_SHOULDER_BAR = QColor(140, 140, 160)

    TRAIL_LENGTH = 300

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(250, 300)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._result: GolferSimulationResult | None = None
        self._current_idx: int = 0
        self._trail: deque = deque(maxlen=self.TRAIL_LENGTH)
        self._pixels_per_meter: float = 120.0

        # Zoom & pan
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._drag_start: QPoint | None = None
        self._drag_pan_start: tuple[float, float] = (0.0, 0.0)

        # Feature toggles
        self._show_forces: bool = False
        self._show_zero_torque_forces: bool = False
        self._gravity_on: bool = True
        self._force_scale: float = 1.0
        self._show_mob_ellipsoids: bool = False
        self._show_force_ellipsoids: bool = False
        self._mob_ellipsoid_scale: float = 1.0
        self._force_ellipsoid_scale: float = 1.0

    # ------------------------------------------------------------------
    # Public interface (_SimViewer protocol)
    # ------------------------------------------------------------------

    def set_simulation(self, result: GolferSimulationResult) -> None:
        self._result = result
        self._current_idx = 0
        self._trail.clear()
        self.update()

    def set_frame(self, idx: int) -> None:
        if self._result is None:
            return
        idx = max(0, min(idx, self._result.n_steps - 1))
        pos = self._result.positions_at(idx)
        self._trail.append(pos["club_tip"])
        self._current_idx = idx
        self.update()

    def clear(self) -> None:
        self._result = None
        self._current_idx = 0
        self._trail.clear()
        self.update()

    def set_show_forces(self, show: bool) -> None:
        self._show_forces = show
        self.update()

    def set_show_zero_torque_forces(self, show: bool) -> None:
        self._show_zero_torque_forces = show
        self.update()

    def set_gravity_on(self, on: bool) -> None:
        self._gravity_on = on
        self.update()

    def set_force_scale(self, scale: float) -> None:
        self._force_scale = max(0.01, float(scale))
        self.update()

    def set_show_mob_ellipsoids(self, show: bool) -> None:
        self._show_mob_ellipsoids = show
        self.update()

    def set_show_force_ellipsoids(self, show: bool) -> None:
        self._show_force_ellipsoids = show
        self.update()

    def set_mob_ellipsoid_scale(self, scale: float) -> None:
        self._mob_ellipsoid_scale = max(0.01, float(scale))
        self.update()

    def set_force_ellipsoid_scale(self, scale: float) -> None:
        self._force_ellipsoid_scale = max(0.01, float(scale))
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    # ------------------------------------------------------------------
    # Zoom / Pan
    # ------------------------------------------------------------------

    def wheelEvent(self, event: object) -> None:
        from PyQt6.QtGui import QWheelEvent

        if not isinstance(event, QWheelEvent):
            return
        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else (1.0 / 1.15)
        cx = event.position().x()
        cy = event.position().y()
        self._pan_x = cx - factor * (cx - self._pan_x)
        self._pan_y = cy - factor * (cy - self._pan_y)
        self._zoom *= factor
        self._zoom = max(0.1, min(20.0, self._zoom))
        self.update()

    def mousePressEvent(self, event: object) -> None:
        if not isinstance(event, QMouseEvent):
            return
        if event.button() == Qt.MouseButton.LeftButton:
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
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _compute_base_scale(self) -> float:
        total_len = 2.5  # approximate max reach of golfer
        if self._result is not None:
            p = self._result.params
            total_len = max(
                p.L_hub + p.L_r_upper + p.L_r_fore + p.L_club,
                p.L_hub + p.L_l_upper + p.L_l_fore + p.L_club,
                2.0,
            )
        usable_w = self.width() * 0.42
        usable_h = self.height() * 0.55
        w_scale = usable_w / total_len
        h_scale = usable_h / total_len
        return max(30.0, min(w_scale, h_scale))

    def _world_to_pixel(self, x: float, y: float) -> QPointF:
        ppm = self._pixels_per_meter
        cx = self.width() / 2.0 + self._pan_x
        cy = self.height() * 0.30 + self._pan_y
        return QPointF(cx + x * ppm, cy - y * ppm)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: object) -> None:
        self._pixels_per_meter = self._compute_base_scale() * self._zoom

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.COLOR_BG)

        self._draw_grid(painter)

        if self._result is None:
            self._draw_placeholder(painter)
            painter.end()
            return

        self._draw_trail(painter)
        self._draw_golfer(painter)

        if self._show_forces:
            self._draw_force_vectors(painter)

        self._draw_info(painter)
        painter.end()

    def _draw_grid(self, painter: QPainter) -> None:
        max_range = 4.0
        step = 0.5
        r = -max_range
        while r <= max_range:
            is_major = abs(r % 1.0) < 1e-9
            color = self.COLOR_GRID_MAJOR if is_major else self.COLOR_GRID
            pen = QPen(color, 1 if is_major else 0.5)
            painter.setPen(pen)
            p1 = self._world_to_pixel(-max_range, r)
            p2 = self._world_to_pixel(max_range, r)
            painter.drawLine(p1, p2)
            p1 = self._world_to_pixel(r, max_range)
            p2 = self._world_to_pixel(r, -max_range)
            painter.drawLine(p1, p2)
            r += step

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

    def _draw_golfer(self, painter: QPainter) -> None:
        """Draw the full golfer topology."""
        assert self._result is not None
        pos = self._result.positions_at(self._current_idx)

        origin = self._world_to_pixel(0.0, 0.0)
        hub = self._world_to_pixel(*pos["hub"])
        rs = self._world_to_pixel(*pos["rs"])
        ls = self._world_to_pixel(*pos["ls"])
        re = self._world_to_pixel(*pos["re"])
        le = self._world_to_pixel(*pos["le"])
        rh = self._world_to_pixel(*pos["rh"])
        lh = self._world_to_pixel(*pos["lh"])
        club_base = self._world_to_pixel(*pos["club_base"])
        club_tip = self._world_to_pixel(*pos["club_tip"])

        # Hub standoff (origin → hub)
        pen = QPen(self.COLOR_HUB, 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(origin, hub)

        # Shoulder bar (RS → LS through hub)
        pen = QPen(self.COLOR_SHOULDER_BAR, 3, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(rs, ls)

        # Right arm: RS → RE → RH
        pen = QPen(self.COLOR_RIGHT_ARM, 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(rs, re)
        painter.drawLine(re, rh)

        # Left arm: LS → LE → LH
        pen = QPen(self.COLOR_LEFT_ARM, 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(ls, le)
        painter.drawLine(le, lh)

        # Club shaft
        pen = QPen(self.COLOR_CLUB_SHAFT, 5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(club_base, club_tip)

        # Grip connection lines (hands → club grip points)
        grip_r = self._world_to_pixel(*pos["grip_right"])
        grip_l = self._world_to_pixel(*pos["grip_left"])
        pen = QPen(self.COLOR_GRIP, 2, Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.drawLine(rh, grip_r)
        painter.drawLine(lh, grip_l)

        # Joint markers
        self._draw_joint(painter, origin, 6, self.COLOR_HUB)
        self._draw_joint(painter, hub, 7, self.COLOR_HUB)
        self._draw_joint(painter, rs, 5, self.COLOR_RIGHT_ARM)
        self._draw_joint(painter, re, 5, self.COLOR_RIGHT_ARM)
        self._draw_joint(painter, rh, 5, self.COLOR_GRIP)
        self._draw_joint(painter, ls, 5, self.COLOR_LEFT_ARM)
        self._draw_joint(painter, le, 5, self.COLOR_LEFT_ARM)
        self._draw_joint(painter, lh, 5, self.COLOR_GRIP)
        self._draw_joint(painter, club_tip, 6, self.COLOR_CLUBHEAD)

    def _draw_joint(
        self, painter: QPainter, pos: QPointF, radius: float, color: QColor
    ) -> None:
        glow = QColor(color)
        glow.setAlpha(60)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(pos, radius * 2, radius * 2)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(pos, radius, radius)

    def _draw_force_vectors(self, painter: QPainter) -> None:
        if self._result is None:
            return
        try:
            forces = self._result.joint_forces_at(self._current_idx)
        except Exception:
            return
        if not forces:
            return

        pos = self._result.positions_at(self._current_idx)
        magnitudes = [np.hypot(f[0], f[1]) for f in forces.values()]
        max_mag = max(1.0, max(magnitudes))
        scale = 0.4 * self._pixels_per_meter * self._force_scale / max_mag

        joint_pos_map = {
            "hub": pos.get("hub"),
            "re": pos.get("re"),
            "rh": pos.get("rh"),
            "le": pos.get("le"),
            "lh": pos.get("lh"),
            "club_tip": pos.get("club_tip"),
        }

        painter.setPen(QPen(self.COLOR_FORCE, 2))
        for key, force in forces.items():
            jp = joint_pos_map.get(key)
            if jp is None:
                continue
            fx, fy = force
            end = (
                jp[0] + fx * scale / self._pixels_per_meter,
                jp[1] + fy * scale / self._pixels_per_meter,
            )
            p1 = self._world_to_pixel(*jp)
            p2 = self._world_to_pixel(*end)
            painter.drawLine(p1, p2)

    def _draw_info(self, painter: QPainter) -> None:
        assert self._result is not None
        t = self._result.t[self._current_idx]
        s = self._result.states[self._current_idx]
        theta_deg = np.degrees(s[0])

        painter.setFont(QFont("Monospace", 9))
        painter.setPen(self.COLOR_TEXT)
        lines = [
            f"t = {t:.3f} s",
            f"hub = {theta_deg:+.1f} deg",
            f"zoom {self._zoom:.1f}x",
        ]
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
            "Golfer Upper-Body Model\n\n"
            "Configure parameters and click 'Run Simulation'\n\n"
            "Scroll=zoom  Drag=pan  Double-click=reset",
        )
