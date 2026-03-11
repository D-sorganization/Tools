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
from PyQt6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from ..jacobians import ellipsoids_double, ellipsoids_triple
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
        self._show_zero_torque_forces: bool = False
        self._gravity_on: bool = True
        self._force_scale: float = 1.0
        self._show_mob_ellipsoids: bool = False
        self._show_force_ellipsoids: bool = False
        self._show_com: bool = False

        # Ellipsoid display scales (separate for mobility vs force)
        self._mob_ellipsoid_scale: float = 1.0
        self._force_ellipsoid_scale: float = 1.0

        # Per-segment overlay visibility (#1100, #1101)
        # None = show all segments (default); set[str] = only these joints
        self._visible_segments: set[str] | None = None

        # Swing plane tilt angle in radians (#1113)
        self._tilt_angle: float = 0.0

        # Pre-computed counterfactual forces (list[dict] or None)
        self._zero_torque_forces: list[dict] | None = None

        # Perf: precomputed tip position cache (n_steps × 2)
        self._tip_positions_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_simulation(self, result: SimulationResult) -> None:
        """Load a new simulation result, precompute caches, and reset display."""
        assert result is not None
        self._result = result
        self._current_idx = 0
        self._trail.clear()

        # Vectorized precomputation of tip positions for fast trail/scrubbing
        self._tip_positions_cache = self._precompute_tips(result)

        # Pre-compute zero-torque counterfactual forces for all frames
        self._zero_torque_forces = self._precompute_zero_torque_forces(result)

        self.update()

    @staticmethod
    def _precompute_tips(result: SimulationResult) -> np.ndarray:
        """Vectorised forward kinematics for the tip joint across all frames.

        Returns shape (n_steps, 2) float64 array of (x, y) world coords.
        Bypasses the per-frame Python call overhead.
        """
        states = result.states  # (n, 4) or (n, 6)
        params = result.params
        L1 = params.L1
        L2 = params.L2

        theta1 = states[:, 0]
        phi_or_1 = states[:, 1]

        if states.shape[1] == 6:
            # Triple pendulum: wrist2 is at arm+phi1+phi2
            L3 = getattr(params, "L3", L2)
            phi2 = states[:, 2]
            # wrist1
            wx1 = L1 * np.sin(theta1)
            wy1 = -L1 * np.cos(theta1)
            abs_phi1 = theta1 + phi_or_1
            # wrist2
            wx2 = wx1 + L2 * np.sin(abs_phi1)
            wy2 = wy1 - L2 * np.cos(abs_phi1)
            abs_phi2 = theta1 + phi_or_1 + phi2
            tx = wx2 + L3 * np.sin(abs_phi2)
            ty = wy2 - L3 * np.cos(abs_phi2)
        else:
            # Double pendulum
            abs_angle2 = theta1 + phi_or_1
            wx = L1 * np.sin(theta1)
            wy = -L1 * np.cos(theta1)
            tx = wx + L2 * np.sin(abs_angle2)
            ty = wy - L2 * np.cos(abs_angle2)

        return np.column_stack([tx, ty])

    @staticmethod
    def _precompute_zero_torque_forces(
        result: SimulationResult,
    ) -> list[dict]:
        """Pre-compute zero-torque counterfactual joint forces for every frame.

        Runs the passive dynamics at each state so rendering never calls it
        per-frame during animation.  Uses vectorised loop over states.

        Returns list of force dicts (one per step), same keys as joint_forces_at.
        """
        from ..counterfactual import (
            zero_torque_joint_forces_double,
            zero_torque_joint_forces_triple,
        )

        forces: list[dict] = []
        params = result.params
        for state in result.states:
            try:
                if state.shape[0] >= 6:
                    forces.append(zero_torque_joint_forces_triple(state, params))  # type: ignore[arg-type]
                else:
                    forces.append(zero_torque_joint_forces_double(state, params))
            except Exception:
                # Fallback: empty dict so rendering skips gracefully
                forces.append({})
        return forces

    def set_frame(self, idx: int) -> None:
        """Advance to frame idx and update the trail."""
        if self._result is None:
            return
        idx = max(0, min(idx, self._result.n_steps - 1))
        # Use cached tip positions for O(1) trail append
        if self._tip_positions_cache is not None:
            self._trail.append(tuple(self._tip_positions_cache[idx]))
        else:
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

    def set_show_zero_torque_forces(self, show: bool) -> None:
        """Toggle zero-torque counterfactual force vector overlay."""
        self._show_zero_torque_forces = show
        self.update()

    def set_gravity_on(self, on: bool) -> None:
        """Toggle gravity indicator (visual only — physics uses g from params)."""
        self._gravity_on = on
        self.update()

    def set_force_scale(self, scale: float) -> None:
        """Set the display scale multiplier for force vectors."""
        self._force_scale = max(0.01, float(scale))
        self.update()

    def set_show_mob_ellipsoids(self, show: bool) -> None:
        """Toggle mobility ellipsoid overlay at segment endpoints."""
        self._show_mob_ellipsoids = show
        self.update()

    def set_show_force_ellipsoids(self, show: bool) -> None:
        """Toggle force ellipsoid overlay at segment endpoints."""
        self._show_force_ellipsoids = show
        self.update()

    def set_mob_ellipsoid_scale(self, scale: float) -> None:
        """Set the display scale for mobility ellipsoids."""
        assert scale > 0, "Ellipsoid scale must be positive"
        self._mob_ellipsoid_scale = float(scale)
        self.update()

    def set_force_ellipsoid_scale(self, scale: float) -> None:
        """Set the display scale for force ellipsoids."""
        assert scale > 0, "Ellipsoid scale must be positive"
        self._force_ellipsoid_scale = float(scale)
        self.update()

    def set_show_com(self, show: bool) -> None:
        """Toggle combined center of mass display."""
        self._show_com = show
        self.update()

    def set_visible_segments(self, segments: set[str] | None) -> None:
        """Set which joint segments are visible for overlays.

        Parameters
        ----------
        segments : set[str] or None
            Joint names to show overlays for (e.g. {"wrist", "tip"}).
            None means show all segments.

        Closes #1100, #1101.
        """
        self._visible_segments = segments
        self.update()

    def set_tilt_angle(self, angle_rad: float) -> None:
        """Set the swing plane tilt angle for display projection.

        Parameters
        ----------
        angle_rad : float
            Tilt from vertical in radians (0 = vertical, π/2 = horizontal).

        Closes #1113.
        """
        self._tilt_angle = float(angle_rad)
        self.update()

    def reset_view(self) -> None:
        """Reset zoom and pan to default (also callable from toolstrip button)."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
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
        """Compute base pixels_per_meter ignoring user zoom.

        Shoulder is placed at 35% from the widget top.  The pendulum can swing
        through a full circle of radius total_len, so we allow a 2*total_len
        diameter to fit comfortably.  We use 45% of width and 55% of the
        remaining height (below shoulder) so there is always room regardless
        of how the pendulum is configured.
        """
        if self._result is not None:
            total_len = self._result.params.L1 + self._result.params.L2
        else:
            total_len = 2.0
        total_len = max(total_len, 1e-6)
        # Available width/height for full swing circle (diameter = 2 * total_len)
        usable_w = self.width() * 0.42
        usable_h = self.height() * 0.55
        w_scale = usable_w / total_len
        h_scale = usable_h / total_len
        return max(30.0, min(w_scale, h_scale))

    def _world_to_pixel(self, x_world: float, y_world: float) -> QPointF:
        """Convert physics coords to widget pixels, applying zoom, pan, and tilt.

        Shoulder is placed at horizontal centre and 35% from the top so
        full-circle swings remain on-screen without the user needing to pan.

        When a tilt angle is set (#1113), the Y axis is foreshortened by
        cos(tilt) to simulate viewing the pendulum on a tilted plane.
        """
        base_ppm = self._pixels_per_meter
        cx = self.width() / 2.0 + self._pan_x
        cy = self.height() * 0.35 + self._pan_y  # 35% from top (was 20%)
        # Apply tilt foreshortening to Y axis only
        y_projected = y_world * float(np.cos(self._tilt_angle))
        px = cx + x_world * base_ppm
        py = cy - y_projected * base_ppm
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

        if self._show_zero_torque_forces:
            pos = self._result.positions_at(self._current_idx)
            self._draw_zero_torque_force_vectors(painter, pos)

        if self._show_mob_ellipsoids or self._show_force_ellipsoids:
            self._draw_ellipsoids_at_frame(painter)

        if self._show_com:
            self._draw_com(painter)

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
            # Per-segment visibility filter (#1100)
            if self._visible_segments is not None and key not in self._visible_segments:
                continue
            joint_pos = joint_map.get(key)
            if joint_pos is None:
                continue
            fx, fy = force
            end = (
                joint_pos[0] + fx * scale / self._pixels_per_meter,
                joint_pos[1] + fy * scale / self._pixels_per_meter,
            )
            self._draw_arrow(painter, joint_pos, end)

    # Colour for zero-torque (passive drift) force vectors — distinct from
    # regular force vectors (COLOR_FORCE, typically yellow/orange)
    COLOR_ZERO_TORQUE = QColor(210, 120, 255)  # violet/purple

    def _draw_zero_torque_force_vectors(self, painter: QPainter, pos: dict) -> None:
        """Draw zero-torque (passive drift) force vectors at each joint.

        Dashed purple arrows show the joint forces that would exist if all
        applied driving torques were zero at this instant.  The same force
        scale factor as regular force vectors is used so magnitudes are
        directly visually comparable.
        """
        if self._zero_torque_forces is None or not self._zero_torque_forces:
            return
        forces = self._zero_torque_forces[self._current_idx]
        if not forces:
            return

        magnitudes = [np.hypot(f[0], f[1]) for f in forces.values()]
        max_mag = max(1.0, max(magnitudes))
        scale = 0.4 * self._pixels_per_meter * self._force_scale / max_mag

        joint_map = {
            "shoulder": pos.get("shoulder"),
            "wrist": pos.get("wrist"),
            "wrist1": pos.get("wrist1"),
            "wrist2": pos.get("wrist2"),
        }

        pen = QPen(self.COLOR_ZERO_TORQUE, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for key, force in forces.items():
            # Per-segment visibility filter (#1100)
            if self._visible_segments is not None and key not in self._visible_segments:
                continue
            joint_pos = joint_map.get(key)
            if joint_pos is None:
                continue
            fx, fy = force
            end = (
                joint_pos[0] + fx * scale / self._pixels_per_meter,
                joint_pos[1] + fy * scale / self._pixels_per_meter,
            )
            self._draw_arrow(painter, joint_pos, end)

    # ------------------------------------------------------------------
    # Ellipsoid drawing
    # ------------------------------------------------------------------

    # Color constants for ellipsoids
    COLOR_MOB_ELLIPSOID = QColor(100, 200, 255, 70)  # translucent cyan
    COLOR_MOB_OUTLINE = QColor(100, 200, 255, 180)
    COLOR_FORCE_ELLIPSOID = QColor(255, 160, 80, 60)  # translucent orange
    COLOR_FORCE_OUTLINE = QColor(255, 160, 80, 180)

    def _draw_ellipsoids_at_frame(self, painter: QPainter) -> None:
        """Compute and draw mobility/force ellipsoids for the current frame."""
        assert self._result is not None
        state = self._result.states[self._current_idx]
        params = self._result.params
        ppm = self._pixels_per_meter

        if state.shape[0] >= 6:
            # Triple pendulum
            theta1, phi1, phi2 = float(state[0]), float(state[1]), float(state[2])
            data = ellipsoids_triple(
                theta1,
                phi1,
                phi2,
                params.L1,
                params.L2,
                params.L3,  # type: ignore[attr-defined]
            )
            pos = self._result.positions_at(self._current_idx)
            endpoint_map = {
                "wrist1": pos.get("wrist1", pos.get("wrist", (0.0, 0.0))),
                "wrist2": pos.get("wrist2", (0.0, 0.0)),
                "tip": pos["tip"],
            }
        else:
            # Double pendulum
            theta1, phi = float(state[0]), float(state[1])
            data = ellipsoids_double(theta1, phi, params.L1, params.L2)
            pos = self._result.positions_at(self._current_idx)
            endpoint_map = {
                "wrist": pos.get("wrist", (0.0, 0.0)),
                "tip": pos["tip"],
            }

        for name, ell in data.items():
            # Per-segment visibility filter (#1100)
            if self._visible_segments is not None and name not in self._visible_segments:
                continue
            world_pos = endpoint_map.get(name)
            if world_pos is None:
                continue
            cx_px, cy_px = (
                self._world_to_pixel(*world_pos).x(),
                self._world_to_pixel(*world_pos).y(),
            )

            dirs = ell["directions"]  # (2,2), columns are principal axes
            if self._show_mob_ellipsoids:
                mob = ell["mob_semi_axes"]  # (2,) in physics units
                mob_scale = self._mob_ellipsoid_scale * ppm * 0.3
                self._draw_ellipse_axes(
                    painter,
                    cx_px,
                    cy_px,
                    dirs,
                    mob * mob_scale,
                    fill=self.COLOR_MOB_ELLIPSOID,
                    outline=self.COLOR_MOB_OUTLINE,
                    label="M",
                )
            if self._show_force_ellipsoids and ell["force_semi_axes"] is not None:
                force = ell["force_semi_axes"]  # (2,)
                force_scale = self._force_ellipsoid_scale * ppm * 0.3
                self._draw_ellipse_axes(
                    painter,
                    cx_px,
                    cy_px,
                    dirs,
                    force * force_scale,
                    fill=self.COLOR_FORCE_ELLIPSOID,
                    outline=self.COLOR_FORCE_OUTLINE,
                    label="F",
                )

    def _draw_ellipse_axes(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        directions: np.ndarray,
        semi_axes_px: np.ndarray,
        fill: QColor,
        outline: QColor,
        label: str = "",
    ) -> None:
        """Draw a 2-D ellipse defined by principal axes and semi-axis lengths (pixels).

        Contract
        --------
        directions : (2, 2) orthonormal matrix; columns are principal axes.
        semi_axes_px : (2,) non-negative pixel semi-axis lengths.
        """
        assert directions.shape == (2, 2), "directions must be (2, 2)"
        assert semi_axes_px.shape == (2,), "semi_axes_px must be (2,)"

        a = float(semi_axes_px[0])  # major semi-axis (pixels)
        b = float(semi_axes_px[1])  # minor semi-axis (pixels)

        # Clamp to sensible display range
        max_px = min(self.width(), self.height()) * 0.8
        a = max(2.0, min(a, max_px))
        b = max(2.0, min(b, max_px))

        # Principal-axis direction (column 0 of U = major axis)
        dx, dy = float(directions[0, 0]), float(directions[1, 0])
        # In screen coords: y is flipped, so negate dy
        angle_deg = float(np.degrees(np.arctan2(-dy, dx)))

        painter.save()
        painter.translate(cx, cy)
        painter.rotate(angle_deg)
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(outline, 1.2))
        # drawEllipse(x, y, w, h) where x,y is top-left corner
        painter.drawEllipse(
            QPointF(0.0, 0.0),
            a,  # half-width  = major semi-axis
            b,  # half-height = minor semi-axis
        )
        painter.restore()

        # Small label at ellipse edge
        if label:
            lbl_x = cx + float(directions[0, 0]) * a + 4
            lbl_y = cy - float(directions[1, 0]) * a
            painter.setFont(QFont("Monospace", 7))
            painter.setPen(outline)
            painter.drawText(QPointF(lbl_x, lbl_y), label)

    def _draw_arrow(self, painter: QPainter, origin: tuple, end: tuple) -> None:
        """Draw a force/torque vector with a filled triangular arrowhead."""
        p0 = self._world_to_pixel(origin[0], origin[1])
        p1 = self._world_to_pixel(end[0], end[1])
        painter.drawLine(p0, p1)

        # Arrowhead — filled triangle
        dx = p1.x() - p0.x()
        dy = p1.y() - p0.y()
        length = max(1.0, np.hypot(dx, dy))
        ux, uy = dx / length, dy / length
        arrow_len = 10.0
        arrow_w = 4.0

        tip = p1
        left = QPointF(
            p1.x() - arrow_len * ux + arrow_w * uy,
            p1.y() - arrow_len * uy - arrow_w * ux,
        )
        right = QPointF(
            p1.x() - arrow_len * ux - arrow_w * uy,
            p1.y() - arrow_len * uy + arrow_w * ux,
        )

        path = QPainterPath()
        path.moveTo(tip)
        path.lineTo(left)
        path.lineTo(right)
        path.closeSubpath()

        old_brush = painter.brush()
        painter.setBrush(QBrush(painter.pen().color()))
        painter.drawPath(path)
        painter.setBrush(old_brush)

    # ------------------------------------------------------------------
    # Center of Mass drawing
    # ------------------------------------------------------------------

    COLOR_COM = QColor(255, 255, 80)  # bright yellow

    def _draw_com(self, painter: QPainter) -> None:
        """Draw the combined center of mass of the system."""
        if self._result is None:
            return

        state = self._result.states[self._current_idx]
        params = self._result.params

        # Compute COM from all mass point positions
        if state.shape[0] >= 6:
            # Triple pendulum
            pos = self._result.positions_at(self._current_idx)
            masses = [params.m1, params.m2, getattr(params, "m3", params.m2)]
            wrist1 = np.array(pos.get("wrist1", pos.get("wrist", (0, 0))))
            wrist2 = np.array(pos.get("wrist2", (0, 0)))
            tip = np.array(pos["tip"])
            # Approximate: mass at midpoint of each segment
            shoulder = np.array(pos["shoulder"])
            com1 = 0.5 * (shoulder + wrist1)
            com2 = 0.5 * (wrist1 + wrist2)
            com3 = 0.5 * (wrist2 + tip)
            total_m = sum(masses)
            com = (masses[0] * com1 + masses[1] * com2 + masses[2] * com3) / total_m
        else:
            # Double pendulum
            theta1 = state[0]
            phi = state[1]
            abs2 = theta1 + phi
            # Mass at midpoint of each segment
            c1x = 0.5 * params.L1 * np.sin(theta1)
            c1y = -0.5 * params.L1 * np.cos(theta1)
            wx = params.L1 * np.sin(theta1)
            wy = -params.L1 * np.cos(theta1)
            c2x = wx + 0.5 * params.L2 * np.sin(abs2)
            c2y = wy - 0.5 * params.L2 * np.cos(abs2)
            total_m = params.m1 + params.m2
            com = np.array([
                (params.m1 * c1x + params.m2 * c2x) / total_m,
                (params.m1 * c1y + params.m2 * c2y) / total_m,
            ])

        com_px = self._world_to_pixel(float(com[0]), float(com[1]))

        # Draw COM marker: crosshair + circle
        r = 6
        painter.setPen(QPen(self.COLOR_COM, 2))
        painter.setBrush(QBrush(QColor(255, 255, 80, 100)))
        painter.drawEllipse(com_px, r, r)
        # Crosshair lines
        painter.drawLine(
            QPointF(com_px.x() - r * 1.5, com_px.y()),
            QPointF(com_px.x() + r * 1.5, com_px.y()),
        )
        painter.drawLine(
            QPointF(com_px.x(), com_px.y() - r * 1.5),
            QPointF(com_px.x(), com_px.y() + r * 1.5),
        )

        # Label
        painter.setFont(QFont("Monospace", 7))
        painter.drawText(QPointF(com_px.x() + r + 2, com_px.y() - 2), "COM")

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
