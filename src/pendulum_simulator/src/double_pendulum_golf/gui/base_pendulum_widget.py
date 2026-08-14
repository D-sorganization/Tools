# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""
BasePendulumWidget — shared base class for all pendulum visualization widgets.

Extracts common state, feature toggles, zoom/pan interaction, coordinate
mapping, grid drawing, trail rendering, and joint/COM drawing that were
previously duplicated between PendulumWidget and GolferPendulumWidget.

Design by Contract
------------------
- Pre:  ``set_force_scale(s)`` requires ``s > 0``.
- Pre:  ``set_*_ellipsoid_scale(s)`` requires ``s > 0``.
- Inv:  ``_zoom`` is always in [0.1, 20.0].
- Inv:  ``_pixels_per_meter`` is always >= 30.0 after ``paintEvent``.

Closes DRY violation between PendulumWidget and GolferPendulumWidget.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections import deque

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


class BasePendulumWidget(QWidget):
    """Abstract base for zoomable/pannable pendulum visualizations.

    Subclasses must implement:
    - ``_get_total_length() -> float``:  max reach of the current model
    - ``_draw_model(painter)``:  draw segments + joints for the specific model
    - ``_draw_info(painter)``:  draw frame-specific text overlay
    - ``_draw_placeholder(painter)``:  draw "no data" message
    - ``_has_result() -> bool``:  whether simulation data is loaded
    """

    # ── Shared color palette ──────────────────────────────────────────
    COLOR_BG = QColor(16, 16, 28)
    COLOR_TRAIL = QColor(255, 80, 80)
    COLOR_GRID = QColor(40, 40, 58)
    COLOR_GRID_MAJOR = QColor(55, 55, 75)
    COLOR_TEXT = QColor(180, 180, 205)
    COLOR_GROUND = QColor(40, 110, 40, 70)
    COLOR_FORCE = QColor(200, 240, 120)
    COLOR_NO_GRAVITY = QColor(255, 180, 60)
    COLOR_COM = QColor(255, 255, 80)
    COLOR_TILT_PLANE = QColor(80, 180, 80, 40)

    TRAIL_LENGTH = 300

    # Number of Catmull-Rom subdivisions per trail segment (#1116)
    SPLINE_SUBDIV = 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(250, 300)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._pixels_per_meter: float = 120.0
        self._trail: deque = deque(maxlen=self.TRAIL_LENGTH)

        # Zoom & pan state
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._drag_start: QPoint | None = None
        self._drag_pan_start: tuple[float, float] = (0.0, 0.0)

        # Bulletproof view fitting (#anim-blank-fix)
        # Trajectory bounding box in WORLD coordinates: (xmin, xmax, ymin, ymax).
        # Set by `compute_and_store_trajectory_bbox()` (called from subclass
        # set_simulation), used by auto_fit_view() and _compute_base_scale().
        self._trajectory_bbox: tuple[float, float, float, float] | None = None
        # Center of the world view, in world coordinates. When None, the
        # widget falls back to the legacy "anchor at top of total length"
        # behaviour driven by `_shoulder_y_fraction`.
        self._view_center_world: tuple[float, float] | None = None
        # Sticky flag: when True, every paint reapplies auto-fit. Cleared
        # the moment the user pans, zooms, or rotates manually so user
        # intent always wins, but reset_view()/double-click puts it back.
        self._auto_fit_locked: bool = True

        # 3D rotation state (right-click drag)
        self._rotate_start: QPoint | None = None

        # Feature toggles (shared across all model types)
        self._show_forces: bool = False
        self._show_zero_torque_forces: bool = False
        self._gravity_on: bool = True
        self._force_scale: float = 1.0
        self._show_mob_ellipsoids: bool = False
        self._show_force_ellipsoids: bool = False
        self._show_com: bool = False

        # Torque / moment vector display (#1208)
        self._show_torque_vectors: bool = False
        self._show_moment_of_force: bool = False
        self._show_sum_moments: bool = False

        # Ellipsoid display scales
        self._mob_ellipsoid_scale: float = 1.0
        self._force_ellipsoid_scale: float = 1.0

        # Per-segment overlay visibility (#1100, #1101, #1102)
        self._visible_segments: set[str] | None = None

        # Swing plane tilt angle in radians (#1113)
        self._tilt_angle: float = 0.0

        # View azimuth rotation in radians (#1118)
        self._view_azimuth: float = 0.0

        # 3D segment rendering mode (#1155)
        self._3d_mode: bool = False

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_total_length(self) -> float:
        """Return the maximum reach of the model for auto-scaling."""
        ...

    @abstractmethod
    def _draw_model(self, painter: QPainter) -> None:
        """Draw the pendulum segments, joints, and model-specific overlays."""
        ...

    @abstractmethod
    def _draw_info(self, painter: QPainter) -> None:
        """Draw frame info text overlay (time, angles, zoom)."""
        ...

    @abstractmethod
    def _draw_placeholder(self, painter: QPainter) -> None:
        """Draw placeholder text when no simulation is loaded."""
        ...

    @abstractmethod
    def _has_result(self) -> bool:
        """Return True if simulation data is currently loaded."""
        ...

    # ------------------------------------------------------------------
    # Public feature-toggle interface (shared protocol)
    # ------------------------------------------------------------------

    def set_show_forces(self, show: bool) -> None:
        """Toggle force vector overlay."""
        if show is None:
            raise ValueError("show must be provided")
        self._show_forces = bool(show)
        self.update()

    def set_show_zero_torque_forces(self, show: bool) -> None:
        """Toggle zero-torque counterfactual force vector overlay."""
        if show is None:
            raise ValueError("show must be provided")
        self._show_zero_torque_forces = bool(show)
        self.update()

    def set_gravity_on(self, on: bool) -> None:
        """Toggle gravity indicator (visual only)."""
        if on is None:
            raise ValueError("on must be provided")
        self._gravity_on = bool(on)
        self.update()

    def set_force_scale(self, scale: float) -> None:
        """Set display scale multiplier for force vectors.

        Pre: scale > 0
        """
        if not (scale > 0):
            raise ValueError(f"Force scale must be positive, got {scale}")
        self._force_scale = float(scale)
        self.update()

    def set_show_mob_ellipsoids(self, show: bool) -> None:
        """Toggle display of manipulability ellipsoids."""
        if show is None:
            raise ValueError("show must be provided")
        self._show_mob_ellipsoids = bool(show)
        self.update()

    def set_show_force_ellipsoids(self, show: bool) -> None:
        """Toggle display of force ellipsoids."""
        if show is None:
            raise ValueError("show must be provided")
        self._show_force_ellipsoids = bool(show)
        self.update()

    def set_mob_ellipsoid_scale(self, scale: float) -> None:
        """Pre: scale > 0"""
        if not (scale > 0):
            raise ValueError(f"Ellipsoid scale must be positive, got {scale}")
        self._mob_ellipsoid_scale = float(scale)
        self.update()

    def set_force_ellipsoid_scale(self, scale: float) -> None:
        """Pre: scale > 0"""
        if not (scale > 0):
            raise ValueError(f"Ellipsoid scale must be positive, got {scale}")
        self._force_ellipsoid_scale = float(scale)
        self.update()

    def set_show_com(self, show: bool) -> None:
        """Toggle display of centre-of-mass markers."""
        if show is None:
            raise ValueError("show must be provided")
        self._show_com = bool(show)
        self.update()

    def set_visible_segments(self, segments: set[str] | None) -> None:
        """Set which joint segments show overlays (None = all)."""
        self._visible_segments = segments
        self.update()

    def set_tilt_angle(self, angle_rad: float) -> None:
        """Set swing plane tilt for display projection (#1113)."""
        if angle_rad is None:
            raise ValueError("angle_rad must be provided")
        self._tilt_angle = float(angle_rad)
        self.update()

    def set_view_azimuth(self, angle_rad: float) -> None:
        """Set view azimuth for canvas rotation (#1118)."""
        if angle_rad is None:
            raise ValueError("angle_rad must be provided")
        self._view_azimuth = float(angle_rad)
        self.update()

    def set_show_torque_vectors(self, show: bool) -> None:
        """Toggle torque vector display at each joint (#1208)."""
        if show is None:
            raise ValueError("show must be provided")
        self._show_torque_vectors = bool(show)
        self.update()

    def set_show_moment_of_force(self, show: bool) -> None:
        """Toggle moment-of-force (proximal-on-distal) vector display (#1208)."""
        if show is None:
            raise ValueError("show must be provided")
        self._show_moment_of_force = bool(show)
        self.update()

    def set_show_sum_moments(self, show: bool) -> None:
        """Toggle sum-of-moments (resultant) vector display (#1208)."""
        if show is None:
            raise ValueError("show must be provided")
        self._show_sum_moments = bool(show)
        self.update()

    def reset_view(self) -> None:
        """Reset zoom and pan, then auto-fit to the trajectory if loaded.

        This is the canonical "I cannot find the system" recovery: it
        clears any user pan/zoom/orbit, re-locks auto-fit, and forces a
        fresh fit on the next paint. With or without simulation data the
        result is always a sane, in-view default.
        """
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._tilt_angle = 0.0
        self._view_azimuth = 0.0
        self._auto_fit_locked = True
        # Force a fresh recompute on the next paint by clearing the
        # cached center; the bbox itself stays so we don't pay to rescan.
        self._view_center_world = None
        self.update()

    def auto_fit_view(self) -> None:
        """Compute pan/zoom so the entire trajectory bbox fits with margin.

        Safe to call from any thread that holds the GUI lock. Idempotent.
        Falls back to a centered default when no bbox is known.
        """
        # Reset interaction state so the fit is the only contributor.
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._auto_fit_locked = True

        bbox = self._trajectory_bbox
        if bbox is None:
            # No trajectory loaded yet — center the world origin in the
            # widget so the placeholder + ground line are visible.
            self._view_center_world = (0.0, 0.0)
            self.update()
            return

        xmin, xmax, ymin, ymax = bbox
        # Center the trajectory bbox; the paint pass picks the scale.
        self._view_center_world = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
        self.update()

    def compute_and_store_trajectory_bbox(
        self,
        sample_positions: list[dict[str, tuple[float, float] | None]],
    ) -> None:
        """Scan a list of per-frame joint dicts and cache the world bbox.

        Subclasses call this from ``set_simulation`` so the next paint
        and any subsequent ``auto_fit_view`` have a precomputed bbox.

        Pre:  ``sample_positions`` is non-empty and every dict has at
              least one (x, y) tuple.
        Post: ``self._trajectory_bbox`` is set to a finite, non-degenerate
              bbox, and ``auto_fit_view()`` has been called.
        """
        if not sample_positions:
            self._trajectory_bbox = None
            self.auto_fit_view()
            return

        xs: list[float] = []
        ys: list[float] = []
        # Always include the world origin (the standoff anchor) so the
        # pivot is in view even if the joints swing far from it.
        xs.append(0.0)
        ys.append(0.0)
        for pos in sample_positions:
            for value in pos.values():
                if value is None:
                    continue
                try:
                    x, y = float(value[0]), float(value[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                xs.append(x)
                ys.append(y)

        if len(xs) < 2:
            self._trajectory_bbox = None
            self.auto_fit_view()
            return

        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)

        # Pad degenerate dimensions so we never divide by zero in the
        # scale computation. 0.5m of slack is well below pendulum reach.
        if xmax - xmin < 1e-3:
            xmax = xmin + 0.5
        if ymax - ymin < 1e-3:
            ymax = ymin + 0.5

        self._trajectory_bbox = (xmin, xmax, ymin, ymax)
        self.auto_fit_view()

    def is_view_auto_fit(self) -> bool:
        """Return True iff the view is in auto-fit mode (no manual pan/zoom)."""
        return self._auto_fit_locked

    def _release_auto_fit(self) -> None:
        """Drop auto-fit lock — called from any user manipulation."""
        self._auto_fit_locked = False

    # ------------------------------------------------------------------
    # Zoom / Pan — mouse events
    # ------------------------------------------------------------------

    def wheelEvent(self, event: object) -> None:
        """Handle mouse wheel for zoom, centered on cursor position."""
        if event is None:
            raise ValueError("event must be provided")
        from PyQt6.QtGui import QWheelEvent

        if not isinstance(event, QWheelEvent):
            return
        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else (1.0 / 1.15)

        cursor_x = event.position().x()
        cursor_y = event.position().y()
        self._pan_x = cursor_x - factor * (cursor_x - self._pan_x)
        self._pan_y = cursor_y - factor * (cursor_y - self._pan_y)
        self._zoom *= factor
        self._zoom = max(0.1, min(20.0, self._zoom))
        self._release_auto_fit()
        self.update()

    def mousePressEvent(self, event: object) -> None:
        """Begin pan (left-click) or orbit (right-click) drag interaction."""
        if event is None:
            raise ValueError("event must be provided")
        if not isinstance(event, QMouseEvent):
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if hasattr(
                self, "_handle_zoom_button_click"
            ) and self._handle_zoom_button_click(event.pos()):
                return
            self._drag_start = event.pos()
            self._drag_pan_start = (self._pan_x, self._pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self._rotate_start = event.pos()
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseMoveEvent(self, event: object) -> None:
        """Continue pan or orbit drag, updating view transform."""
        if event is None:
            raise ValueError("event must be provided")
        if not isinstance(event, QMouseEvent):
            return
        if self._drag_start is not None:
            delta = event.pos() - self._drag_start
            self._pan_x = self._drag_pan_start[0] + delta.x()
            self._pan_y = self._drag_pan_start[1] + delta.y()
            self._release_auto_fit()
            self.update()
        elif self._rotate_start is not None:
            delta = event.pos() - self._rotate_start
            sensitivity = 0.01  # radians per pixel
            self._view_azimuth += delta.x() * sensitivity
            self._tilt_angle += delta.y() * sensitivity
            # Clamp tilt to [-pi/2, pi/2]
            max_tilt = float(np.pi / 2)
            self._tilt_angle = max(-max_tilt, min(max_tilt, self._tilt_angle))
            self._release_auto_fit()
            self.update()

    def mouseReleaseEvent(self, event: object) -> None:
        """End drag interaction and restore the default cursor."""
        if event is None:
            raise ValueError("event must be provided")
        if not isinstance(event, QMouseEvent):
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self._rotate_start = None
            self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseDoubleClickEvent(self, event: object) -> None:
        """Double-click resets zoom & pan."""
        self.reset_view()

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _shoulder_y_fraction(self) -> float:
        """Fraction from top where the anchor point is placed.

        Override in subclasses for different anchor positioning.
        """
        return 0.35

    def _compute_base_scale(self) -> float:
        """Compute base pixels_per_meter ignoring user zoom.

        When a trajectory bbox is available, the scale fits the bbox
        into the widget with a 12 % margin on every side. Otherwise it
        falls back to the legacy ``_get_total_length`` heuristic.

        Post: result >= 30.0 (so the widget never collapses to nothing)
        """
        margin = 0.88  # leave 12% padding around the bbox

        bbox = self._trajectory_bbox
        if bbox is not None:
            xmin, xmax, ymin, ymax = bbox
            world_w = max(xmax - xmin, 1e-3)
            world_h = max(ymax - ymin, 1e-3)
            usable_w = max(self.width(), 1) * margin
            usable_h = max(self.height(), 1) * margin
            w_scale = usable_w / world_w
            h_scale = usable_h / world_h
            return max(30.0, min(w_scale, h_scale))

        total_len = max(self._get_total_length(), 1e-6)
        usable_w = self.width() * 0.42
        usable_h = self.height() * 0.55
        w_scale = usable_w / total_len
        h_scale = usable_h / total_len
        result = max(30.0, min(w_scale, h_scale))
        if not (result >= 30.0):
            raise ValueError("DbC Blocked: Precondition failed.")
        return result

    def _world_to_pixel(self, x_world: float, y_world: float) -> QPointF:
        """Convert physics coords to widget pixels with 3D projection.

        Applies azimuth rotation (#1118) and tilt foreshortening (#1113).
        When a ``_view_center_world`` is set (auto-fit mode), the world
        is centered on it; otherwise the legacy anchor-at-top behaviour
        is used so existing presets continue to look correct.
        """
        if x_world is None:
            raise ValueError("x_world must be provided")
        base_ppm = self._pixels_per_meter

        center = self._view_center_world
        if center is not None:
            cx = self.width() / 2.0 + self._pan_x
            cy = self.height() / 2.0 + self._pan_y
            x_local = x_world - center[0]
            y_local = y_world - center[1]
        else:
            cx = self.width() / 2.0 + self._pan_x
            cy = self.height() * self._shoulder_y_fraction() + self._pan_y
            x_local = x_world
            y_local = y_world

        cos_az = float(np.cos(self._view_azimuth))
        sin_az = float(np.sin(self._view_azimuth))
        x_rot = x_local * cos_az
        depth = x_local * sin_az

        cos_tilt = float(np.cos(self._tilt_angle))
        y_proj = y_local * cos_tilt - depth * float(np.sin(self._tilt_angle))

        px = cx + x_rot * base_ppm
        py = cy - y_proj * base_ppm
        return QPointF(px, py)

    # ------------------------------------------------------------------
    # Off-screen detection / recovery overlay
    # ------------------------------------------------------------------

    def _world_points_in_view(
        self, points: list[tuple[float, float]]
    ) -> tuple[bool, QPointF]:
        """Check if any of the given world points lies inside the widget.

        Returns ``(any_visible, centroid_pixel)`` where the centroid is
        the average pixel position of all the input points (used for
        drawing the off-screen indicator arrow).
        """
        rect = self.rect()
        n = len(points)
        if n == 0:
            return True, QPointF(rect.center())
        sum_x = 0.0
        sum_y = 0.0
        any_visible = False
        for x, y in points:
            p = self._world_to_pixel(x, y)
            sum_x += p.x()
            sum_y += p.y()
            if rect.contains(int(p.x()), int(p.y())):
                any_visible = True
        return any_visible, QPointF(sum_x / n, sum_y / n)

    def _draw_offscreen_indicator(
        self, painter: QPainter, system_centroid: QPointF
    ) -> None:
        """Draw a banner + arrow when the system is fully off-screen.

        Always-visible recovery affordance: tells the user where to look
        and offers double-click-to-fit. The user can never get into a
        state where they cannot find the system.
        """
        rect = self.rect()
        # Banner
        banner_h = 28
        banner_rect = QRectF(rect.left(), rect.top(), float(rect.width()), banner_h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(180, 60, 60, 200)))
        painter.drawRect(banner_rect)
        painter.setPen(QColor(255, 240, 240))
        painter.setFont(QFont("Sans", 10, QFont.Weight.Bold))
        painter.drawText(
            banner_rect,
            Qt.AlignmentFlag.AlignCenter,
            "⚠  System off-screen — double-click or press F to fit view",
        )

        # Arrow from widget center pointing toward the system centroid
        from PyQt6.QtGui import QPolygonF

        widget_center = QPointF(rect.center())
        dx = system_centroid.x() - widget_center.x()
        dy = system_centroid.y() - widget_center.y()
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1.0:
            return
        nx = dx / length
        ny = dy / length

        # Clamp the arrow tip to the widget edge so it stays visible
        margin = 36.0
        max_x = max(margin, rect.width() - margin)
        max_y = max(margin + banner_h, rect.height() - margin)
        # Project from center along (nx, ny) until we hit the edge
        scale_x = (max_x / 2.0) / max(abs(nx), 1e-6)
        scale_y = (max_y / 2.0) / max(abs(ny), 1e-6)
        arrow_len = max(40.0, min(scale_x, scale_y) - 20.0)
        tip = QPointF(
            widget_center.x() + nx * arrow_len,
            widget_center.y() + ny * arrow_len,
        )

        # Draw shaft
        painter.setPen(QPen(QColor(255, 220, 60), 4, Qt.PenStyle.SolidLine))
        painter.drawLine(widget_center, tip)
        # Draw arrowhead
        head_len = 14.0
        # Perpendicular for the arrow wings
        px = -ny
        py = nx
        wing1 = QPointF(
            tip.x() - nx * head_len + px * head_len * 0.6,
            tip.y() - ny * head_len + py * head_len * 0.6,
        )
        wing2 = QPointF(
            tip.x() - nx * head_len - px * head_len * 0.6,
            tip.y() - ny * head_len - py * head_len * 0.6,
        )
        painter.setBrush(QBrush(QColor(255, 220, 60)))
        painter.drawPolygon(QPolygonF([tip, wing1, wing2]))

    def keyPressEvent(self, event: object) -> None:  # noqa: N802
        """F → fit view; double-click also fits via mouseDoubleClickEvent."""
        from PyQt6.QtGui import QKeyEvent

        if isinstance(event, QKeyEvent) and event.key() == Qt.Key.Key_F:
            self.reset_view()
            event.accept()
            return
        super().keyPressEvent(event)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Shared drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid(self, painter: QPainter) -> None:
        """Draw subtle reference grid."""
        if painter is None:
            raise ValueError("painter must be provided")
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

    def _draw_ground_line(self, painter: QPainter, ground_y: float) -> None:
        """Draw a ground reference line at the given world-Y coordinate.

        Parameters
        ----------
        ground_y : float
            World-space Y coordinate for the ground plane edge.
        """
        if painter is None:
            raise ValueError("painter must be provided")
        p1 = self._world_to_pixel(-3.5, ground_y)
        p2 = self._world_to_pixel(3.5, ground_y)
        pen = QPen(self.COLOR_GROUND, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

    def _draw_ground_plane(self, painter: QPainter, ground_y: float) -> None:
        """Draw a filled semi-transparent ground plane below the pendulum.

        Renders a filled rectangle from the ground line down, plus the
        ground line itself, giving a clearer sense of the surface.
        """
        if painter is None:
            raise ValueError("painter must be provided")
        from PyQt6.QtGui import QLinearGradient

        # Draw the filled ground region
        left = self._world_to_pixel(-4.0, ground_y)
        right = self._world_to_pixel(4.0, ground_y)
        bottom_left = self._world_to_pixel(-4.0, ground_y - 1.5)
        bottom_right = self._world_to_pixel(4.0, ground_y - 1.5)

        # Gradient from ground surface color to transparent
        gradient = QLinearGradient(left, bottom_left)
        gradient.setColorAt(0.0, QColor(40, 110, 40, 60))
        gradient.setColorAt(1.0, QColor(40, 110, 40, 0))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(gradient))
        from PyQt6.QtGui import QPolygonF

        poly = QPolygonF([left, right, bottom_right, bottom_left])
        painter.drawPolygon(poly)

        # Ground surface line
        pen = QPen(QColor(50, 140, 50, 120), 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawLine(left, right)

    def _draw_tilt_plane(self, painter: QPainter) -> None:
        """Draw a visual indicator of the swing plane tilt angle.

        When tilt != 0, draws a subtle angled plane through the pivot
        to show the user the orientation of the swing surface.
        """
        if painter is None:
            raise ValueError("painter must be provided")
        if abs(self._tilt_angle) < 1e-4:
            return

        from PyQt6.QtGui import QPolygonF

        # Draw tilted plane as a semi-transparent quadrilateral
        # The plane passes through origin and extends in x-z
        extent = 2.5
        sin_t = float(np.sin(self._tilt_angle))

        # Four corners of the plane in world coords (projected)
        # The plane is in the x-z plane, tilted around x-axis
        corners_world = [
            (-extent, 0.0),
            (extent, 0.0),
            (extent, -extent * sin_t),
            (-extent, -extent * sin_t),
        ]

        corners_px = [self._world_to_pixel(x, y) for x, y in corners_world]
        poly = QPolygonF(corners_px)

        painter.setPen(QPen(QColor(80, 180, 80, 80), 1, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(self.COLOR_TILT_PLANE))
        painter.drawPolygon(poly)

        # Angle label
        tilt_deg = np.degrees(self._tilt_angle)
        label_pos = self._world_to_pixel(extent + 0.3, -extent * sin_t * 0.5)
        painter.setPen(QColor(80, 180, 80, 160))
        painter.setFont(QFont("Monospace", 8))
        painter.drawText(label_pos, f"tilt {tilt_deg:.1f}°")

    def _draw_ball(
        self, painter: QPainter, x: float, y: float, radius_m: float = 0.0214
    ) -> None:
        """Draw a golf ball at the given world coordinates.

        Pre: radius_m > 0

        Parameters
        ----------
        x, y : float
            World coordinates of the ball center.
        radius_m : float
            Ball radius in meters (default: golf ball = 21.4mm).
        """
        if not (radius_m > 0):
            raise ValueError(f"Ball radius must be positive, got {radius_m}")
        center = self._world_to_pixel(x, y)
        r_px = max(4.0, radius_m * self._pixels_per_meter)

        # Ball shadow
        shadow = QColor(0, 0, 0, 40)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(shadow))
        painter.drawEllipse(QPointF(center.x() + 2, center.y() + 2), r_px, r_px)

        # Ball body
        from PyQt6.QtGui import QRadialGradient

        gradient = QRadialGradient(
            center.x() - r_px * 0.3,
            center.y() - r_px * 0.3,
            r_px * 1.5,
        )
        gradient.setColorAt(0.0, QColor(255, 255, 255))
        gradient.setColorAt(0.5, QColor(240, 240, 240))
        gradient.setColorAt(1.0, QColor(200, 200, 200))
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(160, 160, 160), 1))
        painter.drawEllipse(center, r_px, r_px)

    def _draw_trail(self, painter: QPainter) -> None:
        """Draw Catmull-Rom smoothed tip trail with fade-in.

        Note: this method must NOT be JIT-compiled — it operates on Qt
        objects (QPainter, QColor, QPen) which numba cannot type. A prior
        ``@jit(nopython=True)`` decorator here crashed the entire process
        on the first paint after a simulation finished.
        """
        if painter is None:
            raise ValueError("painter must be provided")
        n = len(self._trail)
        if n < 2:
            return

        if n >= 4:
            smooth = self._catmull_rom_smooth(list(self._trail), self.SPLINE_SUBDIV)
        else:
            smooth = list(self._trail)

        ns = len(smooth)
        for i in range(1, ns):
            t = i / ns
            alpha = int(30 + 180 * t)
            width = 1.0 + 2.5 * t
            color = QColor(self.COLOR_TRAIL)
            color.setAlpha(alpha)
            pen = QPen(color, width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            x0, y0 = smooth[i - 1]
            x1, y1 = smooth[i]
            painter.drawLine(
                self._world_to_pixel(x0, y0),
                self._world_to_pixel(x1, y1),
            )

    @staticmethod
    def _catmull_rom_smooth(
        points: list[tuple[float, float]],
        n_sub: int = 4,
    ) -> list[tuple[float, float]]:
        """Catmull-Rom spline interpolation over trail points (#1116).

        Delegates to the Qt-free catmull_rom module for testability.
        Pre: len(points) >= 4, n_sub >= 1
        Post: len(result) >= len(points)
        """
        if points is None:
            raise ValueError("points must be provided")
        from .catmull_rom import catmull_rom_smooth

        return catmull_rom_smooth(points, n_sub)

    def _draw_joint(
        self, painter: QPainter, pos: QPointF, radius: float, color: QColor
    ) -> None:
        """Draw a joint marker with glow effect.

        Pre: radius > 0
        """
        if not (radius > 0):
            raise ValueError(f"Joint marker radius must be positive, got {radius}")
        glow = QColor(color)
        glow.setAlpha(60)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(pos, radius * 2, radius * 2)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(pos, radius, radius)

    def _draw_no_gravity_badge(self, painter: QPainter) -> None:
        """Draw a 'No Gravity' indicator badge."""
        if painter is None:
            raise ValueError("painter must be provided")
        painter.setPen(QPen(self.COLOR_NO_GRAVITY, 2))
        painter.setFont(QFont("Sans", 10, QFont.Weight.Bold))
        painter.drawText(self.width() - 120, 20, "⚠ No Gravity")

    # ------------------------------------------------------------------
    # 3D segment rendering (#1155)
    # ------------------------------------------------------------------

    def set_3d_mode(self, enabled: bool) -> None:
        """Toggle 3D tapered segment rendering (#1155).

        Pre: enabled is bool.
        """
        if enabled is None:
            raise ValueError("enabled must be provided")
        self._3d_mode = bool(enabled)
        self.update()

    def _draw_3d_segment(
        self,
        painter: QPainter,
        p1: QPointF,
        p2: QPointF,
        width_start: float,
        width_end: float,
        color: QColor,
    ) -> None:
        """Draw a tapered, gradient-shaded segment between two pixel points.

        Creates a pseudo-3D effect by drawing a filled quadrilateral with
        a lateral gradient (highlight on one side, shadow on the other).

        Pre: width_start > 0, width_end > 0.
        Post: A tapered polygon is rendered between p1 and p2.
        """
        if painter is None:
            raise ValueError("painter must be provided")
        from PyQt6.QtGui import QLinearGradient, QPolygonF

        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        length = (dx**2 + dy**2) ** 0.5
        if length < 1e-3:
            return

        # Normal vector (perpendicular to segment direction)
        nx = -dy / length
        ny = dx / length

        # Build tapered quad: 4 corners
        hw1 = width_start / 2.0
        hw2 = width_end / 2.0
        poly = QPolygonF(
            [
                QPointF(p1.x() + nx * hw1, p1.y() + ny * hw1),
                QPointF(p2.x() + nx * hw2, p2.y() + ny * hw2),
                QPointF(p2.x() - nx * hw2, p2.y() - ny * hw2),
                QPointF(p1.x() - nx * hw1, p1.y() - ny * hw1),
            ]
        )

        # Gradient across the width for 3D cylinder effect
        mid_x = (p1.x() + p2.x()) / 2
        mid_y = (p1.y() + p2.y()) / 2
        grad = QLinearGradient(
            QPointF(mid_x + nx * hw1, mid_y + ny * hw1),
            QPointF(mid_x - nx * hw1, mid_y - ny * hw1),
        )
        highlight = QColor(color)
        highlight.setAlpha(min(255, color.alpha() + 60))
        shadow = QColor(color)
        shadow.setAlpha(max(30, color.alpha() - 80))

        grad.setColorAt(0.0, highlight)
        grad.setColorAt(0.5, color)
        grad.setColorAt(1.0, shadow)

        painter.setPen(QPen(QColor(0, 0, 0, 40), 1))
        painter.setBrush(QBrush(grad))
        painter.drawPolygon(poly)

    def _draw_3d_joint_cap(
        self,
        painter: QPainter,
        pos: QPointF,
        radius: float,
        color: QColor,
    ) -> None:
        """Draw a spherical joint cap with radial gradient for 3D effect.

        Creates a pseudo-3D sphere at the joint using radial gradient shading
        with a specular highlight offset from center.

        Pre: radius > 0
        """
        if not (radius > 0):
            raise ValueError(f"Joint cap radius must be positive, got {radius}")
        from PyQt6.QtGui import QRadialGradient

        # Shadow beneath the joint
        shadow_color = QColor(0, 0, 0, 50)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(shadow_color))
        painter.drawEllipse(
            QPointF(pos.x() + 1.5, pos.y() + 1.5),
            radius * 1.2,
            radius * 1.2,
        )

        # Sphere with specular highlight
        gradient = QRadialGradient(
            pos.x() - radius * 0.3,
            pos.y() - radius * 0.3,
            radius * 1.5,
        )
        highlight = QColor(color)
        highlight.setAlpha(255)
        gradient.setColorAt(0.0, highlight.lighter(160))
        gradient.setColorAt(0.4, color)
        gradient.setColorAt(0.8, color.darker(140))
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))

        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(color.darker(130), 0.5))
        painter.drawEllipse(pos, radius, radius)

    def _draw_shadow_projection(
        self,
        painter: QPainter,
        world_points: list[tuple[float, float]],
        ground_y: float,
    ) -> None:
        """Project a shadow of the pendulum segments onto the ground plane.

        Draws a semi-transparent flattened silhouette at the ground level,
        giving depth perception even in 2D projection.

        Pre: len(world_points) >= 2
        """
        if len(world_points) < 2:
            return

        shadow_color = QColor(0, 0, 0, 25)
        painter.setPen(QPen(shadow_color, 2))

        for i in range(len(world_points) - 1):
            x1, _ = world_points[i]
            x2, _ = world_points[i + 1]
            p1 = self._world_to_pixel(x1, ground_y)
            p2 = self._world_to_pixel(x2, ground_y)
            painter.drawLine(p1, p2)

    # ------------------------------------------------------------------
    # Image export (#1779)
    # ------------------------------------------------------------------

    def export_image(
        self, file_path: str, width: int = 1920, height: int = 1080
    ) -> None:
        """Export the current visualization as a high-resolution image.

        Supports PNG, SVG, and PDF formats based on file extension.

        Pre: file_path ends with .png, .svg, or .pdf
        Pre: width > 0, height > 0
        Post: File is written to disk.
        """
        if not (width > 0 and height > 0):
            raise ValueError(f"Dimensions must be positive: {width}x{height}")

        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""

        if ext == "svg":
            from PyQt6.QtSvg import QSvgGenerator

            generator = QSvgGenerator()
            generator.setFileName(file_path)
            generator.setSize(self.size())
            generator.setViewBox(self.rect())
            generator.setTitle("Pendulum Simulator Export")
            painter = QPainter(generator)
            self.render(painter)
            painter.end()

        elif ext == "pdf":
            from PyQt6.QtGui import QPageLayout, QPageSize
            from PyQt6.QtCore import QMarginsF
            from PyQt6.QtPrintSupport import QPrinter

            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(file_path)
            printer.setPageLayout(
                QPageLayout(
                    QPageSize(QPageSize.PageSizeId.A4),
                    QPageLayout.Orientation.Landscape,
                    QMarginsF(10, 10, 10, 10),
                )
            )
            painter = QPainter(printer)
            self.render(painter)
            painter.end()

        else:
            # Default: PNG raster export
            from PyQt6.QtGui import QImage

            image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(self.COLOR_BG)
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # Scale to fit the export dimensions
            scale_x = width / max(1, self.width())
            scale_y = height / max(1, self.height())
            scale = min(scale_x, scale_y)
            painter.scale(scale, scale)
            self.render(painter)
            painter.end()
            image.save(file_path)
