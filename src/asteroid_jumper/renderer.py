"""Simulation renderer widget — PyQt6 QPainter-based canvas.

Draws with Catppuccin Mocha colour palette to match Tools repo theme.
Animation runs via QTimer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

if TYPE_CHECKING:
    from asteroid_jumper.controller import SimController

# Catppuccin Mocha colour palette
C_BASE = QColor("#1e1e2e")
C_MANTLE = QColor("#181825")
C_CRUST = QColor("#11111b")
C_TEXT = QColor("#cdd6f4")
C_SUBTEXT = QColor("#a6adc8")
C_SURFACE0 = QColor("#313244")
C_SURFACE1 = QColor("#45475a")
C_BLUE = QColor("#89b4fa")
C_GREEN = QColor("#a6e3a1")
C_YELLOW = QColor("#f9e2af")
C_RED = QColor("#f38ba8")
C_MAUVE = QColor("#cba6f7")
C_TEAL = QColor("#94e2d5")
C_PEACH = QColor("#fab387")
C_LAVENDER = QColor("#b4befe")
C_SKY = QColor("#89dceb")
C_FLAMINGO = QColor("#f2cdcd")

STAR_POSITIONS: list[tuple[float, float]] = [
    (i * 0.618033988 % 1.0, (i * 0.381966 % 1.0)) for i in range(200)
]

FPS = 60
SIM_SPEED = 1.0  # simulation seconds per real second
VIEWPORT_SCALE = 25.0  # pixels per simulation metre (initial)
TRAIL_LENGTH = 120  # max trail points stored


class AsteroidJumperRenderer(QWidget):
    """Interactive PyQt6 canvas rendering the asteroid-jumper simulation."""

    def __init__(
        self, controller: SimController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        assert controller is not None, "controller must not be None"
        self._ctrl = controller
        self._scale = VIEWPORT_SCALE
        self._pan = QPointF(0.0, 0.0)  # world-space offset (m)
        self._running = False
        self._asteroid_trail: list[tuple[float, float]] = []
        self._jumper_trail: list[tuple[float, float]] = []
        self._force_angle_drag = False
        self._force_angle_screen: QPointF | None = None
        self.force_angle_changed = _SimpleSignal()

        self.setMinimumSize(600, 500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._timer = QTimer(self)
        self._timer.setInterval(1000 // FPS)
        self._timer.timeout.connect(self._on_tick)

    # ------------------------------------------------------------------
    # Public API (called by main window)
    # ------------------------------------------------------------------

    def start_animation(self) -> None:
        """Start or resume the animation timer."""
        self._running = True
        self._timer.start()

    def stop_animation(self) -> None:
        """Pause the animation timer."""
        self._running = False
        self._timer.stop()

    def reset_view(self) -> None:
        """Centre the view and clear trails."""
        self._pan = QPointF(0.0, 0.0)
        self._scale = VIEWPORT_SCALE
        self._asteroid_trail.clear()
        self._jumper_trail.clear()
        self.update()

    def set_scale(self, scale: float) -> None:
        """Set zoom level (pixels per metre)."""
        assert scale > 0
        self._scale = scale
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _world_to_screen(self, wx: float, wy: float) -> QPointF:
        """World (m) → screen (px), y-flipped for Qt."""
        assert wx is not None, "wx must be provided"
        cx = self.width() / 2 + self._pan.x()
        cy = self.height() / 2 + self._pan.y()
        return QPointF(cx + wx * self._scale, cy - wy * self._scale)

    def _screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """Screen (px) → world (m)."""
        assert sx is not None, "sx must be provided"
        cx = self.width() / 2 + self._pan.x()
        cy = self.height() / 2 + self._pan.y()
        return (sx - cx) / self._scale, -(sy - cy) / self._scale

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        assert _event is not None, "_event must be provided"
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_background(painter)
        self._draw_stars(painter)
        if self._ctrl.state.phase != "ready":
            self._draw_trails(painter)
        self._draw_force_indicator(painter)
        self._draw_asteroid(painter)
        self._draw_jumper(painter)
        self._draw_hud(painter)
        painter.end()

    def mousePressEvent(self, event: object) -> None:  # noqa: N802
        assert isinstance(event, type(event))
        from PyQt6.QtGui import QMouseEvent

        if isinstance(event, QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                self._force_angle_drag = True
                self._update_force_from_mouse(event.position())
            elif event.button() == Qt.MouseButton.RightButton:
                self._pan_start = event.position()

    def mouseMoveEvent(self, event: object) -> None:  # noqa: N802
        from PyQt6.QtGui import QMouseEvent

        if isinstance(event, QMouseEvent):
            if self._force_angle_drag and event.buttons() == Qt.MouseButton.LeftButton:
                self._update_force_from_mouse(event.position())
                self.update()

    def mouseReleaseEvent(self, event: object) -> None:  # noqa: N802
        self._force_angle_drag = False

    def wheelEvent(self, event: object) -> None:  # noqa: N802
        from PyQt6.QtGui import QWheelEvent

        if isinstance(event, QWheelEvent):
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else 0.9
            self._scale = max(5.0, min(200.0, self._scale * factor))
            self.update()

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------

    def _update_force_from_mouse(self, pos: QPointF) -> None:
        """Set force angle based on mouse position relative to asteroid."""
        assert pos is not None, "pos must be provided"
        ast = self._ctrl.state.asteroid
        asteroid_screen = self._world_to_screen(ast.pos.x, ast.pos.y)
        dx = pos.x() - asteroid_screen.x()
        dy = -(pos.y() - asteroid_screen.y())  # flip y
        angle_deg = math.degrees(math.atan2(dy, dx))
        self._ctrl.set_force_angle(angle_deg)
        self._ctrl.set_jump_direction(angle_deg)
        self._ctrl.state = self._ctrl._build_state()
        self.force_angle_changed.emit(angle_deg)
        self.update()

    # ------------------------------------------------------------------
    # Animation tick
    # ------------------------------------------------------------------

    def _on_tick(self) -> None:
        if not self._running:
            return
        dt = SIM_SPEED / FPS
        if self._ctrl.state.phase in ("jumping", "flight"):
            ast = self._ctrl.state.asteroid
            jmp = self._ctrl.state.jumper
            self._asteroid_trail.append((ast.pos.x, ast.pos.y))
            self._jumper_trail.append((jmp.pos.x, jmp.pos.y))
            if len(self._asteroid_trail) > TRAIL_LENGTH:
                self._asteroid_trail.pop(0)
            if len(self._jumper_trail) > TRAIL_LENGTH:
                self._jumper_trail.pop(0)
            self._ctrl.tick(dt)
        self.update()

    # ------------------------------------------------------------------
    # Drawing methods
    # ------------------------------------------------------------------

    def _draw_background(self, p: QPainter) -> None:
        """Fill with deep-space gradient."""
        assert p is not None, "p must be provided"
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, C_CRUST)
        grad.setColorAt(1.0, C_MANTLE)
        p.fillRect(self.rect(), grad)

    def _draw_stars(self, p: QPainter) -> None:
        """Scatter small white dots as background stars."""
        assert p is not None, "p must be provided"
        p.save()
        w, h = self.width(), self.height()
        for fx, fy in STAR_POSITIONS:
            sx, sy = fx * w, fy * h
            brightness = int(80 + 175 * ((fx * 7 + fy * 13) % 1.0))
            star_color = QColor(brightness, brightness, brightness, 200)
            p.setPen(QPen(star_color, 1.2))
            p.drawPoint(QPointF(sx, sy))
        p.restore()

    def _draw_trails(self, p: QPainter) -> None:
        """Draw position trails for asteroid and jumper."""
        assert p is not None, "p must be provided"
        self._draw_single_trail(p, self._asteroid_trail, C_TEAL)
        self._draw_single_trail(p, self._jumper_trail, C_PEACH)

    def _draw_single_trail(
        self, p: QPainter, trail: list[tuple[float, float]], color: QColor
    ) -> None:
        """Draw a single fading trail."""
        assert p is not None, "p must be provided"
        if len(trail) < 2:
            return
        p.save()
        for i in range(len(trail) - 1):
            alpha = int(20 + 200 * i / len(trail))
            pen_color = QColor(color)
            pen_color.setAlpha(alpha)
            p.setPen(QPen(pen_color, 1.5))
            a = self._world_to_screen(*trail[i])
            b = self._world_to_screen(*trail[i + 1])
            p.drawLine(a, b)
        p.restore()

    def _draw_asteroid(self, p: QPainter) -> None:
        """Draw the asteroid as a textured polygon."""
        assert p is not None, "p must be provided"
        ast = self._ctrl.state.asteroid
        shape = self._ctrl.shape
        p.save()
        # Build world-frame vertices
        path = QPainterPath()
        first = True
        for bx, by in shape.vertices:
            # Rotate by asteroid angle then translate
            cos_a = math.cos(ast.angle)
            sin_a = math.sin(ast.angle)
            wx = bx * cos_a - by * sin_a + ast.pos.x
            wy = bx * sin_a + by * cos_a + ast.pos.y
            sp = self._world_to_screen(wx, wy)
            if first:
                path.moveTo(sp)
                first = False
            else:
                path.lineTo(sp)
        path.closeSubpath()

        # Radial gradient fill — rocky brownish
        rock_center = self._world_to_screen(ast.pos.x, ast.pos.y)
        rg = QRadialGradient(rock_center, self._scale * shape.semi_a * 1.1)
        rg.setColorAt(0.0, QColor("#6c5c4a"))
        rg.setColorAt(0.4, QColor("#4a3e32"))
        rg.setColorAt(1.0, QColor("#2a2320"))
        p.fillPath(path, rg)
        pen = QPen(QColor("#8b7355"), 2)
        p.setPen(pen)
        p.drawPath(path)

        # COM marker
        com_sp = self._world_to_screen(ast.pos.x, ast.pos.y)
        p.setPen(QPen(C_YELLOW, 1.5))
        p.setBrush(QBrush(C_YELLOW))
        r = 4
        p.drawEllipse(com_sp, r, r)

        # Craters
        self._draw_craters(p, ast, shape)
        p.restore()

    def _draw_craters(self, p: QPainter, ast: object, shape: object) -> None:
        """Draw decorative craters on the asteroid surface."""
        from asteroid_jumper.physics import RigidBody

        assert isinstance(ast, RigidBody)
        crater_angles = [0.5, 1.8, 3.1, 4.7, 5.5]
        crater_sizes = [0.6, 0.4, 0.5, 0.3, 0.7]
        p.save()
        for ca, cs in zip(crater_angles, crater_sizes, strict=False):
            cx_b = math.cos(ca) * shape.semi_a * 0.55  # type: ignore[attr-defined]
            cy_b = math.sin(ca) * shape.semi_b * 0.55  # type: ignore[attr-defined]
            cos_a = math.cos(ast.angle)
            sin_a = math.sin(ast.angle)
            wx = cx_b * cos_a - cy_b * sin_a + ast.pos.x
            wy = cx_b * sin_a + cy_b * cos_a + ast.pos.y
            sp = self._world_to_screen(wx, wy)
            cr = cs * self._scale * 0.8
            p.setPen(QPen(QColor("#2a1e15"), 1))
            p.setBrush(QBrush(QColor(42, 30, 21, 180)))
            p.drawEllipse(sp, cr, cr * 0.6)
        p.restore()

    def _draw_jumper(self, p: QPainter) -> None:
        """Draw the astronaut-style jumper with animated legs."""
        assert p is not None, "p must be provided"
        jmp = self._ctrl.state.jumper
        phase = self._ctrl.leg_phase()
        p.save()
        sp = self._world_to_screen(jmp.pos.x, jmp.pos.y)
        angle = jmp.angle

        p.translate(sp)
        p.rotate(-math.degrees(angle))

        px_p_m = self._scale  # pixels per metre

        self._draw_jumper_body(p, px_p_m, phase)
        p.restore()

    def _draw_jumper_body(self, p: QPainter, scale: float, phase: float) -> None:
        """Draw human figure: head, torso, arms, animated legs."""
        assert p is not None, "p must be provided"
        h = scale * JUMPER_HEIGHT_REF  # reference heights in pixels
        head_r = h * 0.12
        torso_h = h * 0.30
        torso_w = h * 0.14

        # Visor / head
        p.setBrush(QBrush(C_LAVENDER))
        p.setPen(QPen(C_SURFACE1, 1))
        p.drawEllipse(QPointF(0, -h * 0.45), head_r, head_r)
        # Visor tint
        p.setBrush(QBrush(QColor(137, 180, 250, 120)))
        p.drawEllipse(QPointF(0, -h * 0.45), head_r * 0.7, head_r * 0.7)

        # Torso (spacesuit)
        torso_rect = QRectF(-torso_w / 2, -h * 0.35, torso_w, torso_h)
        p.setBrush(QBrush(C_SURFACE1))
        p.setPen(QPen(C_BLUE, 1))
        p.drawRoundedRect(torso_rect, 3, 3)
        # Patch
        p.setBrush(QBrush(C_BLUE))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(QRectF(-torso_w * 0.25, -h * 0.28, torso_w * 0.5, h * 0.07))

        # Arms
        arm_angle = math.radians(30 + 60 * phase)  # raise arms during jump
        self._draw_arm(p, scale, arm_angle, left=True)
        self._draw_arm(p, scale, arm_angle, left=False)

        # Legs — animated by phase
        self._draw_legs(p, scale, phase)

    def _draw_arm(
        self, p: QPainter, scale: float, arm_angle: float, *, left: bool
    ) -> None:
        """Draw one arm."""
        assert p is not None, "p must be provided"
        h = scale * JUMPER_HEIGHT_REF
        torso_w = h * 0.14
        arm_len = h * 0.22
        xsign = -1.0 if left else 1.0
        shoulder_x = xsign * torso_w / 2
        shoulder_y = -h * 0.30
        elbow_x = shoulder_x + xsign * arm_len * 0.5 * math.cos(arm_angle)
        elbow_y = shoulder_y + arm_len * 0.5 * math.sin(arm_angle)
        hand_x = elbow_x + xsign * arm_len * 0.5 * math.cos(arm_angle * 0.7)
        hand_y = elbow_y + arm_len * 0.5 * math.sin(arm_angle * 0.7)
        pen = QPen(C_SURFACE1, max(2, scale * 0.06))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(shoulder_x, shoulder_y), QPointF(elbow_x, elbow_y))
        p.drawLine(QPointF(elbow_x, elbow_y), QPointF(hand_x, hand_y))

    def _draw_legs(self, p: QPainter, scale: float, phase: float) -> None:
        """Draw two animated legs: crouch on ground, extend at jump, tuck in flight."""
        assert p is not None, "p must be provided"
        h = scale * JUMPER_HEIGHT_REF
        hip_y = -h * 0.05  # hip position (bottom of torso)
        thigh = h * 0.20
        shin = h * 0.18
        foot_r = max(2, scale * 0.05)

        # Leg phase animation
        # phase 0 → crouched, 0.5 → extending, 1.0 → fully straight/tucked
        crouch = math.pi * 0.55 * (1.0 - phase)  # knee bend angle
        spread = math.radians(12)  # lateral spread of legs

        for xsign in (-1.0, 1.0):
            hip_x = xsign * h * 0.07
            # Thigh direction: slightly outward, angled down
            thigh_angle = math.pi / 2 + xsign * spread + crouch * 0.5
            kx = hip_x + thigh * math.cos(thigh_angle)
            ky = hip_y + thigh * math.sin(thigh_angle)
            # Shin continues down/back
            shin_angle = math.pi / 2 - xsign * spread * 0.5 - crouch
            fx = kx + shin * math.cos(shin_angle)
            fy = ky + shin * math.sin(shin_angle)

            pen = QPen(C_SURFACE1, max(2, scale * 0.06))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(hip_x, hip_y), QPointF(kx, ky))
            p.drawLine(QPointF(kx, ky), QPointF(fx, fy))
            # Foot
            p.setBrush(QBrush(C_SURFACE0))
            p.setPen(QPen(C_BLUE, 1))
            p.drawEllipse(QPointF(fx, fy), foot_r * 1.6, foot_r)

    def _draw_force_indicator(self, p: QPainter) -> None:
        """Draw the adjustable force vector arrow on the asteroid."""
        assert p is not None, "p must be provided"
        if self._ctrl.state.phase != "ready":
            return
        ast = self._ctrl.state.asteroid
        shape = self._ctrl.shape
        angle_rad = math.radians(self._ctrl.force_angle_deg)

        # Contact point on surface
        from asteroid_jumper.asteroid_shape import surface_point_at_angle

        sx, sy = surface_point_at_angle(shape, angle_rad)
        contact_screen = self._world_to_screen(sx + ast.pos.x, sy + ast.pos.y)

        # Arrow from contact outward
        arrow_len = self._scale * 4
        dir_x = math.cos(angle_rad)
        dir_y = -math.sin(angle_rad)  # flip y for screen
        tip = QPointF(
            contact_screen.x() + dir_x * arrow_len,
            contact_screen.y() + dir_y * arrow_len,
        )

        p.save()
        pen = QPen(C_YELLOW, 2.5, Qt.PenStyle.SolidLine)
        p.setPen(pen)
        p.drawLine(contact_screen, tip)
        # Arrowhead
        self._draw_arrowhead(p, contact_screen, tip, C_YELLOW, size=8)

        # Label
        p.setPen(QPen(C_YELLOW))
        p.setFont(QFont("monospace", 9))
        p.drawText(QPointF(tip.x() + 5, tip.y() - 5), "Jump")
        p.restore()

    def _draw_arrowhead(
        self,
        p: QPainter,
        start: QPointF,
        tip: QPointF,
        color: QColor,
        size: float = 8,
    ) -> None:
        """Draw a filled arrowhead at *tip* pointing away from *start*."""
        assert p is not None, "p must be provided"
        dx = tip.x() - start.x()
        dy = tip.y() - start.y()
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # perpendicular
        path = QPainterPath()
        path.moveTo(tip)
        path.lineTo(
            QPointF(
                tip.x() - ux * size + px * size / 2, tip.y() - uy * size + py * size / 2
            )
        )
        path.lineTo(
            QPointF(
                tip.x() - ux * size - px * size / 2, tip.y() - uy * size - py * size / 2
            )
        )
        path.closeSubpath()
        p.fillPath(path, QBrush(color))

    def _draw_hud(self, p: QPainter) -> None:
        """Draw HUD overlay with key metrics."""
        assert p is not None, "p must be provided"
        p.save()
        p.setFont(QFont("monospace", 9))
        phase = self._ctrl.state.phase
        sim_time = self._ctrl.state.time

        lines = [
            f"Phase: {phase.upper()}",
            f"Time:  {sim_time:.2f} s",
            f"Jumper speed:   {self._ctrl.jumper_speed():.3f} m/s",
            f"Jumper ω:       {self._ctrl.jumper_angular_speed():.3f} rad/s",
            f"Asteroid speed: {self._ctrl.asteroid_speed():.3f} m/s",
            f"Asteroid ω:     {self._ctrl.asteroid_angular_speed():.3f} rad/s",
            f"Off-centre:     {self._ctrl.off_centre_fraction():.2%}",
        ]
        if phase == "ready":
            lines.append("← Drag on asteroid to set jump angle")

        bg = QColor(C_MANTLE)
        bg.setAlpha(200)
        row_h = 16
        margin = 8
        box_w = 240
        box_h = row_h * len(lines) + margin * 2
        p.fillRect(QRectF(8, 8, box_w, box_h), bg)
        p.setPen(QPen(C_SURFACE1, 1))
        p.drawRect(QRectF(8, 8, box_w, box_h))

        for i, line in enumerate(lines):
            color = C_BLUE if i == 0 else (C_GREEN if i < 5 else C_SUBTEXT)
            p.setPen(QPen(color))
            p.drawText(QPointF(margin + 8, margin + 8 + (i + 1) * row_h - 3), line)
        p.restore()


# Global reference height used throughout (normalised jumper height in "nice" units)
JUMPER_HEIGHT_REF: float = 0.08  # fraction of scale


# ---------------------------------------------------------------------------
# Mini signal helper
# ---------------------------------------------------------------------------


class _SimpleSignal:
    """Lightweight callable signal (wraps a list of callbacks)."""

    def __init__(self) -> None:
        self._slots: list[object] = []

    def connect(self, slot: object) -> None:
        assert callable(slot)
        self._slots.append(slot)

    def emit(self, *args: object) -> None:
        for slot in self._slots:
            slot(*args)  # type: ignore[operator]
