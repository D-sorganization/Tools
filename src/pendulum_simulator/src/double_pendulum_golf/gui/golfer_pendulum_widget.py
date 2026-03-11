"""
Custom QWidget that draws the golfer upper-body model animation.

Renders the branching topology: hub -> two arm chains -> shared club,
with joint markers, club tip trail, force vectors, and zoom/pan canvas.

Inherits shared zoom/pan, grid, trail, and coordinate mapping from
BasePendulumWidget.  Specializes: golfer segment drawing, branching
topology, golfer-specific COM, and force vectors.

Design by Contract
------------------
- Pre:  ``set_simulation(result)`` requires ``result is not None``.
- Pre:  ``set_frame(idx)`` requires loaded simulation.
- Inv:  ``_force_scale > 0`` (enforced by base class).
"""

from __future__ import annotations

import logging
import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from ..simulation_golfer import GolferSimulationResult
from .base_pendulum_widget import BasePendulumWidget

logger = logging.getLogger(__name__)


class GolferPendulumWidget(BasePendulumWidget):
    """Animated visualization of the golfer upper-body model.

    Draws the branching arm topology and shared club segment.
    Supports zoom (scroll), pan (drag), and double-click reset.
    """

    # Model-specific colors
    COLOR_HUB = QColor(180, 180, 200)
    COLOR_RIGHT_ARM = QColor(70, 140, 240)
    COLOR_LEFT_ARM = QColor(120, 200, 140)
    COLOR_CLUB_SHAFT = QColor(240, 180, 50)
    COLOR_CLUBHEAD = QColor(255, 80, 80)
    COLOR_JOINT = QColor(210, 210, 220)
    COLOR_GRIP = QColor(255, 225, 80)
    COLOR_SHOULDER_BAR = QColor(140, 140, 160)

    # Zero torque vector color (violet/purple, same as PendulumWidget)
    COLOR_ZERO_TORQUE = QColor(210, 120, 255)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._result: GolferSimulationResult | None = None
        self._current_idx: int = 0

        # Pre-computed zero-torque counterfactual forces (#1148)
        self._zero_torque_forces: list[dict] | None = None

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def _get_total_length(self) -> float:
        if self._result is not None:
            p = self._result.params
            return max(
                p.L_hub + p.L_r_upper + p.L_r_fore + p.L_club,
                p.L_hub + p.L_l_upper + p.L_l_fore + p.L_club,
                2.0,
            )
        return 2.5

    def _has_result(self) -> bool:
        return self._result is not None

    def _draw_model(self, painter: QPainter) -> None:
        self._draw_golfer(painter)

    def _shoulder_y_fraction(self) -> float:
        """Golfer hub is higher on the canvas."""
        return 0.30

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_simulation(self, result: GolferSimulationResult) -> None:
        """Load a new golfer simulation result.

        Pre: result is not None
        """
        assert result is not None, "GolferSimulationResult must not be None"
        self._result = result
        self._current_idx = 0
        self._trail.clear()

        # Pre-compute zero-torque counterfactual forces (#1148)
        self._zero_torque_forces = self._precompute_zero_torque_forces(result)

        self.update()

    @staticmethod
    def _precompute_zero_torque_forces(
        result: GolferSimulationResult,
    ) -> list[dict]:
        """Pre-compute zero-torque counterfactual joint forces for every frame."""
        from .counterfactual_golfer import zero_torque_joint_forces

        forces: list[dict] = []
        params = result.params
        for state in result.states:
            try:
                forces.append(zero_torque_joint_forces(state, params))
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                logger.warning("zero_torque_joint_forces failed for state: %s", exc)
                forces.append({})
        return forces

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

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: object) -> None:
        self._pixels_per_meter = self._compute_base_scale() * self._zoom

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.COLOR_BG)

        self._draw_grid(painter)

        # Ground plane + tilt plane visualization
        if self._result is not None:
            L_total = self._get_total_length()
            self._draw_ground_plane(painter, -L_total)
            self._draw_tilt_plane(painter)

        if self._result is None:
            self._draw_placeholder(painter)
            painter.end()
            return

        self._draw_trail(painter)
        self._draw_golfer(painter)

        if self._show_forces:
            self._draw_force_vectors(painter)

        if self._show_zero_torque_forces:
            self._draw_zero_torque_force_vectors(painter)

        if self._show_com:
            self._draw_com(painter)

        self._draw_info(painter)

        if not self._gravity_on:
            self._draw_no_gravity_badge(painter)

        painter.end()

    # ------------------------------------------------------------------
    # Golfer topology drawing
    # ------------------------------------------------------------------

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

        # Hub standoff (origin -> hub)
        pen = QPen(self.COLOR_HUB, 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(origin, hub)

        # Scapula links (#1104, #1111)
        if "rscap" in pos:
            rscap = self._world_to_pixel(*pos["rscap"])
            pen = QPen(QColor(180, 120, 120), 2, Qt.PenStyle.DashDotLine)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(rscap, rs)
            painter.setBrush(QBrush(QColor(180, 120, 120, 150)))
            painter.drawEllipse(rscap, 3, 3)
        if "lscap" in pos:
            lscap = self._world_to_pixel(*pos["lscap"])
            pen = QPen(QColor(120, 120, 180), 2, Qt.PenStyle.DashDotLine)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(lscap, ls)
            painter.setBrush(QBrush(QColor(120, 120, 180, 150)))
            painter.drawEllipse(lscap, 3, 3)

        # Shoulder bar (RS -> LS through hub)
        pen = QPen(self.COLOR_SHOULDER_BAR, 3, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(rs, ls)

        # Right arm: RS -> RE -> RH
        pen = QPen(self.COLOR_RIGHT_ARM, 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(rs, re)
        painter.drawLine(re, rh)

        # Left arm: LS -> LE -> LH
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

        # Grip connection lines (hands -> club grip points)
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

    # ------------------------------------------------------------------
    # Force vectors
    # ------------------------------------------------------------------

    def _draw_force_vectors(self, painter: QPainter) -> None:
        if self._result is None:
            return
        try:
            forces = self._result.joint_forces_at(self._current_idx)
        except (AttributeError, ValueError, IndexError) as exc:
            logger.debug("joint_forces_at unavailable: %s", exc)
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
            if self._visible_segments is not None and key not in self._visible_segments:
                continue
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

            # Filled arrowhead
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            arrow_l = max(1.0, np.hypot(dx, dy))
            ux, uy = dx / arrow_l, dy / arrow_l
            a_len = 10.0
            a_w = 4.0
            left = QPointF(
                p2.x() - a_len * ux + a_w * uy,
                p2.y() - a_len * uy - a_w * ux,
            )
            right = QPointF(
                p2.x() - a_len * ux - a_w * uy,
                p2.y() - a_len * uy + a_w * ux,
            )
            path = QPainterPath()
            path.moveTo(p2)
            path.lineTo(left)
            path.lineTo(right)
            path.closeSubpath()
            old_brush = painter.brush()
            painter.setBrush(QBrush(painter.pen().color()))
            painter.drawPath(path)
            painter.setBrush(old_brush)

    def _draw_zero_torque_force_vectors(self, painter: QPainter) -> None:
        """Draw zero-torque (passive drift) force vectors at each joint (#1148)."""
        if self._result is None or self._zero_torque_forces is None:
            return
        forces = self._zero_torque_forces[self._current_idx]
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

        pen = QPen(self.COLOR_ZERO_TORQUE, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for key, force in forces.items():
            if self._visible_segments is not None and key not in self._visible_segments:
                continue
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

            # Arrowhead
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            arrow_l = max(1.0, np.hypot(dx, dy))
            ux, uy = dx / arrow_l, dy / arrow_l
            a_len, a_w = 8.0, 3.0
            left = QPointF(
                p2.x() - a_len * ux + a_w * uy,
                p2.y() - a_len * uy - a_w * ux,
            )
            right = QPointF(
                p2.x() - a_len * ux - a_w * uy,
                p2.y() - a_len * uy + a_w * ux,
            )
            path = QPainterPath()
            path.moveTo(p2)
            path.lineTo(left)
            path.lineTo(right)
            path.closeSubpath()
            old_brush = painter.brush()
            painter.setBrush(QBrush(self.COLOR_ZERO_TORQUE))
            painter.drawPath(path)
            painter.setBrush(old_brush)

    # ------------------------------------------------------------------
    # Center of Mass drawing
    # ------------------------------------------------------------------

    def _draw_com(self, painter: QPainter) -> None:
        """Draw the combined center of mass of the golfer system."""
        if self._result is None:
            return

        pos = self._result.positions_at(self._current_idx)
        params = self._result.params

        hub = np.array(pos["hub"])
        re = np.array(pos["re"])
        rh = np.array(pos["rh"])
        le = np.array(pos["le"])
        lh = np.array(pos["lh"])
        club_base = np.array(pos["club_base"])
        club_tip = np.array(pos["club_tip"])
        club_com = 0.5 * (club_base + club_tip)

        masses = [
            params.m_hub,
            params.m_r_upper,
            params.m_r_fore,
            params.m_l_upper,
            params.m_l_fore,
            params.m_club,
            params.m_clubhead,
        ]
        positions = [hub, re, rh, le, lh, club_com, club_tip]

        total_m = sum(masses)
        com = sum(m * p for m, p in zip(masses, positions)) / total_m

        com_px = self._world_to_pixel(float(com[0]), float(com[1]))

        r = 6
        painter.setPen(QPen(self.COLOR_COM, 2))
        painter.setBrush(QBrush(QColor(255, 255, 80, 100)))
        painter.drawEllipse(com_px, r, r)
        painter.drawLine(
            QPointF(com_px.x() - r * 1.5, com_px.y()),
            QPointF(com_px.x() + r * 1.5, com_px.y()),
        )
        painter.drawLine(
            QPointF(com_px.x(), com_px.y() - r * 1.5),
            QPointF(com_px.x(), com_px.y() + r * 1.5),
        )
        painter.setFont(QFont("Monospace", 7))
        painter.drawText(QPointF(com_px.x() + r + 2, com_px.y() - 2), "COM")

    # ------------------------------------------------------------------
    # Info and placeholder
    # ------------------------------------------------------------------

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
