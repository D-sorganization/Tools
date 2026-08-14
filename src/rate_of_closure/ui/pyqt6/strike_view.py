"""Impact-zone (strike) viewer — face-scale display, never flight scale.

One of the three scale-separated viewers (epic #4120, V2): a 2D
face-plane view of one simulation run at face scale (millimetres). It
draws the club-face outline (with bulge/roll curvature contours when
the club has a curved face), the impact-offset marker plus a scatter of
previous strikes, the delivery vectors (club path, face normal, and
attack angle projected into the face plane), and a club-info
annotation. Every display parameter has its own checkbox with sourced
guidance, and the axis extents are hard-capped at
:data:`STRIKE_MAX_EXTENT_MM` — ball flight never appears here.

Colors come from the shared UpstreamDrift theme palette
(``get_chart_color``); no app colors are hard-coded.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from rate_of_closure.club import ClubSpec, face_sagitta, head_cog
from rate_of_closure.club.head_profiles import mass_scale, profile_for
from rate_of_closure.simulation import SimulationRun
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)
from rate_of_closure.units import FIELD_GUIDANCE

try:  # Theme palette (optional in standalone/vendored use).
    from shared.python.theme.matplotlib_style import get_chart_color
except ImportError:  # pragma: no cover - theme package always ships in-repo

    def get_chart_color(index: int) -> str:
        """Matplotlib cycle colors as a theme-neutral fallback."""
        return f"C{index % 10}"


logger = logging.getLogger(__name__)

__all__ = ["STRIKE_MAX_EXTENT_MM", "StrikeView"]

#: Hard cap on the face-plane half-extent [mm]: the strike view is a
#: face-scale display (~120 mm across a driver face) and never zooms
#: out to swing or flight scale.
STRIKE_MAX_EXTENT_MM = 120.0

#: Margin factor around the face outline.
_EXTENT_MARGIN = 1.35

#: Superellipse exponent for the face outline (matches the parametric
#: head's face-plate cross-section rounding).
_FACE_EXPONENT = 2.5

#: Delivery arrows are drawn at this fraction of the axis extent.
_ARROW_FRACTION = 0.55

#: Maximum strike-history points kept.
_MAX_HISTORY = 50

#: (checkbox attribute, label, FIELD_GUIDANCE key, default) per display
#: parameter, in bar order.
_DISPLAY_PARAMS: tuple[tuple[str, str, str, bool], ...] = (
    ("curvature", "Curvature", "strike_curvature_visible", True),
    ("vectors", "Delivery Vectors", "strike_vectors_visible", True),
    ("history", "Strike History", "strike_history_visible", True),
    ("club_info", "Club Info", "strike_club_info_visible", True),
    ("show_cg", "Show CG", "show_cg_marker", False),
)


def face_half_extents_mm(club: ClubSpec) -> tuple[float, float]:
    """(half-width, half-height) of the club's face plate, millimetres.

    Derived from the club type's head-profile face cross-section scaled
    by the club's constant-density mass factor — the same envelope the
    3D head mesh uses, so an iron face reads narrower than a driver's.
    """
    scale = mass_scale(club)
    _x, half_h, half_w, _yc = profile_for(club).sections[0]
    return half_w * scale * 1000.0, half_h * scale * 1000.0


class StrikeView(QWidget):
    """Face-plane strike view of one simulation run, face scale only."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._axes = self._figure.add_subplot(111)

        self._run: SimulationRun | None = None
        self._history: list[tuple[float, float]] = []
        self._checks: dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._build_param_bar())
        layout.addWidget(self._canvas)
        self._draw()

    def _build_param_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 4, 4, 0)
        for attr, label, guidance_key, default in _DISPLAY_PARAMS:
            check = QCheckBox(label)
            check.setChecked(default)
            check.setToolTip(FIELD_GUIDANCE[guidance_key])
            check.toggled.connect(lambda _checked: self._draw())
            self._checks[attr] = check
            bar.addWidget(check)
        bar.addStretch(1)
        return bar

    # ── public API ──────────────────────────────────────────────────
    def set_run(self, run: SimulationRun | None) -> None:
        """Adopt a run (or clear with ``None``); records strike history."""
        self._run = run
        if run is not None and run.impact_outcome.is_hit:
            strike = (
                run.config.scenario.impact_offset_toe_mm,
                run.config.scenario.impact_offset_high_mm,
            )
            self._history.append(strike)
            del self._history[:-_MAX_HISTORY]
        self._draw()

    def run(self) -> SimulationRun | None:
        """The run currently rendered, if any."""
        return self._run

    def strike_history(self) -> list[tuple[float, float]]:
        """Recorded (toe, high) strike offsets [mm], oldest first."""
        return list(self._history)

    def clear_history(self) -> None:
        """Drop the strike-history scatter."""
        self._history.clear()
        self._draw()

    def display_check(self, name: str) -> QCheckBox:
        """The display-parameter checkbox for ``name`` (test seam)."""
        return self._checks[name]

    def extents_mm(self) -> tuple[float, float]:
        """Current axis half-extents (x, y) [mm] — the scale invariant."""
        x0, x1 = self._axes.get_xlim()
        y0, y1 = self._axes.get_ylim()
        return (abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0)

    # ── drawing ─────────────────────────────────────────────────────
    def _face_outline(self, half_w: float, half_h: float) -> np.ndarray:
        theta = np.linspace(0.0, 2.0 * np.pi, 120)
        exponent = 2.0 / _FACE_EXPONENT
        x = half_w * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** exponent
        y = half_h * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** exponent
        outline: np.ndarray = np.column_stack([x, y])
        return outline

    def _draw_curvature(self, club: ClubSpec, half_w: float, half_h: float) -> None:
        if not club.has_curved_face:
            return
        toe = np.linspace(-half_w, half_w, 41)
        high = np.linspace(-half_h, half_h, 41)
        grid_toe, grid_high = np.meshgrid(toe, high)
        sag = np.vectorize(
            lambda t, h: face_sagitta(club, t * 1e-3, h * 1e-3) * 1000.0
        )(grid_toe, grid_high)
        contours = self._axes.contour(
            grid_toe,
            grid_high,
            sag,
            levels=5,
            colors=get_chart_color(7),
            linewidths=0.6,
            alpha=0.7,
        )
        self._axes.clabel(contours, inline=True, fontsize=6, fmt="%.1f mm")

    def _draw_vectors(self, run: SimulationRun, extent: float) -> None:
        delivery = run.delivery
        if delivery is None:
            return
        velocity = delivery.clubhead_velocity
        speed = float(np.linalg.norm(velocity))
        if speed <= 0.0:
            return
        path_deg = math.degrees(math.atan2(float(velocity[2]), float(velocity[0])))
        aoa_deg = math.degrees(
            math.atan2(
                float(velocity[1]),
                math.hypot(float(velocity[0]), float(velocity[2])),
            )
        )
        toe_mm = run.config.scenario.impact_offset_toe_mm
        high_mm = run.config.scenario.impact_offset_high_mm
        length = extent * _ARROW_FRACTION
        # Face-plane projection: horizontal axis = toe (+z app),
        # vertical axis = up the face (+y app).
        arrows = (
            # (dx, dy, color index, label)
            (
                math.sin(math.radians(path_deg)) * length,
                math.sin(math.radians(aoa_deg)) * length,
                0,
                f"club path {path_deg:+.1f}° / AoA {aoa_deg:+.1f}°",
            ),
            (
                float(delivery.face_normal[2]) * length,
                float(delivery.face_normal[1]) * length,
                3,
                f"face normal (loft {run.config.club.loft_deg:.1f}°)",
            ),
        )
        for dx, dy, color_index, label in arrows:
            self._axes.annotate(
                "",
                xy=(toe_mm + dx, high_mm + dy),
                xytext=(toe_mm, high_mm),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": get_chart_color(color_index),
                    "lw": 1.6,
                },
            )
            self._axes.plot(
                [], [], color=get_chart_color(color_index), lw=1.6, label=label
            )

    def _draw_cg(self, club: ClubSpec) -> None:
        """Volumetric COG projected into the face plane (themed marker).

        Horizontal = the centroid's toe (z) coordinate; vertical = its
        height relative to the face-plate center — the same head frame
        the 3D view uses, computed by the divergence-theorem
        volumetrics on the generated head.
        """
        report = head_cog(club)
        toe_mm = report.cog[2] * 1000.0
        high_mm = (report.cog[1] - report.face_center[1]) * 1000.0
        self._axes.scatter(
            [toe_mm],
            [high_mm],
            s=55,
            marker="X",
            color=get_chart_color(5),
            zorder=6,
            label=(
                f"volumetric CG (depth {report.cg_depth_m * 1000.0:.0f} mm, "
                f"height {report.cg_height_m * 1000.0:.0f} mm)"
            ),
        )

    def _club_info_text(self, club: ClubSpec) -> str:
        if club.face_bulge_radius_m is not None and club.face_roll_radius_m is not None:
            curvature = (
                f"bulge {club.face_bulge_radius_m * 1000.0:.0f} mm / "
                f"roll {club.face_roll_radius_m * 1000.0:.0f} mm"
            )
        else:
            curvature = "flat face"
        return (
            f"{club.name} — loft {club.loft_deg:.1f}°, "
            f"{club.head_mass_kg * 1000.0:.0f} g, {curvature}"
        )

    def _draw(self) -> None:
        axes = self._axes
        axes.clear()
        run = self._run
        if run is None:
            axes.set_title("Run a simulation to populate the strike view")
            axes.set_xticks([])
            axes.set_yticks([])
            self._canvas.draw_idle()
            return

        club = run.config.club
        half_w, half_h = face_half_extents_mm(club)
        extent = min(max(half_w, half_h) * _EXTENT_MARGIN, STRIKE_MAX_EXTENT_MM)

        outline = self._face_outline(half_w, half_h)
        axes.plot(
            outline[:, 0],
            outline[:, 1],
            color=get_chart_color(1),
            lw=1.5,
            label="face outline",
        )
        axes.axhline(0.0, color=get_chart_color(7), lw=0.5, alpha=0.5)
        axes.axvline(0.0, color=get_chart_color(7), lw=0.5, alpha=0.5)

        if self._checks["curvature"].isChecked():
            self._draw_curvature(club, half_w, half_h)
        if self._checks["history"].isChecked() and len(self._history) > 1:
            past = np.array(self._history[:-1])
            axes.scatter(
                past[:, 0],
                past[:, 1],
                s=18,
                color=get_chart_color(2),
                alpha=0.45,
                label="previous strikes",
            )
        is_hit = run.impact_outcome.is_hit
        if self._checks["vectors"].isChecked() and is_hit:
            self._draw_vectors(run, extent)
        if self._checks["show_cg"].isChecked():
            self._draw_cg(club)

        if is_hit:
            toe_mm = run.config.scenario.impact_offset_toe_mm
            high_mm = run.config.scenario.impact_offset_high_mm
            axes.scatter(
                [toe_mm],
                [high_mm],
                s=70,
                color=get_chart_color(4),
                zorder=5,
                label=f"impact ({toe_mm:+.1f}, {high_mm:+.1f}) mm",
            )

        if not is_hit:
            closest_mm = run.impact_outcome.closest_approach_m * 1000.0
            axes.set_title(
                f"No Impact — closest sampled approach {closest_mm:.1f} mm",
                fontsize=9,
            )
        elif self._checks["club_info"].isChecked():
            axes.set_title(self._club_info_text(club), fontsize=9)
        axes.set_xlim(-extent, extent)
        axes.set_ylim(-extent, extent)
        axes.set_aspect("equal")
        axes.set_xlabel("toe (+) / heel (−) [mm]")
        axes.set_ylabel("above (+) / below (−) center [mm]")
        axes.legend(loc="lower left", fontsize=7)
        self._canvas.draw_idle()
