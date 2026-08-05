"""Dedicated ball-flight viewer — flight-scale display (tens of metres).

One of the three scale-separated viewers (epic #4120, V2): a side
profile (height vs carry), a top-down view (lateral vs carry), and the
3D trajectory polyline, all auto-scaled to the flight regime, with the
landing point annotated. Each panel and annotation has its own display
checkbox with sourced guidance. The view accepts either a full
:class:`~rate_of_closure.simulation.SimulationRun` or a bare
trajectory, so the standalone Flight Explorer reuses it with no swing.

Colors come from the shared UpstreamDrift theme palette
(``get_chart_color``); no app colors are hard-coded.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from rate_of_closure.simulation import SimulationRun
from rate_of_closure.simulation.targets import TargetRegion, hold_stats
from rate_of_closure.ui.course import CourseLayout
from rate_of_closure.ui.pyqt6.course_scene import (
    draw_course_ground_3d,
    draw_course_side,
    draw_course_top,
    draw_target_region_top,
)
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)
from rate_of_closure.units import (
    DISTANCE_UNITS,
    FIELD_GUIDANCE,
    display_distance_unit,
    format_distance_m,
)

try:  # Theme palette (optional in standalone/vendored use).
    from shared.python.theme.matplotlib_style import get_chart_color
except ImportError:  # pragma: no cover - theme package always ships in-repo

    def get_chart_color(index: int) -> str:
        """Matplotlib cycle colors as a theme-neutral fallback."""
        return f"C{index % 10}"


logger = logging.getLogger(__name__)

__all__ = ["FlightView"]

#: Minimum plotted extents so degenerate flights stay readable.
_MIN_CARRY_M = 10.0
_MIN_HEIGHT_M = 5.0
_MIN_LATERAL_M = 5.0

#: (checkbox attribute, label, FIELD_GUIDANCE key, default) per display
#: parameter, in bar order.
_DISPLAY_PARAMS: tuple[tuple[str, str, str, bool], ...] = (
    ("side", "Side Profile", "flight_side_visible", True),
    ("top", "Top-Down", "flight_top_visible", True),
    ("three_d", "3D Trajectory", "flight_3d_visible", True),
    ("landing", "Landing Point", "flight_landing_visible", True),
    ("apex", "Apex", "flight_apex_visible", True),
    ("course", "Course Elements", "course_visible", True),
)


def distance_axis(axes: object, which: str) -> str:
    """Format a metres axis in the display distance unit (#4125 H6).

    Data stays in canonical metres; only tick labels convert. Returns
    the axis label text (e.g. ``carry [yd]``).
    """
    from matplotlib.ticker import FuncFormatter

    unit = display_distance_unit()
    factor = DISTANCE_UNITS[unit]
    formatter = FuncFormatter(lambda value, _pos: f"{value / factor:.0f}")
    getattr(axes, f"{which}axis").set_major_formatter(formatter)
    return str(unit)


class FlightView(QWidget):
    """Flight-scale trajectory viewer: side + top-down 2D panels + 3D."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(7, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)

        self._positions: np.ndarray = np.zeros((0, 3))
        self._run: SimulationRun | None = None
        self._checks: dict[str, QCheckBox] = {}
        self._course_layout = CourseLayout()
        self._target_region: TargetRegion | None = None
        # (carry, lateral) landing scatter [m] from the Variation engine.
        self._scatter: tuple[np.ndarray, np.ndarray] | None = None

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
        """Adopt the flight trajectory of a full simulation run."""
        self._run = run
        self._positions = (
            np.zeros((0, 3)) if run is None else run.flight_positions.copy()
        )
        self._draw()

    def set_trajectory(self, positions: np.ndarray | None) -> None:
        """Adopt an (N, 3) app-frame trajectory (or clear with ``None``).

        App frame: x downrange along the target line [m], y up [m],
        z right of target [m].
        """
        self._run = None
        self._positions = (
            np.zeros((0, 3)) if positions is None else np.asarray(positions, float)
        )
        self._draw()

    def trajectory(self) -> np.ndarray:
        """The (N, 3) app-frame trajectory currently rendered."""
        return self._positions

    def display_check(self, name: str) -> QCheckBox:
        """The display-parameter checkbox for ``name`` (test seam)."""
        return self._checks[name]

    def course_layout(self) -> CourseLayout:
        """The course furniture layout rendered behind the flight."""
        return self._course_layout

    def set_course_layout(self, layout: CourseLayout) -> None:
        """Adopt a course layout (H7b target edits drive this) and redraw."""
        self._course_layout = layout
        self._draw()

    def set_target_region(self, region: TargetRegion | None) -> None:
        """Show (or clear) the target-region boundary in the top-down view."""
        self._target_region = region
        self._draw()

    def target_region(self) -> TargetRegion | None:
        """The target region currently overlaid, if any."""
        return self._target_region

    def set_landing_scatter(
        self, carries_m: np.ndarray | None, laterals_m: np.ndarray | None = None
    ) -> None:
        """Overlay a Variation landing scatter (or clear with ``None``).

        #4125 H7b tie-in: when a target region is set, the top-down
        title reports the share of shots holding the target — the
        headline Monte-Carlo output.
        """
        if carries_m is None or laterals_m is None:
            self._scatter = None
        else:
            self._scatter = (
                np.asarray(carries_m, float),
                np.asarray(laterals_m, float),
            )
        self._draw()

    def extents_m(self) -> tuple[float, float, float]:
        """(carry, height, lateral) plot extents [m] — flight regime."""
        pos = self._positions
        if not len(pos):
            return (_MIN_CARRY_M, _MIN_HEIGHT_M, _MIN_LATERAL_M)
        carry = max(_MIN_CARRY_M, float(np.max(pos[:, 0])) * 1.05)
        lateral = max(_MIN_LATERAL_M, float(np.max(np.abs(pos[:, 2]))) * 1.3)
        if self._scatter is not None and len(self._scatter[0]):
            carries, laterals = self._scatter
            finite = np.isfinite(carries) & np.isfinite(laterals)
            if np.any(finite):
                carry = max(carry, float(np.max(carries[finite])) * 1.05)
                lateral = max(lateral, float(np.max(np.abs(laterals[finite]))) * 1.1)
        return (
            carry,
            max(_MIN_HEIGHT_M, float(np.max(pos[:, 1])) * 1.2),
            lateral,
        )

    # ── drawing ─────────────────────────────────────────────────────
    def _annotate_landing(self, axes, x: float, y: float, text: str) -> None:  # type: ignore[no-untyped-def]
        axes.scatter([x], [y], s=45, color=get_chart_color(4), zorder=5)
        axes.annotate(
            text,
            xy=(x, y),
            xytext=(-8, 10),
            textcoords="offset points",
            fontsize=7,
            ha="right",
            color=get_chart_color(4),
        )

    def _draw_side(self, axes, pos: np.ndarray, extents) -> None:  # type: ignore[no-untyped-def]
        carry_ext, height_ext, _ = extents
        # Course-styled ground (#4125 H7a): grass band + green/flag.
        draw_course_side(
            axes,
            carry_ext,
            layout=self._course_layout,
            elements=self._checks["course"].isChecked(),
        )
        axes.plot(pos[:, 0], pos[:, 1], color=get_chart_color(2), lw=1.6)
        if self._checks["apex"].isChecked():
            apex_index = int(np.argmax(pos[:, 1]))
            axes.scatter(
                [pos[apex_index, 0]],
                [pos[apex_index, 1]],
                s=30,
                color=get_chart_color(3),
                zorder=5,
            )
            axes.annotate(
                f"apex {pos[apex_index, 1]:.1f} m",
                xy=(pos[apex_index, 0], pos[apex_index, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=get_chart_color(3),
            )
        if self._checks["landing"].isChecked():
            self._annotate_landing(
                axes, pos[-1, 0], pos[-1, 1], f"carry {format_distance_m(pos[-1, 0])}"
            )
        axes.set_xlim(0.0, carry_ext)
        axes.set_ylim(0.0, height_ext)
        axes.set_xlabel(f"carry [{distance_axis(axes, 'x')}]", fontsize=8)
        axes.set_ylabel("height [m]", fontsize=8)
        axes.set_title("Side profile", fontsize=9)
        axes.tick_params(labelsize=7)

    def _draw_top(self, axes, pos: np.ndarray, extents) -> None:  # type: ignore[no-untyped-def]
        carry_ext, _, lateral_ext = extents
        # Course-styled ground (#4125 H7a): rough, fairway strip, green.
        draw_course_top(
            axes,
            carry_ext,
            lateral_ext,
            layout=self._course_layout,
            elements=self._checks["course"].isChecked(),
        )
        axes.plot(pos[:, 0], pos[:, 2], color=get_chart_color(2), lw=1.6)
        axes.axhline(0.0, color=get_chart_color(7), lw=0.6, alpha=0.6)
        if self._checks["landing"].isChecked():
            self._annotate_landing(
                axes,
                pos[-1, 0],
                pos[-1, 2],
                f"lateral {'+' if pos[-1, 2] >= 0 else '-'}"
                f"{format_distance_m(abs(pos[-1, 2]))}",
            )
        title = "Top-down"
        # Target region + Variation landing scatter (#4125 H7b).
        if self._target_region is not None:
            draw_target_region_top(axes, self._target_region)
        if self._scatter is not None:
            carries, laterals = self._scatter
            axes.scatter(
                carries,
                laterals,
                s=10,
                alpha=0.55,
                color=get_chart_color(0),
                edgecolors="none",
                zorder=4,
            )
            if self._target_region is not None:
                held, total = hold_stats(carries, laterals, self._target_region)
                pct = 100.0 * held / total if total else float("nan")
                title = f"Top-down — {held}/{total} shots hold the target ({pct:.0f}%)"
        axes.set_xlim(0.0, carry_ext)
        axes.set_ylim(-lateral_ext, lateral_ext)
        axes.set_xlabel(f"carry [{distance_axis(axes, 'x')}]", fontsize=8)
        axes.set_ylabel(f"right (+) [{distance_axis(axes, 'y')}]", fontsize=8)
        axes.set_title(title, fontsize=9)
        axes.tick_params(labelsize=7)

    def _draw_3d(self, axes, pos: np.ndarray, extents) -> None:  # type: ignore[no-untyped-def]
        carry_ext, height_ext, lateral_ext = extents
        # Course-styled ground plane (#4125 H7a).
        draw_course_ground_3d(
            axes,
            carry_ext,
            layout=self._course_layout,
            elements=self._checks["course"].isChecked(),
        )
        # Display axes: (z right, x downrange, y up) like the swing view.
        axes.plot(pos[:, 2], pos[:, 0], pos[:, 1], color=get_chart_color(2), lw=1.6)
        if self._checks["landing"].isChecked():
            axes.scatter(
                [pos[-1, 2]],
                [pos[-1, 0]],
                [pos[-1, 1]],
                s=40,
                color=get_chart_color(4),
            )
        axes.set_xlim(-lateral_ext, lateral_ext)
        axes.set_ylim(0.0, carry_ext)
        axes.set_zlim(0.0, height_ext)
        axes.set_xlabel(f"z — right [{distance_axis(axes, 'x')}]", fontsize=7)
        axes.set_ylabel(f"x — target [{distance_axis(axes, 'y')}]", fontsize=7)
        axes.set_zlabel("y — up [m]", fontsize=7)
        axes.set_title("3D trajectory", fontsize=9)
        axes.tick_params(labelsize=6)

    def _draw(self) -> None:
        self._figure.clear()
        pos = self._positions
        panels = [
            name
            for name in ("side", "top", "three_d")
            if self._checks[name].isChecked()
        ]
        if not len(pos) or not panels:
            axes = self._figure.add_subplot(111)
            axes.set_xticks([])
            axes.set_yticks([])
            if not len(pos) and self._run is not None:
                title = (
                    "No Flight — fixed-ball contact was missed; "
                    "the swing remains available"
                )
            elif not len(pos):
                title = "Run a flight to populate the view"
            else:
                title = "Enable a panel to display the flight"
            axes.set_title(title)
            self._canvas.draw_idle()
            return

        want_3d = "three_d" in panels
        left = [name for name in panels if name != "three_d"]
        grid = self._figure.add_gridspec(
            max(len(left), 1), 2 if want_3d and left else 1
        )
        extents = self.extents_m()
        for row, name in enumerate(left):
            axes = self._figure.add_subplot(grid[row, 0])
            if name == "side":
                self._draw_side(axes, pos, extents)
            else:
                self._draw_top(axes, pos, extents)
        if want_3d:
            spec = grid[:, 1] if left else grid[:, 0]
            axes_3d = self._figure.add_subplot(spec, projection="3d")
            self._draw_3d(axes_3d, pos, extents)
        self._canvas.draw_idle()
