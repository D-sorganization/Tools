"""Panel drawing mixin for the scale-locked ball-flight viewer."""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtWidgets import QCheckBox

from rate_of_closure.simulation.targets import TargetRegion, hold_stats
from rate_of_closure.ui.course import CourseLayout
from rate_of_closure.ui.pyqt6.course_scene import (
    draw_course_ground_3d,
    draw_course_side,
    draw_course_top,
    draw_target_region_top,
)
from rate_of_closure.ui.pyqt6.flight_playback_rendering import FlightPlaybackArtists
from rate_of_closure.ui.pyqt6.flight_view_axes import distance_axis
from rate_of_closure.ui.pyqt6.flight_wind_overlay import (
    plot_wind_pair_2d,
    plot_wind_pair_3d,
)
from rate_of_closure.units import format_distance_m

try:
    from shared.python.theme.matplotlib_style import get_chart_color
except ImportError:  # pragma: no cover - theme package ships in-repo

    def get_chart_color(index: int) -> str:
        """Matplotlib cycle colors as a theme-neutral fallback."""
        return f"C{index % 10}"


class FlightViewPanelsMixin:
    """Render side, top-down, and 3D panels from one app-frame trajectory."""

    _checks: dict[str, QCheckBox]
    _course_layout: CourseLayout
    _playback_artists: FlightPlaybackArtists
    _scatter: tuple[np.ndarray, np.ndarray] | None
    _target_region: TargetRegion | None
    comparison_positions: np.ndarray

    @staticmethod
    def _distance_axis(axes: Any, which: str) -> str:
        return str(distance_axis(axes, which))

    @staticmethod
    def _annotate_landing(axes: Any, x: float, y: float, text: str) -> None:
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

    def _draw_side(
        self, axes: Any, positions: np.ndarray, extents: tuple[float, float, float]
    ) -> None:
        carry_ext, height_ext, _ = extents
        draw_course_side(
            axes,
            carry_ext,
            layout=self._course_layout,
            elements=self._checks["course"].isChecked(),
        )
        plot_wind_pair_2d(axes, positions, self.comparison_positions, 1)
        self._playback_artists.add_2d(axes, 1)
        if self._checks["apex"].isChecked():
            apex_index = int(np.argmax(positions[:, 1]))
            axes.scatter(
                [positions[apex_index, 0]],
                [positions[apex_index, 1]],
                s=30,
                color=get_chart_color(3),
                zorder=5,
            )
            axes.annotate(
                f"apex {positions[apex_index, 1]:.1f} m",
                xy=(positions[apex_index, 0], positions[apex_index, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=get_chart_color(3),
            )
        if self._checks["landing"].isChecked():
            landing_carry = float(positions[-1, 0])
            landing_height = float(positions[-1, 1])
            self._annotate_landing(
                axes,
                landing_carry,
                landing_height,
                f"carry {format_distance_m(landing_carry)}",
            )
        axes.set_xlim(0.0, carry_ext)
        axes.set_ylim(0.0, height_ext)
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel(f"carry [{self._distance_axis(axes, 'x')}]", fontsize=8)
        axes.set_ylabel("height [m]", fontsize=8)
        axes.set_title("Side profile", fontsize=9)
        if len(self.comparison_positions):
            axes.legend(fontsize=7)
        axes.tick_params(labelsize=7)

    def _draw_top(
        self, axes: Any, positions: np.ndarray, extents: tuple[float, float, float]
    ) -> None:
        carry_ext, _, lateral_ext = extents
        draw_course_top(
            axes,
            carry_ext,
            lateral_ext,
            layout=self._course_layout,
            elements=self._checks["course"].isChecked(),
        )
        plot_wind_pair_2d(axes, positions, self.comparison_positions, 2)
        self._playback_artists.add_2d(axes, 2)
        axes.axhline(0.0, color=get_chart_color(7), lw=0.6, alpha=0.6)
        if self._checks["landing"].isChecked():
            landing_carry = float(positions[-1, 0])
            landing_lateral = float(positions[-1, 2])
            direction = "+" if landing_lateral >= 0 else "-"
            self._annotate_landing(
                axes,
                landing_carry,
                landing_lateral,
                f"lateral {direction}{format_distance_m(abs(landing_lateral))}",
            )
        title = "Top-down"
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
                percent = 100.0 * held / total if total else float("nan")
                title = (
                    f"Top-down — {held}/{total} shots hold the target ({percent:.0f}%)"
                )
        axes.set_xlim(0.0, carry_ext)
        axes.set_ylim(-lateral_ext, lateral_ext)
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel(f"carry [{self._distance_axis(axes, 'x')}]", fontsize=8)
        axes.set_ylabel(f"right (+) [{self._distance_axis(axes, 'y')}]", fontsize=8)
        axes.set_title(title, fontsize=9)
        if len(self.comparison_positions):
            axes.legend(fontsize=7)
        axes.tick_params(labelsize=7)

    def _draw_3d(
        self, axes: Any, positions: np.ndarray, extents: tuple[float, float, float]
    ) -> None:
        carry_ext, height_ext, lateral_ext = extents
        draw_course_ground_3d(
            axes,
            carry_ext,
            layout=self._course_layout,
            elements=self._checks["course"].isChecked(),
        )
        plot_wind_pair_3d(axes, positions, self.comparison_positions)
        self._playback_artists.add_3d(axes)
        if self._checks["landing"].isChecked():
            axes.scatter(
                [positions[-1, 2]],
                [positions[-1, 0]],
                [positions[-1, 1]],
                s=40,
                color=get_chart_color(4),
            )
        axes.set_xlim(-lateral_ext, lateral_ext)
        axes.set_ylim(0.0, carry_ext)
        axes.set_zlim(0.0, height_ext)
        axes.set_box_aspect((2.0 * lateral_ext, carry_ext, height_ext))
        axes.set_xlabel(f"z — right [{self._distance_axis(axes, 'x')}]", fontsize=7)
        axes.set_ylabel(f"x — target [{self._distance_axis(axes, 'y')}]", fontsize=7)
        axes.set_zlabel("y — up [m]", fontsize=7)
        axes.set_title("3D trajectory", fontsize=9)
        if len(self.comparison_positions):
            axes.legend(fontsize=7)
        axes.tick_params(labelsize=6)


__all__ = ["FlightViewPanelsMixin"]
