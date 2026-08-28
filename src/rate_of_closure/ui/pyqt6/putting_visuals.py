"""Bounded synchronized Matplotlib views for one accepted putt result.

The top-down green carries the #4800 P6 read: the target line, the
start line the ball actually left on, the apex of the break, and the
hole-capture geometry — the 54 mm rim beside the *effective* rim the
Holmes/Penner model leaves at the arrival speed. Every one of those is
read off the accepted ``swing_sim.putting_result/2`` record, never
recomputed here, so the picture and the result rows can never disagree.
"""

from __future__ import annotations

import math
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from rate_of_closure.putting_sample_inspector import (
    PuttingNavigation,
    PuttingSamplePlan,
    navigate_putting_samples,
    nearest_putting_sample,
)
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)
from rate_of_closure.ui.pyqt6.flight_view import distance_axis
from shared.python.swing_sim.putting import (
    HOLE_RADIUS_M,
    PuttingResultDocument,
    PuttResult,
    capture_speed_mps,
)


class PuttingPlotView(QWidget):
    """Own fixed display geometry, synchronized selection, and status."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5.0, 6.0), layout="constrained")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.setAccessibleName("Interactive putt path sample inspector")
        self._canvas.setAccessibleDescription(
            "Select a displayed trajectory sample. Left and Right move; "
            "Home and End jump; Escape clears."
        )
        self._canvas.setToolTip(
            "Top-down green and synchronized speed plot. Click an exact "
            "displayed sample or use Left/Right/Home/End/Escape."
        )
        self._canvas.installEventFilter(self)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._status = QLabel("No trajectory sample selected.")
        self._status.setAccessibleName("Selected putting trajectory sample")
        self._status.setWordWrap(True)
        self._error = QLabel()
        self._error.setAccessibleName("Putting recompute error")
        self._error.setWordWrap(True)
        self._error.hide()
        self._context = QLabel("No accepted putting result is displayed.")
        self._context.setAccessibleName("Displayed putting result context")
        self._context.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._error)
        layout.addWidget(self._context)
        layout.addWidget(self._status)
        layout.addWidget(self._canvas, 1)
        self._result: PuttResult | None = None
        self._document: PuttingResultDocument | None = None
        self._plan: PuttingSamplePlan | None = None
        self._generation: object | None = None
        self._selected_raw_index: int | None = None
        self._hole_x = self._grade = self._aspect = 0.0
        self._top: Axes | None = None
        self._bottom: Axes | None = None
        self._selected_artists: tuple[Any, ...] = ()

    def canvas(self) -> FigureCanvas:
        """Return the sole focusable visual surface."""
        return self._canvas

    def path_axes(self) -> Axes:
        """Return the rendered path axes for exact pixel probes."""
        if self._top is None:
            raise RuntimeError("putting path axes are not rendered")
        return self._top

    def selected_raw_index(self) -> int | None:
        """Return the runtime-local exact raw sample selection."""
        return self._selected_raw_index

    def status_text(self) -> str:
        """Return the visible exact selected-sample status."""
        return str(self._status.text())

    def error_text(self) -> str:
        """Return the separate bounded recompute error announcement."""
        return str(self._error.text())

    def context_text(self) -> str:
        """Return the visible authority for the displayed result."""
        return str(self._context.text())

    def selected_artists(self) -> tuple[Any, ...]:
        """Return synchronized path and speed marker artists."""
        return self._selected_artists

    def set_result(
        self,
        result: PuttResult,
        plan: PuttingSamplePlan,
        *,
        generation: object,
        hole_x: float,
        grade: float,
        aspect: float,
        context_text: str,
        document: PuttingResultDocument | None = None,
    ) -> None:
        """Atomically adopt one result; only replacement clears selection."""
        replacement = generation is not self._generation
        previous = (
            self._result,
            self._document,
            self._plan,
            self._generation,
            self._hole_x,
            self._grade,
            self._aspect,
            self._selected_raw_index,
            self._status.text(),
            self._context.text(),
            self._error.text(),
            self._error.isHidden(),
        )
        try:
            self._result, self._plan, self._generation = result, plan, generation
            self._document = document
            self._hole_x, self._grade, self._aspect = hole_x, grade, aspect
            self._context.setText(f"Displayed result: {context_text}")
            if replacement:
                self._selected_raw_index = None
                self._status.setText("No trajectory sample selected.")
            self._draw()
        except Exception:
            (
                self._result,
                self._document,
                self._plan,
                self._generation,
                self._hole_x,
                self._grade,
                self._aspect,
                self._selected_raw_index,
                status,
                context,
                error,
                error_hidden,
            ) = previous
            self._status.setText(status)
            self._context.setText(context)
            self._error.setText(error)
            self._error.setHidden(error_hidden)
            if self._result is None or self._plan is None:
                self._figure.clear()
                self._top = self._bottom = None
                self._selected_artists = ()
                self._canvas.draw()
            else:
                self._draw()
            raise
        if replacement:
            self._error.clear()
            self._error.hide()

    def set_error(self, message: str) -> None:
        """Label a failed recompute without discarding accepted evidence."""
        bounded = message.strip()[:512] or "Putting inputs were rejected."
        retained = self._result is not None and self._plan is not None
        outcome = (
            "the accepted context below remains displayed"
            if retained
            else "no accepted putt is available"
        )
        self._error.setText(f"Putt recompute failed; {outcome}. {bounded}")
        self._error.show()

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:  # noqa: N802
        """Map exact navigation keys without changing focus."""
        if (
            watched is self._canvas
            and event is not None
            and event.type() == QEvent.Type.KeyPress
        ):
            key_event = event if isinstance(event, QKeyEvent) else None
            commands: dict[int, PuttingNavigation] = {
                int(Qt.Key.Key_Left): "previous",
                int(Qt.Key.Key_Right): "next",
                int(Qt.Key.Key_Home): "home",
                int(Qt.Key.Key_End): "end",
                int(Qt.Key.Key_Escape): "clear",
            }
            command = commands.get(key_event.key()) if key_event else None
            if command is not None and self._plan is not None:
                self._select(
                    navigate_putting_samples(
                        self._plan, self._selected_raw_index, command
                    )
                )
                return True
        return bool(super().eventFilter(watched, event))

    def path_display_points(self) -> tuple[tuple[int, float, float], ...]:
        """Return actual Matplotlib display pixels for bounded path samples."""
        return self._display_points(self._top)

    def select_nearest_pixel(self, axes: Axes, x_px: float, y_px: float) -> None:
        """Select within 12 rendered pixels on either synchronized axes."""
        if self._plan is not None:
            self._select(
                nearest_putting_sample(self._display_points(axes), (x_px, y_px))
            )

    def _display_points(
        self, axes: Axes | None
    ) -> tuple[tuple[int, float, float], ...]:
        if self._plan is None or axes is None:
            return ()
        if axes is self._top:
            values = ((sample.x_m, sample.y_m) for sample in self._plan.samples)
        elif axes is self._bottom:
            values = (
                (sample.cumulative_distance_m, sample.speed_mps)
                for sample in self._plan.samples
            )
        else:
            return ()
        points: list[tuple[int, float, float]] = []
        for sample, value in zip(self._plan.samples, values, strict=True):
            transformed = axes.transData.transform(value)
            points.append(
                (sample.raw_index, float(transformed[0]), float(transformed[1]))
            )
        return tuple(points)

    def _on_click(self, event: Any) -> None:
        if event.inaxes in (self._top, self._bottom):
            self.select_nearest_pixel(event.inaxes, float(event.x), float(event.y))

    def _select(self, raw_index: int | None) -> None:
        self._selected_raw_index = raw_index
        self._update_status()
        self._draw()

    def _update_status(self) -> None:
        if self._plan is None or self._selected_raw_index is None:
            self._status.setText("No trajectory sample selected.")
            return
        sample = self._plan.raw_sample(self._selected_raw_index)
        phase = "pure roll" if sample.phase == "pure-roll" else "skid"
        self._status.setText(
            f"Source sample {sample.raw_index} (zero-based); t {sample.time_s:.3f} s; "
            f"distance {sample.cumulative_distance_m:.3f} m; x {sample.x_m:.3f} m; "
            f"y {sample.y_m:.3f} m; speed {sample.speed_mps:.3f} m/s; {phase}."
        )

    def _draw(self) -> None:
        if self._plan is None or self._result is None:
            return
        self._figure.clear()
        top, bottom = self._figure.subplots(
            2, 1, height_ratios=[2.2, 1.0], sharex=False
        )
        self._top, self._bottom = top, bottom
        samples = {sample.raw_index: sample for sample in self._plan.samples}
        if self._plan.skid_polyline_indices:
            skid = [samples[index] for index in self._plan.skid_polyline_indices]
            top.plot(
                [item.x_m for item in skid],
                [item.y_m for item in skid],
                color="tab:orange",
                linewidth=2.2,
                label="Skid",
            )
        roll = [samples[index] for index in self._plan.pure_roll_polyline_indices]
        top.plot(
            [item.x_m for item in roll],
            [item.y_m for item in roll],
            color="tab:green",
            linewidth=2.2,
            label="Pure roll",
        )
        self._draw_path_context()
        bottom.plot(
            [item.cumulative_distance_m for item in self._plan.samples],
            [item.speed_mps for item in self._plan.samples],
            color="tab:blue",
        )
        bottom.axhline(
            capture_speed_mps(),
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label="Capture bound",
        )
        bottom.axvline(
            self._plan.cumulative_distance_m[self._plan.skid_end_index],
            color="tab:orange",
            linestyle=":",
            linewidth=1.0,
            label="First pure-roll sample",
        )
        bottom.set_xlabel(f"Distance rolled [{distance_axis(bottom, 'x')}]")
        bottom.set_ylabel("Speed [m/s]")
        bottom.legend(loc="best", fontsize=8)
        self._draw_selected()
        self._canvas.draw()

    def _draw_path_context(self) -> None:
        assert self._top is not None
        self._top.add_patch(
            Circle(
                (self._hole_x, 0.0),
                HOLE_RADIUS_M,
                fill=False,
                color="black",
                linewidth=1.5,
                label="Hole rim",
            )
        )
        self._draw_capture_geometry()
        self._draw_break_read()
        if self._grade > 0:
            aspect = math.radians(self._aspect)
            origin = (self._hole_x * 0.5, 0.0)
            self._top.annotate(
                "",
                xy=(origin[0] + 0.4 * math.cos(aspect), 0.4 * math.sin(aspect)),
                xytext=origin,
                arrowprops={"arrowstyle": "-|>", "color": "grey"},
            )
        self._top.axhline(0.0, color="grey", linewidth=0.8, linestyle="-.")
        self._top.set_xlabel(f"Along putt line [{distance_axis(self._top, 'x')}]")
        self._top.set_ylabel(f"Lateral [{distance_axis(self._top, 'y')}] (left +)")
        self._top.set_title("Top-down green")
        self._top.axis("equal")
        self._top.legend(loc="best", fontsize=8)

    def _draw_capture_geometry(self) -> None:
        """The rim the ball could actually use, from the accepted record.

        ``effective_hole_radius_m`` is the Holmes/Penner radius at the
        speed the ball crossed closest to the centre: a fast putt sees
        a small opening even though the hole is always 54 mm.
        """
        if self._top is None or self._document is None:
            return
        radius = self._document.effective_hole_radius_m
        if radius <= 0.0:
            return
        self._top.add_patch(
            Circle(
                (self._hole_x, 0.0),
                radius,
                fill=False,
                color="tab:red",
                linestyle="--",
                linewidth=1.2,
                label="Effective rim at arrival speed",
            )
        )

    def _draw_break_read(self) -> None:
        """The start line and the apex of the break, from the record."""
        if self._top is None or self._document is None or self._result is None:
            return
        document = self._document
        # Start line: the direction the ball left the face, drawn over
        # the along-line span the putt actually covered. The record's
        # azimuth is right-positive while y is left-positive.
        reach = max(self._hole_x, max(self._result.path_x_m))
        self._top.plot(
            (0.0, reach),
            (0.0, -reach * math.tan(math.radians(document.start_azimuth_deg))),
            color="tab:purple",
            linestyle=":",
            linewidth=1.2,
            label="Start line",
        )
        self._top.scatter(
            document.apex_break_at_m,
            document.apex_break_m,
            s=60,
            marker="D",
            facecolors="none",
            edgecolors="tab:blue",
            linewidths=1.6,
            zorder=8,
            label="Apex break",
        )

    def _draw_selected(self) -> None:
        if (
            self._plan is None
            or self._top is None
            or self._bottom is None
            or self._selected_raw_index is None
        ):
            self._selected_artists = ()
            return
        sample = self._plan.raw_sample(self._selected_raw_index)
        path_artist = self._top.scatter(
            sample.x_m,
            sample.y_m,
            s=90,
            facecolors="none",
            edgecolors="#eab308",
            linewidths=2.5,
            zorder=9,
        )
        speed_artist = self._bottom.scatter(
            sample.cumulative_distance_m,
            sample.speed_mps,
            s=75,
            facecolors="none",
            edgecolors="#eab308",
            linewidths=2.5,
            zorder=9,
        )
        self._selected_artists = (path_artist, speed_artist)


__all__ = ["PuttingPlotView"]
