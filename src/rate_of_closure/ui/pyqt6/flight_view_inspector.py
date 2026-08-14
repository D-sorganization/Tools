"""Exact sample picking and keyboard navigation for :mod:`flight_view`."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PyQt6.QtCore import QEvent, QObject, Qt

from rate_of_closure.flight_sample_inspector import (
    FlightSamplePlan,
    navigate_flight_samples,
    nearest_flight_sample,
)
from rate_of_closure.ui.course import get_chart_color

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent

    from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas


class FlightViewInspectorMixin:
    """Presentation-only selection behavior; calm comparison is never selectable."""

    if TYPE_CHECKING:
        _canvas: LifecycleSafeFigureCanvas
        _sample_plan: FlightSamplePlan | None
        _selected_raw_index: int | None
        _inspector_axes: dict[str, Axes]

        def _draw(self, *, sync: bool = False) -> None: ...

        def set_playback_time(self, time_s: float) -> None: ...

        def sampleSelected(self, raw_index: int) -> None: ...  # noqa: N802

    def _initialize_sample_inspector(self) -> None:
        self._sample_plan = None
        self._selected_raw_index = None
        self._inspector_axes = {}
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.setAccessibleName("Flight trajectory sample inspector")
        self._canvas.setAccessibleDescription(
            "Current primary side and top profiles. Click within 12 pixels; "
            "Left and Right move, Home and End jump, Escape clears."
        )
        self._canvas.setToolTip(
            "Click the current primary side/top trajectory. Left/Right move exact "
            "samples; Home/End jump; Escape clears. Calm ghost is comparison-only."
        )
        self._canvas.installEventFilter(cast(QObject, self))
        self._canvas.mpl_connect("button_press_event", self._on_sample_click)

    def set_sample_plan(self, plan: FlightSamplePlan | None) -> None:
        """Adopt exact primary sample authority and clear prior selection."""
        if plan is not None and not isinstance(plan, FlightSamplePlan):
            raise TypeError("sample plan must be a FlightSamplePlan or None")
        self._sample_plan = plan
        self._selected_raw_index = None

    def selected_raw_index(self) -> int | None:
        """Current exact raw index, if selected."""
        return self._selected_raw_index

    def _select_raw_sample(self, raw_index: int | None) -> None:
        plan = self._sample_plan
        if plan is None:
            return
        if raw_index is not None:
            plan.raw_sample(raw_index)
        previous = self._selected_raw_index
        self._selected_raw_index = raw_index
        try:
            self._draw(sync=True)
        except Exception as exc:
            self._selected_raw_index = previous
            restoration_failed = False
            try:
                self._draw(sync=True)
            except Exception:
                restoration_failed = True
            if restoration_failed:
                self._canvas.pause_idle_draws()
            else:
                self._canvas.resume_idle_draws()
            self.sampleSelectionFailed.emit(  # type: ignore[attr-defined]
                str(exc)[:512], restoration_failed
            )
            return
        self._canvas.resume_idle_draws()
        self._canvas.setFocus()
        self.sampleSelected.emit(-1 if raw_index is None else raw_index)  # type: ignore[attr-defined]

    def _on_sample_click(self, event: MouseEvent) -> None:
        plan = self._sample_plan
        name = next(
            (key for key, axes in self._inspector_axes.items() if axes is event.inaxes),
            None,
        )
        if (
            plan is None
            or name not in {"side", "top"}
            or event.x is None
            or event.y is None
        ):
            return
        projected = []
        for sample in plan.samples:
            vertical = sample.height_m if name == "side" else sample.right_m
            x_pixel, y_pixel = self._inspector_axes[name].transData.transform(
                (sample.downrange_m, vertical)
            )
            projected.append(("current", sample.raw_index, x_pixel, y_pixel))
        selection = nearest_flight_sample(plan, projected, (event.x, event.y))
        if selection is not None:
            self._canvas.setFocus()
            self._select_raw_sample(selection.raw_index)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if watched is self._canvas and event.type() == QEvent.Type.KeyPress:
            key = cast(object, event).key()  # type: ignore[attr-defined]
            commands = {
                Qt.Key.Key_Left: "previous",
                Qt.Key.Key_Right: "next",
                Qt.Key.Key_Home: "home",
                Qt.Key.Key_End: "end",
                Qt.Key.Key_Escape: "clear",
            }
            command = commands.get(key)
            if command and self._sample_plan is not None:
                self._select_raw_sample(
                    navigate_flight_samples(
                        self._sample_plan, self._selected_raw_index, command
                    )
                )
                return True
        return bool(super().eventFilter(watched, event))  # type: ignore[misc]

    def _draw_sample_marker(self, axes: Axes, name: str) -> None:
        self._inspector_axes[name] = axes
        if self._sample_plan is None or self._selected_raw_index is None:
            return
        sample = self._sample_plan.raw_sample(self._selected_raw_index)
        vertical = sample.height_m if name == "side" else sample.right_m
        axes.scatter(
            [sample.downrange_m], [vertical], s=48, color=get_chart_color(4), zorder=20
        )
