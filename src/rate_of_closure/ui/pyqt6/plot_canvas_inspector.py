"""Exact point/bin interaction for one managed Matplotlib plot pane."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from matplotlib.artist import Artist
from matplotlib.backend_bases import MouseEvent
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QLabel

from rate_of_closure.plot_point_inspector import (
    DEFAULT_PLOT_HIT_RADIUS_PX,
    HistogramSelection,
    PlotInspectionPlan,
    PlotNavigation,
    PlotSelection,
    SeriesSelection,
    histogram_bin_at_data,
    navigate_plot_selection,
    nearest_series_point,
    plan_plot_inspection,
)
from rate_of_closure.plotting import PlotData

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas


class PlotCanvasInspectorMixin:
    """Presentation-only inspection; scientific PlotData remains unchanged."""

    if TYPE_CHECKING:
        _canvas: LifecycleSafeFigureCanvas
        _data: PlotData | None
        _figure: Figure
        _inspection_error: str | None
        _inspection_plan: PlotInspectionPlan | None
        _inspection_status: QLabel
        _selection: PlotSelection | None
        _selection_artists: list[Artist]

    def _initialize_plot_inspector(self) -> QLabel:
        self._inspection_plan = None
        self._inspection_error = None
        self._selection = None
        self._selection_artists = []
        status = QLabel(
            "No exact point selected. Click within 12 pixels or use the keyboard."
        )
        status.setWordWrap(True)
        status.setAccessibleName("Selected plot evidence")
        self._inspection_status = status
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.setAccessibleDescription(
            "Exact plot evidence inspector. Arrow keys navigate series points or "
            "histogram bins; Home and End jump; Escape clears."
        )
        self._canvas.installEventFilter(cast(QObject, self))
        self._canvas.mpl_connect("button_press_event", self._on_inspection_click)
        return status

    @staticmethod
    def _plan_for_data(
        data: PlotData,
    ) -> tuple[PlotInspectionPlan | None, str | None]:
        try:
            plan = plan_plot_inspection(
                data.spec.kind,
                data.x,
                [
                    {"label": label, "values": values}
                    for label, values in data.series.items()
                ],
            )
        except ValueError as exc:
            return None, str(exc)[:512]
        return plan, None

    def selected_evidence(self) -> PlotSelection | None:
        """Return the selected exact point or derived bin identity."""
        return self._selection

    def inspection_status(self) -> str:
        """Return the visible and accessible exact-evidence description."""
        return str(self._inspection_status.text())

    def _adopt_selection(self, selection: PlotSelection | None) -> None:
        previous = self._selection
        self._selection = selection
        try:
            self._draw_selection_marker()
        except Exception as exc:  # noqa: BLE001 - presentation rollback boundary
            self._selection = previous
            self._update_inspection_status()
            prior = self._inspection_status.text()
            self._inspection_status.setText(
                f"Selection failed; prior evidence retained: {str(exc)[:256]}. {prior}"
            )
            return
        self._update_inspection_status()
        self._canvas.setFocus()
        self._canvas.draw_idle()

    def _on_inspection_click(self, event: MouseEvent) -> None:
        plan = self._inspection_plan
        if (
            plan is None
            or not self._figure.axes
            or event.inaxes is not self._figure.axes[0]
            or event.x is None
            or event.y is None
        ):
            return
        axes = self._figure.axes[0]
        selection: PlotSelection | None
        if plan.kind == "series":
            projected = [
                [
                    axes.transData.transform((x, y))
                    for x, y in zip(plan.x, item.values, strict=True)
                ]
                for item in plan.series
            ]
            selection = nearest_series_point(
                plan,
                projected,
                (event.x, event.y),
                DEFAULT_PLOT_HIT_RADIUS_PX * self._canvas.devicePixelRatioF(),
            )
        elif event.xdata is not None and event.ydata is not None:
            selection = histogram_bin_at_data(plan, event.xdata, event.ydata)
        else:
            selection = None
        if selection is not None:
            self._adopt_selection(selection)

    def _handle_inspection_event(
        self, watched: QObject | None, event: QEvent | None
    ) -> bool:
        if (
            watched is self._canvas
            and isinstance(event, QKeyEvent)
            and event.type() == QEvent.Type.KeyPress
            and self._inspection_plan is not None
        ):
            commands: dict[int, PlotNavigation] = {
                Qt.Key.Key_Left: "previous",
                Qt.Key.Key_Right: "next",
                Qt.Key.Key_Up: "up",
                Qt.Key.Key_Down: "down",
                Qt.Key.Key_Home: "home",
                Qt.Key.Key_End: "end",
                Qt.Key.Key_Escape: "clear",
            }
            command = commands.get(event.key())
            if command:
                self._adopt_selection(
                    navigate_plot_selection(
                        self._inspection_plan, self._selection, command
                    )
                )
                return True
        return False

    def _draw_selection_marker(self) -> None:
        plan, selection = self._inspection_plan, self._selection
        candidate: list[Artist] = []
        if plan is not None and selection is not None and self._figure.axes:
            axes = self._figure.axes[0]
            if isinstance(selection, SeriesSelection) and plan.kind == "series":
                series = plan.series[selection.series_index]
                candidate.append(
                    axes.scatter(
                        [plan.x[selection.raw_index]],
                        [series.values[selection.raw_index]],
                        s=54,
                        facecolor="#f8fafc",
                        edgecolor="#0f172a",
                        linewidth=1.5,
                        zorder=20,
                    )
                )
            elif isinstance(selection, HistogramSelection) and plan.kind == "histogram":
                item = plan.bins[selection.bin_index]
                candidate.append(
                    axes.axvspan(
                        item.lower,
                        item.upper,
                        facecolor="none",
                        edgecolor="#f8fafc",
                        linewidth=2.0,
                        zorder=20,
                    )
                )
        for artist in self._selection_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._selection_artists = candidate

    def _update_inspection_status(self) -> None:
        plan, selection, data = self._inspection_plan, self._selection, self._data
        if plan is None or data is None or selection is None:
            message = (
                f"Exact inspection unavailable: {self._inspection_error}."
                if self._inspection_error
                else "No exact point selected. Click within 12 pixels; use arrow "
                "keys, Home, End, or Escape."
            )
            self._inspection_status.setText(message)
            return
        if isinstance(selection, SeriesSelection) and plan.kind == "series":
            series = plan.series[selection.series_index]
            self._inspection_status.setText(
                f"Series {series.label}; source point {selection.raw_index + 1}/"
                f"{plan.raw_count}; {data.x_label} "
                f"{plan.x[selection.raw_index]:.6g}; {data.y_label} "
                f"{series.values[selection.raw_index]:.6g}."
            )
            return
        assert isinstance(selection, HistogramSelection) and plan.kind == "histogram"
        item = plan.bins[selection.bin_index]
        closing = "]" if item.index == len(plan.bins) - 1 else ")"
        self._inspection_status.setText(
            f"Histogram bin {item.index + 1}/{len(plan.bins)}; {data.x_label} "
            f"[{item.lower:.6g}, {item.upper:.6g}{closing}; count {item.count}."
        )


__all__ = ["PlotCanvasInspectorMixin"]
