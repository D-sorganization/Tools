"""Content-bearing launch-monitor scatter preview for the PyQt workspace."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QWidget

from rate_of_closure.launch_monitor_analysis import numeric_columns
from rate_of_closure.launch_monitor_linked_scatter import (
    LinkedScatterPlan,
    navigate_linked_scatter,
    plan_linked_scatter,
)
from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas


class _FrameRows(Sequence[Mapping[str, Any]]):
    """Lazy retained-row view; never materialize a second full record table."""

    def __init__(self, frame: pd.DataFrame, fields: tuple[str, ...]) -> None:
        self._row_count = len(frame)
        self._values = {field: frame[field].array for field in fields}

    def __len__(self) -> int:
        return self._row_count

    def __getitem__(self, index: int | slice) -> Mapping[str, Any]:
        if isinstance(index, slice):
            raise TypeError("linked scatter row slices are not supported")
        return {field: values[index] for field, values in self._values.items()}


def demo_frame() -> pd.DataFrame:
    """Return deterministic non-vendor records for the initial preview."""
    index = np.arange(120)
    club_speed = 38.0 + index * 0.11
    attack_angle = -4.0 + (index % 17) * 0.4
    club_path = -3.0 + (index % 13) * 0.5
    face_angle = club_path * 0.65 + np.sin(index * 0.7) * 0.8
    ball_speed = club_speed * 1.46 + attack_angle * 0.08 + np.sin(index) * 0.25
    return pd.DataFrame(
        {
            "shot_id": [f"demo-{item + 1}" for item in index],
            "session_id": np.where(index < 60, "demo-a", "demo-b"),
            "monitor_vendor": np.where(index % 2, "FlightScope", "TrackMan"),
            "observation_kind": "shot",
            "club_speed": club_speed,
            "attack_angle": attack_angle,
            "club_path": club_path,
            "face_angle": face_angle,
            "ball_speed": ball_speed,
            "carry_distance": ball_speed * 3.25 + attack_angle * 0.9,
        }
    )


class LaunchMonitorPreviewCanvas(LifecycleSafeFigureCanvas):
    """Plot the selected relationship from the retained source records."""

    selection_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        figure = Figure(figsize=(6.4, 3.2), layout="constrained")
        super().__init__(figure)
        if parent is not None:
            self.setParent(parent)
        self._axes = figure.add_subplot(111)
        self._plan: LinkedScatterPlan | None = None
        self.setAccessibleName("Launch Monitor Selected Relationship Preview")
        self.setToolTip(
            "Select the nearest displayed point. Left/Right and Home/End navigate; "
            "Escape clears. Selection is the retained row ordinal only."
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(260)
        self.mpl_connect("button_press_event", self._select_nearest)

    def set_frame(
        self,
        frame: pd.DataFrame,
        outcome: str,
        predictors: tuple[str, ...],
        selected_raw_index: int | None = None,
    ) -> LinkedScatterPlan:
        """Render finite paired values without altering the retained records."""
        numeric = tuple(numeric_columns(frame))
        if len(numeric) < 2:
            raise ValueError("linked scatter requires two eligible numeric columns")
        outcome = outcome if outcome in numeric else numeric[0]
        predictor = next(
            (name for name in predictors if name in numeric and name != outcome),
            next(name for name in numeric if name != outcome),
        )
        fields = tuple(
            dict.fromkeys(
                (predictor, outcome, "shot_id", "session_id", "monitor_vendor")
            )
        )
        plan = plan_linked_scatter(
            _FrameRows(frame, tuple(field for field in fields if field in frame)),
            predictor,
            outcome,
            selected_raw_index=selected_raw_index,
        )
        self._plan = plan
        self._axes.clear()
        self._axes.scatter(
            [point.x for point in plan.points],
            [point.y for point in plan.points],
            s=18,
            alpha=0.7,
        )
        selected = next(
            (point for point in plan.points if point.raw_index == selected_raw_index),
            None,
        )
        if selected is not None:
            self._axes.plot(
                [selected.x],
                [selected.y],
                marker="o",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="#f59e0b",
                markeredgewidth=2.5,
            )
        self._axes.set_xlabel(predictor)
        self._axes.set_ylabel(outcome)
        self._axes.set_title(f"{outcome} versus {predictor}")
        self._axes.grid(alpha=0.2)
        self.draw_idle()
        return plan

    def _select_nearest(self, event: object) -> None:
        plan = self._plan
        x_data = getattr(event, "xdata", None)
        y_data = getattr(event, "ydata", None)
        if plan is None or x_data is None or y_data is None or not plan.points:
            return
        event_x = getattr(event, "x", None)
        event_y = getattr(event, "y", None)
        if event_x is None or event_y is None:
            return
        nearest = min(
            plan.points,
            key=lambda point: sum(
                (projected - observed) ** 2
                for projected, observed in zip(
                    self._axes.transData.transform((point.x, point.y)),
                    (event_x, event_y),
                    strict=True,
                )
            ),
        )
        self.selection_changed.emit(nearest.raw_index)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:  # noqa: N802
        if event is None or self._plan is None:
            return
        commands = {
            Qt.Key.Key_Left: "previous",
            Qt.Key.Key_Right: "next",
            Qt.Key.Key_Home: "home",
            Qt.Key.Key_End: "end",
            Qt.Key.Key_Escape: "clear",
        }
        command = commands.get(event.key())
        if command is None:
            super().keyPressEvent(event)
            return
        self.selection_changed.emit(
            navigate_linked_scatter(
                self._plan,
                self._plan.selected_raw_index,
                command,
            )
        )
        event.accept()


__all__ = ["LaunchMonitorPreviewCanvas", "demo_frame"]
