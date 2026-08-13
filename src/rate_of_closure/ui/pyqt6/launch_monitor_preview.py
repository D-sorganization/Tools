"""Content-bearing launch-monitor scatter preview for the PyQt workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget

from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas


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

    def __init__(self, parent: QWidget | None = None) -> None:
        figure = Figure(figsize=(6.4, 3.2), layout="constrained")
        super().__init__(figure)
        if parent is not None:
            self.setParent(parent)
        self._axes = figure.add_subplot(111)
        self.setAccessibleName("Launch Monitor Selected Relationship Preview")
        self.setMinimumHeight(260)

    def set_frame(
        self, frame: pd.DataFrame, outcome: str, predictors: tuple[str, ...]
    ) -> None:
        """Render finite paired values without altering the retained records."""
        numeric = tuple(frame.select_dtypes(include="number").columns)
        outcome = outcome if outcome in numeric else numeric[0]
        predictor = (
            predictors[0]
            if predictors
            else next((name for name in numeric if name != outcome), outcome)
        )
        x_values = pd.to_numeric(frame[predictor], errors="coerce").to_numpy()
        y_values = pd.to_numeric(frame[outcome], errors="coerce").to_numpy()
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        self._axes.clear()
        self._axes.scatter(x_values[finite], y_values[finite], s=18, alpha=0.7)
        self._axes.set_xlabel(predictor)
        self._axes.set_ylabel(outcome)
        self._axes.set_title(f"{outcome} versus {predictor}")
        self._axes.grid(alpha=0.2)
        self.draw_idle()


__all__ = ["LaunchMonitorPreviewCanvas", "demo_frame"]
