"""Small controller keeping target editing, rendering, and residuals aligned."""

from __future__ import annotations

import numpy as np

from rate_of_closure.simulation.targets import layout_for_region
from rate_of_closure.ui.pyqt6.flight_view import FlightView
from rate_of_closure.ui.pyqt6.spatial_target_panel import SpatialTargetPanel
from rate_of_closure.ui.pyqt6.spatial_target_trajectory import trajectory_target_miss
from shared.python.swing_sim.solver import SpatialTarget


class SpatialTargetWorkflow:
    """Coordinate one editor and one plot without embedding physics in a tab."""

    def __init__(self, panel: SpatialTargetPanel, view: FlightView) -> None:
        if not isinstance(panel, SpatialTargetPanel):
            raise TypeError("panel must be a SpatialTargetPanel")
        if not isinstance(view, FlightView):
            raise TypeError("view must be a FlightView")
        self._panel = panel
        self._view = view
        self._positions: np.ndarray = np.zeros((0, 3), dtype=float)
        panel.targetChanged.connect(self._on_target_changed)
        self._on_target_changed(panel.target())

    def set_trajectory(self, positions_m: np.ndarray | None) -> None:
        """Adopt the latest app-frame trajectory and refresh residuals."""
        positions = (
            np.zeros((0, 3), dtype=float)
            if positions_m is None
            else np.asarray(positions_m, dtype=float)
        )
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("positions_m must have shape (N, 3)")
        self._positions = positions
        self._refresh_miss()

    def set_unavailable(self, reason: str) -> None:
        """Clear retained trajectory and expose a specific unavailable reason."""
        self._positions = np.zeros((0, 3), dtype=float)
        self._panel.set_miss_unavailable(reason)

    def set_simulation_trajectory(self, positions_m: np.ndarray) -> None:
        """Adopt a simulation flight or expose the fixed-ball miss state."""
        if len(positions_m):
            self.set_trajectory(positions_m)
        else:
            self.set_unavailable("no flight exists because impact was not detected")

    def _on_target_changed(self, target: SpatialTarget) -> None:
        self._view.set_spatial_target(target)
        if target.kind == "landing_area":
            self._view.set_course_layout(layout_for_region(target.to_target_region()))
        self._refresh_miss()

    def _refresh_miss(self) -> None:
        if not len(self._positions):
            self._panel.set_miss_unavailable("run a flight first")
            return
        if not self._panel.is_valid():
            self._panel.set_miss_unavailable(
                "correct invalid target entries; the plot remains at the last "
                "valid target"
            )
            return
        target = self._panel.target()
        miss = trajectory_target_miss(target, self._positions)
        self._panel.set_miss(miss, landing=target.kind == "landing_area")


def build_spatial_target_workflow(
    view: FlightView,
) -> tuple[SpatialTargetPanel, SpatialTargetWorkflow]:
    """Construct the standard editor/controller pair for a flight view."""
    panel = SpatialTargetPanel()
    return panel, SpatialTargetWorkflow(panel, view)


__all__ = ["SpatialTargetWorkflow", "build_spatial_target_workflow"]
