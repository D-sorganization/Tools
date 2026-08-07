"""Spatial and legacy target coordination for the simulation session."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rate_of_closure.simulation.targets import TargetRegion, layout_for_region
from rate_of_closure.ui.pyqt6.flight_view import FlightView
from rate_of_closure.ui.pyqt6.simulation_view import SimulationView
from rate_of_closure.ui.pyqt6.spatial_target_panel import SpatialTargetPanel
from rate_of_closure.ui.pyqt6.spatial_target_workflow import (
    SpatialTargetWorkflow,
    build_spatial_target_workflow,
)
from shared.python.swing_sim.solver import SpatialTarget

if TYPE_CHECKING:
    from rate_of_closure.simulation import SimulationRun
    from rate_of_closure.ui.pyqt6.ball_setup_control import BallSetupControl
    from rate_of_closure.ui.pyqt6.inspector_view import InspectorView
    from rate_of_closure.ui.pyqt6.solver_panel import SolverPanel
    from shared.python.swing_sim.ball_setup import BallSetup


class SimulationTargetWorkflowMixin:
    """Own target controls, course projection, and run residual updates."""

    _flight_view: FlightView
    _ball_setup_control: BallSetupControl
    _inspector: InspectorView
    _solver_panel: SolverPanel
    _spatial_target_panel: SpatialTargetPanel
    _target_workflow: SpatialTargetWorkflow
    _view: SimulationView

    if TYPE_CHECKING:

        def _emit_config(self, *_args: object) -> None: ...

    def _build_spatial_target_control(self) -> SpatialTargetPanel:
        self._spatial_target_panel, self._target_workflow = (
            build_spatial_target_workflow(self._flight_view)
        )
        self._spatial_target_panel.targetChanged.connect(
            self._on_spatial_target_changed
        )
        self._on_spatial_target_changed(self._spatial_target_panel.target())
        return self._spatial_target_panel

    def _update_spatial_target_after_run(self, run: SimulationRun) -> None:
        self._target_workflow.set_simulation_trajectory(run.flight_positions)

    def _on_target_region_changed(self, region: TargetRegion) -> None:
        """Lift a legacy solver edit into the canonical spatial editor."""
        self._spatial_target_panel.set_target(SpatialTarget.from_target_region(region))

    def _on_spatial_target_changed(self, target: SpatialTarget) -> None:
        """Keep the landing solver/course projection aligned when possible."""
        self._inspector.set_spatial_target(target)
        target_panel = self._solver_panel.target_panel()
        landing = target.kind == "landing_area"
        target_panel.setEnabled(landing)
        if not landing:
            target_panel.setToolTip(
                "Aerial waypoints are assessed against the flight trajectory. "
                "The legacy landing optimizer accepts course-surface targets only."
            )
            return
        region = target.to_target_region()
        target_panel.setToolTip("")
        target_panel.set_region(region, emit=False)
        layout = layout_for_region(region)
        self._flight_view.set_target_region(region)
        self._view.set_course_layout(layout)

    def _on_imported_simulation_settings(
        self, setup: BallSetup, target: SpatialTarget
    ) -> None:
        """Apply a fully parsed project atomically at the UI boundary."""
        from rate_of_closure.ui.pyqt6.spatial_target_trajectory import (
            validate_landing_surface,
        )
        from shared.python.swing_sim.ball_setup import BallSetup as BallSetupType

        if not isinstance(setup, BallSetupType):
            raise TypeError("setup must be a BallSetup")
        if not isinstance(target, SpatialTarget):
            raise TypeError("target must be a SpatialTarget")
        validate_landing_surface(target)
        self._spatial_target_panel.set_target(target)
        self._ball_setup_control.set_setup(setup)
        self._emit_config()

    def set_landing_scatter(
        self, carries_m: np.ndarray | None, laterals_m: np.ndarray | None = None
    ) -> None:
        """Forward variation landing cohorts to the flight-scale view."""
        self._flight_view.set_landing_scatter(carries_m, laterals_m)


__all__ = ["SimulationTargetWorkflowMixin"]
